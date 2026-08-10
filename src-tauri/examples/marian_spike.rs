//! Spike: validate the full Marian ONNX inference chain for opus-mt-en-zh.
//!
//! Model files live in tmp/spike-model (gitignored). Run:
//!   cargo run --example marian_spike --features test-utils

use std::fs;

use anyhow::Context;
use ort::session::Session;
use ort::value::Tensor;
use rust_tokenizers::tokenizer::{MarianTokenizer, Tokenizer, TruncationStrategy};
use serde_json::Value;

const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tmp/spike-model");
const VOCAB_SIZE: usize = 65001;
const D_MODEL: usize = 512;

fn main() -> anyhow::Result<()> {
    let cfg: Value = serde_json::from_str(&fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)
        .context("read config.json")?;
    let decoder_start = cfg["decoder_start_token_id"].as_i64().context("decoder_start")?;
    let eos = cfg["eos_token_id"].as_i64().context("eos")?;
    let pad = cfg["pad_token_id"].as_i64().context("pad")?;
    let max_len = cfg["max_length"].as_i64().unwrap_or(512) as usize;

    let tokenizer = MarianTokenizer::from_files(
        &format!("{MODEL_DIR}/vocab.json"),
        &format!("{MODEL_DIR}/source.spm"),
        false,
    )
    .context("load Marian tokenizer")?;

    let mut encoder =
        Session::builder()?.commit_from_file(format!("{MODEL_DIR}/onnx/encoder_model.onnx"))?;
    let mut decoder =
        Session::builder()?.commit_from_file(format!("{MODEL_DIR}/onnx/decoder_model.onnx"))?;

    let text = "The quick brown fox jumps over the lazy dog.";
    let encoded = tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
    let src_ids = encoded.token_ids;
    let src_len = src_ids.len();

    let encoder_out = encoder.run(ort::inputs![
        "input_ids" => Tensor::from_array((vec![1usize, src_len], src_ids))?,
        "attention_mask" => Tensor::from_array((vec![1usize, src_len], vec![1i64; src_len]))?,
    ])?;
    let hidden = encoder_out[0].try_extract_tensor::<f32>()?.1.to_vec();
    assert_eq!(hidden.len(), src_len * D_MODEL);

    let mut tgt_ids: Vec<i64> = vec![decoder_start];
    for _step in 0..max_len {
        let tgt_len = tgt_ids.len();
        let dec_out = decoder.run(ort::inputs![
            "input_ids" => Tensor::from_array((vec![1usize, tgt_len], tgt_ids.clone()))?,
            "encoder_hidden_states" => {
                Tensor::from_array((vec![1usize, src_len, D_MODEL], hidden.clone()))?
            },
            "encoder_attention_mask" => {
                Tensor::from_array((vec![1usize, src_len], vec![1i64; src_len]))?
            },
        ])?;
        let logits = dec_out[0].try_extract_tensor::<f32>()?.1;
        let row_start = (tgt_len - 1) * VOCAB_SIZE;
        let next = argmax_ignoring(&logits[row_start..row_start + VOCAB_SIZE], pad);
        if next == eos {
            break;
        }
        tgt_ids.push(next);
    }

    let translated = tokenizer.decode(tgt_ids, true, true);
    println!("src: {text}");
    println!("tgt: {translated}");
    Ok(())
}

fn argmax_ignoring(scores: &[f32], ignore: i64) -> i64 {
    let mut best = 0i64;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &score) in scores.iter().enumerate() {
        if i as i64 == ignore {
            continue;
        }
        if score > best_score {
            best_score = score;
            best = i as i64;
        }
    }
    best
}
