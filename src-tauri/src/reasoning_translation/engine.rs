use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::Tensor;
use rust_tokenizers::tokenizer::{MarianTokenizer, Tokenizer, TruncationStrategy};
use serde_json::Value;
use thiserror::Error;

use super::model::TranslationModelManager;

const D_MODEL: usize = 512;
const MAX_TOKENS_PER_SEGMENT: usize = 510;
const SENTENCE_BOUNDARIES: &[char] = &['.', '!', '?', '。', '！', '？', '\n'];
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = 64;

#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("model not ready: {0}")]
    NotReady(String),
    #[error("model load failed: {0}")]
    Load(String),
    #[error("inference failed: {0}")]
    Inference(String),
}

impl From<ort::Error> for TranslationError {
    fn from(e: ort::Error) -> Self {
        TranslationError::Inference(e.to_string())
    }
}

pub trait TranslationEngine: Send + Sync {
    fn translate_segments(&self, segments: &[String]) -> Result<Vec<String>, TranslationError>;
    fn reset(&self) {}
    fn unload_if_idle(&self, _timeout: Duration) {}
}

struct LoadedInner {
    encoder: Session,
    decoder: Session,
    decoder_with_past: Session,
    tokenizer: MarianTokenizer,
    decoder_start: i64,
    eos: i64,
    pad: i64,
    max_len: usize,
}

pub struct OnnxMarianEngine {
    model: Arc<TranslationModelManager>,
    inner: Mutex<Option<LoadedInner>>,
    last_used: Mutex<Instant>,
}

impl OnnxMarianEngine {
    pub fn new(model: Arc<TranslationModelManager>) -> Arc<Self> {
        Arc::new(Self {
            model,
            inner: Mutex::new(None),
            last_used: Mutex::new(Instant::now()),
        })
    }

    fn ensure_loaded(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<LoadedInner>>, TranslationError> {
        let mut guard = self.inner.lock().expect("engine lock");
        if guard.is_none() {
            if !matches!(
                self.model.status(),
                super::model::TranslationModelStatus::Ready { .. }
            ) {
                return Err(TranslationError::NotReady(
                    "model is not downloaded".to_string(),
                ));
            }
            let dir = self.model.model_dir();
            *guard = Some(load_onnx(&dir)?);
        }
        Ok(guard)
    }

    fn translate_one(&self, inner: &mut LoadedInner, text: &str) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }

        let encoded = inner.tokenizer.encode(
            text,
            None,
            inner.max_len,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let src_ids = encoded.token_ids;
        let src_len = src_ids.len();
        if src_len == 0 {
            return Ok(String::new());
        }

        let encoder_out = inner
            .encoder
            .run(ort::inputs![
                "input_ids" => Tensor::from_array((vec![1usize, src_len], src_ids))?,
                "attention_mask" => {
                    Tensor::from_array((vec![1usize, src_len], vec![1i64; src_len]))?
                }
            ])
            .map_err(|e| TranslationError::Inference(e.to_string()))?;
        let hidden = encoder_out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TranslationError::Inference(e.to_string()))?
            .1
            .to_vec();

        // First decoder step (no cache yet): computes the cross-attention
        // keys/values for the whole source sentence, plus one generated token.
        let first_out = inner
            .decoder
            .run(ort::inputs![
                "input_ids" => {
                    Tensor::from_array((vec![1usize, 1usize], vec![inner.decoder_start]))?
                },
                "encoder_hidden_states" => {
                    Tensor::from_array((vec![1usize, src_len, D_MODEL], hidden))?
                },
                "encoder_attention_mask" => {
                    Tensor::from_array((vec![1usize, src_len], vec![1i64; src_len]))?
                },
            ])
            .map_err(|e| TranslationError::Inference(e.to_string()))?;
        let first_logits = first_out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TranslationError::Inference(e.to_string()))?
            .1;
        let mut tgt_ids: Vec<i64> =
            vec![inner.decoder_start, argmax_ignoring(first_logits, inner.pad)];

        // Cache layout from the first run:
        // for each layer i: present.{i}.decoder.key, present.{i}.decoder.value,
        // present.{i}.encoder.key, present.{i}.encoder.value.
        let mut dec_past: Vec<Vec<f32>> = Vec::with_capacity(NUM_LAYERS * 2);
        let mut enc_past: Vec<Vec<f32>> = Vec::with_capacity(NUM_LAYERS * 2);
        for i in 0..NUM_LAYERS {
            dec_past.push(
                first_out[1 + i * 4]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| TranslationError::Inference(e.to_string()))?
                    .1
                    .to_vec(),
            );
            dec_past.push(
                first_out[2 + i * 4]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| TranslationError::Inference(e.to_string()))?
                    .1
                    .to_vec(),
            );
            enc_past.push(
                first_out[3 + i * 4]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| TranslationError::Inference(e.to_string()))?
                    .1
                    .to_vec(),
            );
            enc_past.push(
                first_out[4 + i * 4]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| TranslationError::Inference(e.to_string()))?
                    .1
                    .to_vec(),
            );
        }

        if *tgt_ids.last().expect("start token") != inner.eos {
            for _ in 1..inner.max_len {
                let past_len = dec_past[0].len() / (NUM_HEADS * HEAD_DIM);
                let last = *tgt_ids.last().expect("last token");
                let mut inputs = ort::inputs![
                    "input_ids" => Tensor::from_array((vec![1usize, 1usize], vec![last]))?,
                    "encoder_attention_mask" => {
                        Tensor::from_array((vec![1usize, src_len], vec![1i64; src_len]))?
                    },
                ];
                for i in 0..NUM_LAYERS {
                    inputs.push((
                        format!("past_key_values.{i}.decoder.key").into(),
                        Tensor::from_array((
                            vec![1usize, NUM_HEADS, past_len, HEAD_DIM],
                            dec_past[i * 2].clone(),
                        ))?
                        .into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{i}.decoder.value").into(),
                        Tensor::from_array((
                            vec![1usize, NUM_HEADS, past_len, HEAD_DIM],
                            dec_past[i * 2 + 1].clone(),
                        ))?
                        .into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{i}.encoder.key").into(),
                        Tensor::from_array((
                            vec![1usize, NUM_HEADS, src_len, HEAD_DIM],
                            enc_past[i * 2].clone(),
                        ))?
                        .into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{i}.encoder.value").into(),
                        Tensor::from_array((
                            vec![1usize, NUM_HEADS, src_len, HEAD_DIM],
                            enc_past[i * 2 + 1].clone(),
                        ))?
                        .into(),
                    ));
                }

                let out = inner
                    .decoder_with_past
                    .run(inputs)
                    .map_err(|e| TranslationError::Inference(e.to_string()))?;
                let logits = out[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| TranslationError::Inference(e.to_string()))?
                    .1;
                let next = argmax_ignoring(logits, inner.pad);
                for i in 0..NUM_LAYERS {
                    dec_past[i * 2] = out[1 + i * 2]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| TranslationError::Inference(e.to_string()))?
                        .1
                        .to_vec();
                    dec_past[i * 2 + 1] = out[2 + i * 2]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| TranslationError::Inference(e.to_string()))?
                        .1
                        .to_vec();
                }
                tgt_ids.push(next);
                if next == inner.eos {
                    break;
                }
            }
        }

        Ok(inner.tokenizer.decode(tgt_ids, true, true))
    }
}

impl TranslationEngine for OnnxMarianEngine {
    fn translate_segments(&self, segments: &[String]) -> Result<Vec<String>, TranslationError> {
        let mut guard = self.ensure_loaded()?;
        let inner = guard.as_mut().expect("loaded above");
        *self.last_used.lock().expect("last used lock") = Instant::now();
        let mut out = Vec::with_capacity(segments.len());
        for segment in segments {
            let translated = if segment.chars().count() > MAX_TOKENS_PER_SEGMENT * 4 {
                split_long_segment(segment)
                    .into_iter()
                    .map(|part| self.translate_one(inner, &part))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ")
            } else {
                self.translate_one(inner, segment)?
            };
            out.push(translated);
        }
        Ok(out)
    }

    fn reset(&self) {
        *self.inner.lock().expect("engine lock") = None;
    }

    fn unload_if_idle(&self, timeout: Duration) {
        let idle_for = self.last_used.lock().expect("last used lock").elapsed();
        if idle_for >= timeout {
            self.reset();
        }
    }
}

fn load_onnx(dir: &Path) -> Result<LoadedInner, TranslationError> {
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| TranslationError::Load(e.to_string()))?,
    )
    .map_err(|e| TranslationError::Load(e.to_string()))?;
    let decoder_start = cfg["decoder_start_token_id"]
        .as_i64()
        .ok_or_else(|| TranslationError::Load("missing decoder_start_token_id".into()))?;
    let eos = cfg["eos_token_id"]
        .as_i64()
        .ok_or_else(|| TranslationError::Load("missing eos_token_id".into()))?;
    let pad = cfg["pad_token_id"]
        .as_i64()
        .ok_or_else(|| TranslationError::Load("missing pad_token_id".into()))?;
    let max_len = cfg["max_length"].as_i64().unwrap_or(512) as usize;

    let tokenizer = MarianTokenizer::from_files(
        &dir.join("vocab.json").to_string_lossy(),
        &dir.join("source.spm").to_string_lossy(),
        false,
    )
    .map_err(|e| TranslationError::Load(e.to_string()))?;
    let encoder = Session::builder()
        .and_then(|mut b| b.commit_from_file(dir.join("onnx/encoder_model.onnx")))
        .map_err(|e| TranslationError::Load(e.to_string()))?;
    let decoder = Session::builder()
        .and_then(|mut b| b.commit_from_file(dir.join("onnx/decoder_model.onnx")))
        .map_err(|e| TranslationError::Load(e.to_string()))?;
    let decoder_with_past = Session::builder()
        .and_then(|mut b| b.commit_from_file(dir.join("onnx/decoder_with_past_model.onnx")))
        .map_err(|e| TranslationError::Load(e.to_string()))?;

    Ok(LoadedInner {
        encoder,
        decoder,
        decoder_with_past,
        tokenizer,
        decoder_start,
        eos,
        pad,
        max_len,
    })
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

/// Split an over-long segment at sentence boundaries so each chunk stays under
/// the model's token budget. Exposed for unit testing.
fn split_long_segment(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if SENTENCE_BOUNDARIES.contains(&ch) && current.chars().count() >= 200 {
            parts.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    if parts.is_empty() {
        parts.push(text.to_string());
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::split_long_segment;

    #[test]
    fn long_segment_splits_at_sentence_boundaries() {
        let text = "This is a fairly long sentence that keeps going to make sure the chunk gets big enough. "
            .repeat(5)
            + "A short tail.";
        let parts = split_long_segment(&text);
        assert!(parts.len() >= 2);
        assert_eq!(parts.concat(), text);
    }
}
