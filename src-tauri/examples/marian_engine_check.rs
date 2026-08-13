//! End-to-end check of the real translation engine with the downloaded model.
//!
//! Prepare a data dir containing `models/reasoning-translation/<rev>/` (the
//! layout `TranslationModelManager` expects), then run:
//!   CODEG_ENGINE_DATA_DIR=<dir> cargo run --example marian_engine_check

use codeg_lib::reasoning_translation::engine::OnnxMarianEngine;
use codeg_lib::reasoning_translation::model::TranslationModelManager;
use codeg_lib::reasoning_translation::service::TranslationService;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let data_dir = std::env::var("CODEG_ENGINE_DATA_DIR")
        .map_err(|_| anyhow::anyhow!("set CODEG_ENGINE_DATA_DIR to the prepared data dir"))?;
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".to_string());
    let model = TranslationModelManager::new(data_dir.into());
    let service = TranslationService::new(
        OnnxMarianEngine::new(model.clone()),
        model,
    );

    let started = std::time::Instant::now();
    let translated = service
        .translate(vec![text])
        .await
        .map_err(|e| anyhow::anyhow!("translate failed: {e}"))?;
    println!("{}", translated.join("\n"));
    eprintln!("elapsed: {:?}", started.elapsed());
    Ok(())
}
