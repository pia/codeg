use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex as AsyncMutex;

use crate::app_error::AppCommandError;

use super::engine::{TranslationEngine, TranslationError};
use super::model::{TranslationModelManager, TranslationModelStatus};

const IDLE_UNLOAD_TIMEOUT: Duration = Duration::from_secs(10 * 60);

pub struct TranslationService {
    engine: Arc<dyn TranslationEngine>,
    model: Arc<TranslationModelManager>,
    queue: Arc<AsyncMutex<()>>,
}

impl TranslationService {
    pub fn new(
        engine: Arc<dyn TranslationEngine>,
        model: Arc<TranslationModelManager>,
    ) -> Arc<Self> {
        let service = Arc::new(Self {
            engine,
            model,
            queue: Arc::new(AsyncMutex::new(())),
        });
        let service_for_task = service.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(IDLE_UNLOAD_TIMEOUT).await;
                service_for_task
                    .engine
                    .unload_if_idle(IDLE_UNLOAD_TIMEOUT);
            }
        });
        service
    }

    /// Serialized translation: one inference at a time, model downloaded on
    /// first use. Stale streaming requests are the frontend's responsibility.
    pub async fn translate(
        &self,
        segments: Vec<String>,
    ) -> Result<Vec<String>, AppCommandError> {
        let _guard = self.queue.lock().await;
        if !matches!(self.model.status(), TranslationModelStatus::Ready { .. }) {
            self.model.start_download().await?;
        }
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || engine.translate_segments(&segments))
            .await
            .map_err(|e| {
                AppCommandError::task_execution_failed("Translation task failed")
                    .with_detail(e.to_string())
            })?
            .map_err(translation_error_to_command)
    }

    /// Delete the model while holding the queue lock so no inference is in
    /// flight, then drop engine sessions before removing files.
    pub async fn delete_model(&self) -> Result<(), AppCommandError> {
        let _guard = self.queue.lock().await;
        self.engine.reset();
        self.model.delete_model()
    }
}

fn translation_error_to_command(e: TranslationError) -> AppCommandError {
    AppCommandError::task_execution_failed("Translation failed").with_detail(e.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::reasoning_translation::model::TranslationModelManager;

    struct StubEngine {
        log: Arc<Mutex<Vec<String>>>,
        calls: Arc<AtomicUsize>,
    }

    impl StubEngine {
        fn new(log: Arc<Mutex<Vec<String>>>, calls: Arc<AtomicUsize>) -> Arc<Self> {
            Arc::new(Self { log, calls })
        }
    }

    impl TranslationEngine for StubEngine {
        fn translate_segments(&self, segments: &[String]) -> Result<Vec<String>, TranslationError> {
            self.log.lock().unwrap().push(format!("start-{}", self.calls.load(Ordering::SeqCst)));
            std::thread::sleep(std::time::Duration::from_millis(50));
            self.log.lock().unwrap().push(format!("end-{}", self.calls.load(Ordering::SeqCst)));
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(segments.iter().map(|s| format!("译:{s}")).collect())
        }

        fn reset(&self) {
            self.log.lock().unwrap().push("reset".to_string());
        }
    }

    async fn ready_model(dir: &Path) -> Arc<TranslationModelManager> {
        TranslationModelManager::new_for_test_ready(dir.to_path_buf())
    }

    #[tokio::test]
    async fn concurrent_requests_are_serialized() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let engine = StubEngine::new(log.clone(), calls.clone());
        let dir = tempfile::tempdir().unwrap();
        let model = ready_model(dir.path()).await;
        let service = TranslationService::new(engine, model.clone());

        let a = service.translate(vec!["one".to_string()]);
        let b = service.translate(vec!["two".to_string()]);
        let (ra, rb) = tokio::join!(a, b);
        assert_eq!(ra.unwrap(), vec!["译:one".to_string()]);
        assert_eq!(rb.unwrap(), vec!["译:two".to_string()]);

        let log = log.lock().unwrap();
        assert_eq!(log.len(), 4, "start/end pairs must not interleave");
        assert!(log[0].starts_with("start-"));
        assert!(log[1].starts_with("end-"));
        assert!(log[2].starts_with("start-"));
        assert!(log[3].starts_with("end-"));
    }

    #[tokio::test]
    async fn delete_model_resets_engine_then_removes_files() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let engine = StubEngine::new(log.clone(), calls.clone());
        let dir = tempfile::tempdir().unwrap();
        let model = ready_model(dir.path()).await;
        let service = TranslationService::new(engine, model.clone());

        service.delete_model().await.unwrap();
        assert_eq!(log.lock().unwrap().last().map(String::as_str), Some("reset"));
        assert!(!model.model_dir().exists());
    }
}
