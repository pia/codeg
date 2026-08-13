use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex as AsyncMutex;

use crate::app_error::AppCommandError;
use crate::network::download::{download_file_with_progress, DownloadError};

use super::model_manifest::{
    file_url, MODEL_FILES, MODEL_REVISION, MIRROR_BASE_URL, PRIMARY_BASE_URL,
};

/// Per-file download timeout. A slow-but-progressing 200MB file on a weak
/// connection can exceed this; 180s is a pragmatic "too slow" bound that also
/// aborts stalled connections.
const FILE_DOWNLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TranslationModelStatus {
    NotDownloaded,
    Downloading {
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    Ready {
        revision: String,
    },
    Failed {
        message: String,
    },
}

#[derive(Clone)]
struct Manifest {
    revision: String,
    primary: String,
    mirror: String,
    files: Vec<(String, String, u64)>,
}

pub struct TranslationModelManager {
    data_dir: PathBuf,
    manifest: Manifest,
    state: Arc<Mutex<TranslationModelStatus>>,
    cancel_requested: Arc<AtomicBool>,
    download_lock: Arc<AsyncMutex<()>>,
}

impl Clone for TranslationModelManager {
    fn clone(&self) -> Self {
        Self {
            data_dir: self.data_dir.clone(),
            manifest: self.manifest.clone(),
            state: self.state.clone(),
            cancel_requested: self.cancel_requested.clone(),
            download_lock: self.download_lock.clone(),
        }
    }
}

impl TranslationModelManager {
    pub fn new(data_dir: PathBuf) -> Arc<Self> {
        let manager = Arc::new(Self {
            data_dir,
            manifest: Manifest {
                revision: MODEL_REVISION.to_string(),
                primary: PRIMARY_BASE_URL.to_string(),
                mirror: MIRROR_BASE_URL.to_string(),
                files: MODEL_FILES
                    .iter()
                    .map(|(file, hash, size)| {
                        ((*file).to_string(), (*hash).to_string(), *size)
                    })
                    .collect(),
            },
            state: Arc::new(Mutex::new(TranslationModelStatus::NotDownloaded)),
            cancel_requested: Arc::new(AtomicBool::new(false)),
            download_lock: Arc::new(AsyncMutex::new(())),
        });
        manager.sweep_trash();
        manager.refresh_status_from_disk();
        manager
    }

    #[cfg(test)]
    fn new_with_manifest_for_test(
        data_dir: PathBuf,
        primary: String,
        mirror: String,
        files: Vec<(&str, String)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            data_dir,
            manifest: Manifest {
                revision: "rev".to_string(),
                primary,
                mirror,
                files: files
                    .into_iter()
                    .map(|(file, hash)| (file.to_string(), hash, 0))
                    .collect(),
            },
            state: Arc::new(Mutex::new(TranslationModelStatus::NotDownloaded)),
            cancel_requested: Arc::new(AtomicBool::new(false)),
            download_lock: Arc::new(AsyncMutex::new(())),
        })
    }

    /// Test-only: build a manager whose model dir already contains one valid
    /// file, so services can translate without any network traffic.
    #[cfg(test)]
    pub(crate) fn new_for_test_ready(data_dir: PathBuf) -> Arc<Self> {
        let content = b"abc";
        let hash: String = {
            use sha2::Digest as _;
            let mut hasher = Sha256::new();
            hasher.update(content);
            hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        };
        let mgr = Self::new_with_manifest_for_test(
            data_dir,
            "http://primary.invalid".to_string(),
            "http://mirror.invalid".to_string(),
            vec![("a.bin", hash)],
        );
        let dir = mgr.model_dir();
        std::fs::create_dir_all(&dir).expect("create test model dir");
        std::fs::write(dir.join("a.bin"), content).expect("write test model file");
        mgr.set_status(TranslationModelStatus::Ready {
            revision: mgr.manifest.revision.clone(),
        });
        mgr
    }

    pub fn status(&self) -> TranslationModelStatus {
        self.refresh_status_from_disk();
        self.state.lock().expect("model status lock").clone()
    }

    pub fn model_dir(&self) -> PathBuf {
        self.data_dir
            .join("models")
            .join("reasoning-translation")
            .join(&self.manifest.revision)
    }

    fn base_dir(&self) -> PathBuf {
        self.data_dir.join("models").join("reasoning-translation")
    }

    fn set_status(&self, status: TranslationModelStatus) {
        *self.state.lock().expect("model status lock") = status;
    }

    /// Promote `NotDownloaded` to `Ready` when a valid model directory is
    /// already on disk — pre-seeded installs, upgrades, or files placed while
    /// the app was closed would otherwise be reported as missing forever.
    fn refresh_status_from_disk(&self) {
        let is_missing = {
            let guard = self.state.lock().expect("model status lock");
            matches!(&*guard, TranslationModelStatus::NotDownloaded)
        };
        if is_missing && self.verify_dir(&self.model_dir()) {
            self.set_status(TranslationModelStatus::Ready {
                revision: self.manifest.revision.clone(),
            });
        }
    }

    fn update_progress(&self, downloaded_bytes: u64, total_bytes: Option<u64>) {
        self.set_status(TranslationModelStatus::Downloading {
            downloaded_bytes,
            total_bytes,
        });
    }

    fn verify_file(&self, path: &Path, expected: &str) -> bool {
        let Ok(mut file) = std::fs::File::open(path) else {
            return false;
        };
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            use std::io::Read;
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => hasher.update(&buf[..n]),
                Err(_) => return false,
            }
        }
        let actual = hasher.finalize();
        let actual_hex: String = actual.iter().map(|b| format!("{b:02x}")).collect();
        actual_hex.eq_ignore_ascii_case(expected)
    }

    fn verify_dir(&self, dir: &Path) -> bool {
        if !dir.is_dir() {
            return false;
        }
        self.manifest.files.iter().all(|(file, hash, _)| {
            let path = dir.join(file);
            path.is_file() && self.verify_file(&path, hash)
        })
    }

    /// Start a download in the background. The caller gets the current status
    /// immediately; progress/terminal states are observable via `status()`.
    pub fn begin_download(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let _ = manager.start_download().await;
        });
    }

    /// Download the pinned model, verifying SHA-256 per file. Primary source
    /// first, mirror on failure/timeout. Single-flight: concurrent callers
    /// wait on the same lock and reuse the result.
    pub async fn start_download(&self) -> Result<PathBuf, AppCommandError> {
        let _guard = self.download_lock.lock().await;

        let final_dir = self.model_dir();
        if self.verify_dir(&final_dir) {
            self.set_status(TranslationModelStatus::Ready {
                revision: self.manifest.revision.clone(),
            });
            return Ok(final_dir);
        }

        self.cancel_requested.store(false, Ordering::SeqCst);
        let base = self.base_dir();
        std::fs::create_dir_all(&base).map_err(|e| {
            AppCommandError::io_error("Failed to create model directory").with_detail(e.to_string())
        })?;

        let tmp = base.join(format!(".tmp-{}", now_millis()));
        std::fs::create_dir_all(&tmp).map_err(|e| {
            AppCommandError::io_error("Failed to create model temp directory")
                .with_detail(e.to_string())
        })?;

        // Aggregate progress across all files so the UI shows one stable
        // total (~427MB for the pinned model) instead of jumping per file.
        let total_size = self
            .manifest
            .files
            .iter()
            .try_fold(0u64, |acc, (_, _, size)| {
                (*size > 0).then(|| acc + *size)
            });
        let mut aggregate_downloaded: u64 = 0;

        let result: Result<PathBuf, String> = async {
            for (file, expected, size) in &self.manifest.files {
                let dest = tmp.join(file);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        format!("failed to create {}: {e}", parent.display())
                    })?;
                }

                let mut downloaded_ok = false;
                let mut last_downloaded: u64 = 0;
                for base_url in [&self.manifest.primary, &self.manifest.mirror] {
                    if self.cancel_requested.load(Ordering::SeqCst) {
                        return Err("cancelled".to_string());
                    }
                    let url = file_url(base_url, &self.manifest.revision, file);
                    let cancel = self.cancel_requested.clone();
                    let completed_bytes = aggregate_downloaded;
                    let outcome = tokio::time::timeout(
                        FILE_DOWNLOAD_TIMEOUT,
                        download_file_with_progress(
                            &url,
                            &dest,
                            move || cancel.load(Ordering::SeqCst),
                            |downloaded, _file_total| {
                                last_downloaded = downloaded;
                                self.update_progress(
                                    completed_bytes + downloaded,
                                    total_size,
                                );
                            },
                        ),
                    )
                    .await;

                    match outcome {
                        Ok(Ok(())) if self.verify_file(&dest, expected) => {
                            aggregate_downloaded +=
                                if *size > 0 { *size } else { last_downloaded };
                            downloaded_ok = true;
                            break;
                        }
                        Ok(Ok(())) => {
                            let _ = std::fs::remove_file(&dest);
                        }
                        Ok(Err(DownloadError::Cancelled)) => {
                            return Err("cancelled".to_string());
                        }
                        Ok(Err(e)) => {
                            let _ = std::fs::remove_file(&dest);
                            tracing::warn!("[ReasoningTranslation] download failed: {e}");
                        }
                        Err(_) => {
                            let _ = std::fs::remove_file(&dest);
                            tracing::warn!("[ReasoningTranslation] download timed out: {url}");
                        }
                    }
                }
                if !downloaded_ok {
                    return Err(format!("failed to download {file} from all sources"));
                }
            }

            if final_dir.exists() {
                let trash = base.join(format!(".trash-{}", now_millis()));
                std::fs::rename(&final_dir, &trash)
                    .map_err(|e| format!("failed to move stale model dir: {e}"))?;
            }
            std::fs::rename(&tmp, &final_dir)
                .map_err(|e| format!("failed to finalize model dir: {e}"))?;
            Ok(final_dir.clone())
        }
        .await;

        match result {
            Ok(dir) => {
                self.set_status(TranslationModelStatus::Ready {
                    revision: self.manifest.revision.clone(),
                });
                Ok(dir)
            }
            Err(message) => {
                let _ = std::fs::remove_dir_all(&tmp);
                if self.cancel_requested.load(Ordering::SeqCst) {
                    self.set_status(TranslationModelStatus::NotDownloaded);
                } else {
                    self.set_status(TranslationModelStatus::Failed {
                        message: message.clone(),
                    });
                }
                Err(AppCommandError::network("Translation model download failed")
                    .with_detail(message))
            }
        }
    }

    pub fn cancel_download(&self) {
        self.cancel_requested.store(true, Ordering::SeqCst);
    }

    /// Remove the downloaded model. The translation service must reset its
    /// engine first so no ONNX session keeps the files open.
    pub fn delete_model(&self) -> Result<(), AppCommandError> {
        let dir = self.model_dir();
        if !dir.exists() {
            self.set_status(TranslationModelStatus::NotDownloaded);
            return Ok(());
        }
        if std::fs::remove_dir_all(&dir).is_ok() {
            self.set_status(TranslationModelStatus::NotDownloaded);
            return Ok(());
        }

        // Windows may hold file handles; rename aside and sweep next start.
        let trash = self
            .base_dir()
            .join(format!(".trash-delete-{}", now_millis()));
        std::fs::rename(&dir, &trash).map_err(|e| {
            AppCommandError::io_error("Failed to remove translation model").with_detail(e.to_string())
        })?;
        let _ = std::fs::remove_dir_all(&trash);
        self.set_status(TranslationModelStatus::NotDownloaded);
        Ok(())
    }

    fn sweep_trash(&self) {
        let Ok(entries) = std::fs::read_dir(self.base_dir()) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if (name.starts_with(".trash-") || name.starts_with(".tmp-")) && entry.path().is_dir() {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use axum::{http::StatusCode, routing::get, Router};
    use axum_test::TestServer;

    use super::*;

    fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    async fn server_for(route: &str, body: String) -> TestServer {
        TestServer::builder()
            .http_transport()
            .build(Router::new().route(
                route,
                get(move || {
                    let body = body.clone();
                    async move { body }
                }),
            ))
            .unwrap()
    }

    async fn failing_server(route: &str) -> TestServer {
        TestServer::builder()
            .http_transport()
            .build(Router::new().route(
                route,
                get(|| async { StatusCode::NOT_FOUND }),
            ))
            .unwrap()
    }

    async fn slow_server(route: &str, body: String) -> TestServer {
        TestServer::builder()
            .http_transport()
            .build(Router::new().route(
                route,
                get(move || async move {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    body
                }),
            ))
            .unwrap()
    }

    fn manager_for(
        dir: &Path,
        primary: &str,
        mirror: &str,
        files: Vec<(&str, String)>,
    ) -> Arc<TranslationModelManager> {
        TranslationModelManager::new_with_manifest_for_test(
            dir.to_path_buf(),
            primary.to_string(),
            mirror.to_string(),
            files,
        )
    }

    #[tokio::test]
    async fn falls_back_to_mirror_when_primary_fails() {
        let files = vec![("a.bin", sha256_hex(b"abc"))];
        let primary = failing_server("/resolve/rev/a.bin").await;
        let mirror = server_for("/resolve/rev/a.bin", "abc".to_string()).await;
        let dir = tempfile::tempdir().unwrap();
        let mgr = manager_for(
            dir.path(),
            primary.server_url("/").unwrap().as_str(),
            mirror.server_url("/").unwrap().as_str(),
            files,
        );

        let path = mgr.start_download().await.unwrap();
        assert!(path.join("a.bin").is_file());
        assert!(matches!(mgr.status(), TranslationModelStatus::Ready { .. }));
    }

    #[tokio::test]
    async fn checksum_mismatch_fails_and_cleans_temp() {
        let files = vec![("a.bin", sha256_hex(b"xyz"))];
        let primary = server_for("/resolve/rev/a.bin", "abc".to_string()).await;
        let mirror = server_for("/resolve/rev/a.bin", "abc".to_string()).await;
        let dir = tempfile::tempdir().unwrap();
        let mgr = manager_for(
            dir.path(),
            primary.server_url("/").unwrap().as_str(),
            mirror.server_url("/").unwrap().as_str(),
            files,
        );

        let _ = mgr.start_download().await;
        assert!(matches!(
            mgr.status(),
            TranslationModelStatus::Failed { message } if message.contains("failed to download")
        ));
        let leftovers = std::fs::read_dir(mgr.base_dir()).unwrap().count();
        assert_eq!(leftovers, 0, "temp dirs must be cleaned up");
    }

    #[tokio::test]
    async fn cancel_cleans_up_and_resets_status() {
        let files = vec![("a.bin", sha256_hex(b"x"))];
        let primary = slow_server("/resolve/rev/a.bin", "x".repeat(1024)).await;
        let mirror = server_for("/resolve/rev/a.bin", "abc".to_string()).await;
        let dir = tempfile::tempdir().unwrap();
        let mgr = manager_for(
            dir.path(),
            primary.server_url("/").unwrap().as_str(),
            mirror.server_url("/").unwrap().as_str(),
            files,
        );

        let task = tokio::spawn({
            let mgr = mgr.clone();
            async move { mgr.start_download().await }
        });
        for _ in 0..100 {
            if matches!(mgr.status(), TranslationModelStatus::Downloading { .. }) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        mgr.cancel_download();
        let _ = task.await.unwrap();
        assert!(matches!(mgr.status(), TranslationModelStatus::NotDownloaded));
        let leftovers = std::fs::read_dir(mgr.base_dir()).unwrap().count();
        assert_eq!(leftovers, 0);
    }

    #[tokio::test]
    async fn delete_model_removes_ready_dir() {
        let files = vec![("a.bin", sha256_hex(b"abc"))];
        let primary = failing_server("/resolve/rev/a.bin").await;
        let mirror = server_for("/resolve/rev/a.bin", "abc".to_string()).await;
        let dir = tempfile::tempdir().unwrap();
        let mgr = manager_for(
            dir.path(),
            primary.server_url("/").unwrap().as_str(),
            mirror.server_url("/").unwrap().as_str(),
            files,
        );
        mgr.start_download().await.unwrap();
        assert!(mgr.model_dir().is_dir());

        mgr.delete_model().unwrap();
        assert!(!mgr.model_dir().exists());
        assert!(matches!(mgr.status(), TranslationModelStatus::NotDownloaded));
    }

    #[tokio::test]
    async fn status_reflects_pre_seeded_files_without_download() {
        let files = vec![("a.bin", sha256_hex(b"abc"))];
        let dir = tempfile::tempdir().unwrap();
        let mgr = TranslationModelManager::new_with_manifest_for_test(
            dir.path().to_path_buf(),
            "http://primary.invalid".to_string(),
            "http://mirror.invalid".to_string(),
            files,
        );
        let model_dir = mgr.model_dir();
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("a.bin"), b"abc").unwrap();

        assert!(matches!(
            mgr.status(),
            TranslationModelStatus::Ready { .. }
        ));
    }
}
