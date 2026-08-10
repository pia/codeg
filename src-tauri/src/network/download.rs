use std::io::Write;
use std::path::Path;

use futures_util::StreamExt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("HTTP {status} for {url}")]
    Http { status: u16, url: String },
    #[error("request failed: {0}")]
    Request(String),
    #[error("cancelled")]
    Cancelled,
    #[error("io error: {0}")]
    Io(String),
}

pub async fn download_file_with_progress(
    url: &str,
    dest: &Path,
    is_cancelled: impl Fn() -> bool,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<(), DownloadError> {
    let response = reqwest::Client::new()
        .get(url)
        .send()
        .await
        .map_err(|e| DownloadError::Request(e.to_string()))?;

    if !response.status().is_success() {
        return Err(DownloadError::Http {
            status: response.status().as_u16(),
            url: url.to_string(),
        });
    }

    let total = response.content_length();
    let mut file = std::fs::File::create(dest).map_err(|e| DownloadError::Io(e.to_string()))?;
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    loop {
        if is_cancelled() {
            return Err(DownloadError::Cancelled);
        }
        let Some(chunk) = stream.next().await else {
            break;
        };
        let chunk = chunk.map_err(|e| DownloadError::Request(e.to_string()))?;
        file.write_all(&chunk)
            .map_err(|e| DownloadError::Io(e.to_string()))?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use axum::{routing::get, Router};
    use axum_test::TestServer;

    use super::{download_file_with_progress, DownloadError};

    const BLOB: &str = "hello";

    async fn test_server() -> TestServer {
        TestServer::builder()
            .http_transport()
            .build(Router::new().route("/blob", get(|| async { BLOB })))
            .unwrap()
    }

    #[tokio::test]
    async fn writes_body_and_reports_final_size() {
        let srv = test_server().await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.bin");
        let mut final_total = None;
        download_file_with_progress(
            &srv.server_url("/blob").unwrap().to_string(),
            &dest,
            || false,
            |_, total| final_total = total,
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), BLOB);
        assert_eq!(final_total, Some(BLOB.len() as u64));
    }

    #[tokio::test]
    async fn cancel_stops_with_cancelled_error() {
        let srv = TestServer::builder()
            .http_transport()
            .build(Router::new().route(
                "/chunks",
                get(|| async { "x".repeat(10 * 1024 * 1024) }),
            ))
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.bin");
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_for_progress = cancel.clone();
        let err = download_file_with_progress(
            &srv.server_url("/chunks").unwrap().to_string(),
            &dest,
            move || cancel.load(Ordering::SeqCst),
            move |_, _| {
                cancel_for_progress.store(true, Ordering::SeqCst);
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, DownloadError::Cancelled));
    }
}
