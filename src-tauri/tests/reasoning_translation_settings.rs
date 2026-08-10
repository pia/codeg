//! Integration tests for reasoning translation settings (get/update roundtrip).

use std::sync::Arc;

use axum_test::TestServer;
use codeg_lib::app_state::AppState;
use codeg_lib::db::test_helpers::fresh_in_memory_db;
use codeg_lib::web::router::build_router;
use codeg_lib::web::shutdown::ShutdownSignal;
use serde_json::{json, Value};

const TEST_TOKEN: &str = "integration-test-token";

async fn build_test_server() -> (TestServer, tempfile::TempDir, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data dir");
    let static_dir = tempfile::tempdir().expect("static dir");

    let db = fresh_in_memory_db().await;
    let state = Arc::new(AppState::new_for_test(db, data_dir.path().to_path_buf()));
    let shutdown = Arc::new(ShutdownSignal::new());

    let router = build_router(
        state,
        TEST_TOKEN.to_string(),
        static_dir.path().to_path_buf(),
        shutdown,
    );

    let server = TestServer::new(router).expect("test server");
    (server, data_dir, static_dir)
}

fn auth() -> (String, String) {
    ("authorization".to_string(), format!("Bearer {TEST_TOKEN}"))
}

#[tokio::test]
async fn settings_default_to_disabled_zh_cn() {
    let (server, _data, _static) = build_test_server().await;
    let resp = server
        .post("/api/get_reasoning_translation_settings")
        .add_header(auth().0, auth().1)
        .json(&json!({}))
        .await;
    assert_eq!(resp.status_code(), 200);
    let value: Value = resp.json();
    assert_eq!(value["enabled"], false);
    assert_eq!(value["target_language"], "zh-CN");
}

#[tokio::test]
async fn update_persists_enabled() {
    let (server, _data, _static) = build_test_server().await;
    let resp = server
        .post("/api/update_reasoning_translation_settings")
        .add_header(auth().0, auth().1)
        .json(&json!({
            "settings": { "enabled": true, "target_language": "zh-CN" }
        }))
        .await;
    assert_eq!(resp.status_code(), 200);

    let get = server
        .post("/api/get_reasoning_translation_settings")
        .add_header(auth().0, auth().1)
        .json(&json!({}))
        .await;
    assert_eq!(get.status_code(), 200);
    assert_eq!(get.json::<Value>()["enabled"], true);
}
