use std::sync::Arc;

use axum::{extract::Extension, Json};
use serde::Deserialize;

use crate::app_error::AppCommandError;
use crate::app_state::AppState;
use crate::reasoning_translation::model::TranslationModelStatus;

pub async fn model_status(
    Extension(state): Extension<Arc<AppState>>,
) -> Result<Json<TranslationModelStatus>, AppCommandError> {
    Ok(Json(state.translation_model.status()))
}

pub async fn download_model(
    Extension(state): Extension<Arc<AppState>>,
) -> Result<Json<TranslationModelStatus>, AppCommandError> {
    state.translation_model.begin_download();
    Ok(Json(state.translation_model.status()))
}

pub async fn delete_model(
    Extension(state): Extension<Arc<AppState>>,
) -> Result<Json<TranslationModelStatus>, AppCommandError> {
    state.translation_service.delete_model().await?;
    Ok(Json(state.translation_model.status()))
}

#[derive(Deserialize)]
pub struct TranslateParams {
    pub segments: Vec<String>,
}

pub async fn translate(
    Extension(state): Extension<Arc<AppState>>,
    Json(params): Json<TranslateParams>,
) -> Result<Json<Vec<String>>, AppCommandError> {
    Ok(Json(
        state.translation_service.translate(params.segments).await?,
    ))
}
