#[cfg(feature = "tauri-runtime")]
use std::sync::Arc;

#[cfg(feature = "tauri-runtime")]
use crate::app_error::AppCommandError;
#[cfg(feature = "tauri-runtime")]
use crate::reasoning_translation::model::{TranslationModelManager, TranslationModelStatus};
#[cfg(feature = "tauri-runtime")]
use crate::reasoning_translation::service::TranslationService;

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn reasoning_translation_model_status(
    translation_model: tauri::State<'_, Arc<TranslationModelManager>>,
) -> Result<TranslationModelStatus, AppCommandError> {
    Ok(translation_model.status())
}

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn reasoning_translation_download_model(
    translation_model: tauri::State<'_, Arc<TranslationModelManager>>,
) -> Result<TranslationModelStatus, AppCommandError> {
    translation_model.begin_download();
    Ok(translation_model.status())
}

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn reasoning_translation_delete_model(
    translation_service: tauri::State<'_, Arc<TranslationService>>,
    translation_model: tauri::State<'_, Arc<TranslationModelManager>>,
) -> Result<TranslationModelStatus, AppCommandError> {
    translation_service.delete_model().await?;
    Ok(translation_model.status())
}

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn reasoning_translation_translate(
    segments: Vec<String>,
    translation_service: tauri::State<'_, Arc<TranslationService>>,
) -> Result<Vec<String>, AppCommandError> {
    translation_service.translate(segments).await
}
