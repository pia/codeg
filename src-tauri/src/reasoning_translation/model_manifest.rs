pub(crate) const MODEL_REVISION: &str = "ecf05f1f90e7ed45c6656b71e346a1468d20c999";
pub(crate) const PRIMARY_BASE_URL: &str = "https://huggingface.co/onnx-community/opus-mt-en-zh";
pub(crate) const MIRROR_BASE_URL: &str = "https://hf-mirror.com/onnx-community/opus-mt-en-zh";

/// (相对路径, SHA-256)。哈希来自 Task 1 spike 的实际下载。
pub(crate) const MODEL_FILES: &[(&str, &str)] = &[
    (
        "config.json",
        "461edc7f206ec48c8f16595b11dae8d0b75a9cede693d9fafb2ef682b065eace",
    ),
    (
        "tokenizer_config.json",
        "6700ad6f537313acb4c39a3056bbabe604cf6524f9d3b1c8e50985d38caebfb7",
    ),
    (
        "vocab.json",
        "22c957348eed495ee925afc40a36da3e387c8a34a734c8486967c2dca271613e",
    ),
    (
        "source.spm",
        "5775ddc9e3ff2fae91554da56468ad35ff56edaba870fea74447bc7234bfdaa8",
    ),
    (
        "target.spm",
        "81dc94efa84e4025ef38d25d5d07429fe41e3eb29d44003f1db6fe98487b0052",
    ),
    (
        "onnx/encoder_model.onnx",
        "f90588a301687184cc06b589c8399c66621bcb4858a96be17634114ff529c112",
    ),
    (
        "onnx/decoder_model.onnx",
        "679e1b2bcfcd4d8058546e9c3ecff49817ecf8d2ce9d2d1930ff094de1d2ef98",
    ),
];

pub(crate) fn file_url(base: &str, revision: &str, file: &str) -> String {
    format!(
        "{}/resolve/{}/{}",
        base.trim_end_matches('/'),
        revision,
        file
    )
}

#[cfg(test)]
mod tests {
    use super::MODEL_FILES;

    #[test]
    fn manifest_has_real_sha256_hashes() {
        assert!(MODEL_FILES.len() >= 7);
        for (file, hash) in MODEL_FILES {
            assert_eq!(hash.len(), 64, "{file} hash must be 64 hex chars");
            assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "{file} hash must be hex");
        }
        assert!(MODEL_FILES.iter().any(|(f, _)| *f == "onnx/encoder_model.onnx"));
        assert!(MODEL_FILES.iter().any(|(f, _)| *f == "onnx/decoder_model.onnx"));
    }
}
