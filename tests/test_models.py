from emr_extractor import models


def test_offline_model_environment_is_portable(tmp_path, monkeypatch):
    for name in (
        "PADDLE_PDX_CACHE_HOME",
        "MINERU_HOME",
        "MINERU_MODEL_SOURCE",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)

    models.configure_model_environment("all", tmp_path, offline=True)

    assert models.os.environ["PADDLE_PDX_CACHE_HOME"] == str(tmp_path / "paddleocr")
    assert models.os.environ["MINERU_HOME"] == str(tmp_path / "mineru")
    assert models.os.environ["MINERU_MODEL_SOURCE"] == "local"
    assert models.os.environ["HF_HUB_OFFLINE"] == "1"
    assert models.os.environ["PADDLE_PDX_MODEL_SOURCE"] == "huggingface"


def test_download_defaults_to_paddleocr():
    args = models.build_parser().parse_args([])
    assert args.backend == "paddleocr"


def test_mineru_download_is_verified(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(models, "_run_mineru", commands.append)

    models.download_mineru_models(
        tmp_path,
        tier="standard",
        source="modelscope",
        small_backend="onnx",
        vlm_engine="llama-cpp",
    )

    selection = [
        "--tier",
        "standard",
        "--small-backend",
        "onnx",
        "--vlm-engine",
        "llama-cpp",
    ]
    download_options = [
        *selection,
        "--source",
        "modelscope",
    ]
    assert commands == [
        ["mineru-kit", "models", "download", *download_options],
        ["mineru-kit", "models", "verify", *selection],
    ]
