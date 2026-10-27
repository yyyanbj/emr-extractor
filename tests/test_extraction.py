from pathlib import Path

from emr_extractor.extraction import DocumentExtractor, build_parser, iter_documents


class FakeResult:
    def __init__(self, name: str):
        self.name = name

    def save_to_markdown(self, save_path: str):
        output = Path(save_path) / f"{self.name}.md"
        output.write_text("# synthetic", encoding="utf-8")


class FakePipeline:
    def __init__(self):
        self.restructured = False

    def predict(self, input: str):
        return [FakeResult("page-1"), FakeResult("page-2")]

    def restructure_pages(self, results, concatenate_pages: bool):
        assert len(results) == 2
        assert concatenate_pages is True
        self.restructured = True
        return [FakeResult("record")]


class FakeMinerUResult:
    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "markdown.md").write_text("# synthetic MinerU", encoding="utf-8")
        (output_dir / "images").mkdir()


def test_document_extractor_restructures_multi_page_output(tmp_path):
    source = tmp_path / "record.pdf"
    source.write_bytes(b"synthetic")
    pipeline = FakePipeline()

    outputs = DocumentExtractor(pipeline).extract_file(source, tmp_path / "output")

    assert pipeline.restructured is True
    assert outputs == [tmp_path / "output" / "record" / "record.md"]


def test_iter_documents_filters_and_sorts(tmp_path):
    (tmp_path / "b.JPG").write_bytes(b"")
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")

    assert [path.name for path in iter_documents(tmp_path)] == ["a.pdf", "b.JPG"]


def test_cli_defaults_to_paddleocr_and_portable_mineru_runtime():
    args = build_parser().parse_args(["input.pdf", "output"])
    assert args.backend == "paddleocr"
    assert args.mineru_small_backend == "onnx"
    assert args.mineru_vlm_engine == "llama-cpp"


def test_document_extractor_supports_mineru(tmp_path):
    source = tmp_path / "record.pdf"
    source.write_bytes(b"synthetic")
    calls = []

    def parse(path, **options):
        calls.append((path, options))
        return FakeMinerUResult()

    extractor = DocumentExtractor(
        backend="mineru",
        mineru_parser=parse,
        mineru_writer=Path,
        mineru_tier="basic",
        mineru_ocr_mode="ocr",
    )
    outputs = extractor.extract_file(source, tmp_path / "output")

    assert calls == [(str(source), {"tier": "basic", "ocr_mode": "ocr"})]
    assert outputs == [tmp_path / "output" / "record" / "record.md"]
    assert outputs[0].read_text(encoding="utf-8") == "# synthetic MinerU"
