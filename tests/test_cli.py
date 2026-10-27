from emr_extractor.cli import main


def test_cli_help(capsys):
    assert main(["--help"]) == 0
    assert "redact-fields" in capsys.readouterr().out


def test_cli_rejects_unknown_command(capsys):
    assert main(["unknown"]) == 2
    assert "unknown command" in capsys.readouterr().err
