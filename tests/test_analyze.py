import sys
from unittest.mock import patch


def test_cli_exits_with_error_when_file_not_found(capsys):
    with patch("sys.argv", ["analyze.py", "nonexistent.mp4"]):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            import importlib
            import analyze
            importlib.reload(analyze)
            try:
                analyze.main()
            except SystemExit as e:
                assert e.code != 0


def test_cli_exits_with_error_when_ffmpeg_missing(capsys):
    with patch("sys.argv", ["analyze.py", "video.mp4"]):
        with patch("shutil.which", return_value=None):
            import analyze
            try:
                analyze.main()
                assert False, "Should have exited"
            except SystemExit as e:
                assert e.code != 0
            output = capsys.readouterr().out
            assert "ffmpeg" in output.lower()


def test_parse_modules_all():
    with patch("sys.argv", ["analyze.py", "video.mp4"]):
        from analyze import build_parser
        parser = build_parser()
        args = parser.parse_args(["video.mp4"])
        assert args.modules == "all"


def test_parse_modules_specific():
    from analyze import build_parser
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--modules", "vocabulary,complexity"])
    assert args.modules == "vocabulary,complexity"
