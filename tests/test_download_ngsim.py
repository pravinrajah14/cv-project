import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import download_ngsim  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code, headers, chunks):
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_video_downloads_fresh_file_in_one_request(tmp_path):
    out_path = tmp_path / "video.avi"

    def fake_get(url, params=None, headers=None, stream=None, timeout=None):
        assert "Range" not in (headers or {})
        return _FakeResponse(200, {"Content-Length": "5"}, [b"hello"])

    with patch("download_ngsim.list_video_attachments", return_value=[{"filename": "v.avi", "assetId": "abc"}]):
        with patch("download_ngsim.requests.get", side_effect=fake_get):
            download_ngsim.fetch_video("v.avi", out_path)

    assert out_path.read_bytes() == b"hello"


def test_fetch_video_resumes_from_existing_partial_file(tmp_path):
    out_path = tmp_path / "video.avi"
    tmp_part = out_path.with_suffix(out_path.suffix + ".part")
    tmp_part.write_bytes(b"hello")

    def fake_get(url, params=None, headers=None, stream=None, timeout=None):
        assert headers.get("Range") == "bytes=5-"
        return _FakeResponse(206, {"Content-Length": "5"}, [b"world"])

    with patch("download_ngsim.list_video_attachments", return_value=[{"filename": "v.avi", "assetId": "abc"}]):
        with patch("download_ngsim.requests.get", side_effect=fake_get):
            download_ngsim.fetch_video("v.avi", out_path)

    assert out_path.read_bytes() == b"helloworld"


def test_fetch_video_restarts_if_server_ignores_range(tmp_path):
    out_path = tmp_path / "video.avi"
    tmp_part = out_path.with_suffix(out_path.suffix + ".part")
    tmp_part.write_bytes(b"stale-partial-data")

    def fake_get(url, params=None, headers=None, stream=None, timeout=None):
        # server ignores Range and returns 200 with the full content
        return _FakeResponse(200, {"Content-Length": "5"}, [b"fresh"])

    with patch("download_ngsim.list_video_attachments", return_value=[{"filename": "v.avi", "assetId": "abc"}]):
        with patch("download_ngsim.requests.get", side_effect=fake_get):
            download_ngsim.fetch_video("v.avi", out_path)

    assert out_path.read_bytes() == b"fresh"
