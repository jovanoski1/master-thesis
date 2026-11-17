import json

import pytest

from aml import acquire


def fake_kaggle(tmp_path, latest=8):
    """Imitates kagglehub.dataset_download: returns a cache path containing the version."""
    calls = []

    def download(handle, path):
        calls.append(handle)
        v = handle.rsplit("/versions/", 1)[1] if "/versions/" in handle else latest
        p = tmp_path / "cache" / "datasets" / "o" / "d" / "versions" / str(v) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content of {path}")
        return str(p)
    return download, calls


def test_first_download_pins_version_and_later_ones_use_it(tmp_path):
    download, calls = fake_kaggle(tmp_path)
    raw, vfile = tmp_path / "raw", tmp_path / "results" / "DATA_VERSION.json"
    v = acquire.fetch_raw(["A.csv"], raw, vfile, handle="o/d", download=download)
    assert v == 8 and (raw / "A.csv").read_text() == "content of A.csv"
    rec = json.loads(vfile.read_text())
    assert (rec["version"], rec["source"]) == (8, "kagglehub")
    acquire.fetch_raw(["A.csv", "B.txt"], raw, vfile, handle="o/d", download=download)
    assert calls == ["o/d", "o/d/versions/8"]        # A.csv not fetched again; B pinned


def test_version_mismatch_is_refused(tmp_path):
    download, _ = fake_kaggle(tmp_path)
    vfile = tmp_path / "DATA_VERSION.json"
    vfile.write_text(json.dumps({"handle": "o/d", "version": 3}))

    def wrong(handle, path):
        return download("o/d/versions/9", path)
    with pytest.raises(ValueError):
        acquire.fetch_raw(["A.csv"], tmp_path / "raw", vfile, handle="o/d", download=wrong)


def test_manual_files_and_zips_need_no_download(tmp_path):
    import zipfile
    raw, vfile = tmp_path / "raw", tmp_path / "DATA_VERSION.json"
    raw.mkdir()
    (raw / "A.csv").write_text("a")
    with zipfile.ZipFile(raw / "B.txt.zip", "w") as zf:      # as delivered by the website
        zf.writestr("B.txt", "b")

    def no_download(handle, path):
        raise AssertionError("must not download")
    v = acquire.fetch_raw(["A.csv", "B.txt"], raw, vfile, handle="o/d", download=no_download)
    assert v is None and (raw / "B.txt").read_text() == "b"
    assert json.loads(vfile.read_text())["source"] == "placed in data/raw by hand"


def test_failed_download_explains_the_manual_route(tmp_path):
    def unauthorised(handle, path):
        raise PermissionError("401")
    with pytest.raises(RuntimeError, match="upload them"):
        acquire.fetch_raw(["A.csv"], tmp_path / "raw", tmp_path / "v.json",
                          handle="o/d", download=unauthorised)
