"""Acquisition of the raw files (Section 4.2): kagglehub, or files placed by hand.

If the four files (or zip archives containing them) are already in raw_dir, nothing is
downloaded, so a manual download through the Kaggle website works without any API key.

Only the download goes through kagglehub. Parsing stays in `load.read_transactions`,
which verifies the repeated `Account` header before renaming (Section 4.4); the patterns
file is not tabular and cannot be read by kagglehub's pandas adapter in any case.
"""
import datetime
import json
import re
import shutil
import zipfile
from pathlib import Path

from . import config


def _version_in(path: Path):
    m = re.search(r"[/\\]versions[/\\](\d+)(?:[/\\]|$)", str(path))
    return int(m.group(1)) if m else None


MANUAL_HELP = """\
Automatic download failed ({error}).
Download these files by hand from the Data tab of
  https://www.kaggle.com/datasets/{handle}/data
(click each file, then its download icon; do not download the whole dataset):
  {files}
and upload them, zipped or not, to the Drive folder
  {raw_dir}
then run this cell again."""


def unpack_archives(raw_dir, files) -> list:
    """Extract any wanted file found inside a zip archive in raw_dir (for manual
    downloads, which Kaggle may deliver zipped). The archives themselves are kept."""
    raw_dir, wanted, done = Path(raw_dir), set(files), []
    for z in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name in wanted and not (raw_dir / name).exists():
                    with zf.open(member) as src, open(raw_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    done.append(name)
    return done


def fetch_raw(files, raw_dir, version_file, handle: str = config.DATASET_HANDLE,
              download=None):
    """Make the four files available in raw_dir and record where they came from.

    Files already in raw_dir (or in a zip there) are used as they are. Anything missing
    is downloaded with kagglehub; its cache on the Colab machine is wiped between
    sessions, so each file is copied to raw_dir on Drive. The version resolved by the
    first download is recorded and requested explicitly afterwards, so a new upload on
    Kaggle cannot enter the experiment unnoticed. Returns the recorded version, or None
    for files placed by hand (the checksums then carry the guarantee).
    """
    raw_dir, version_file = Path(raw_dir), Path(version_file)
    raw_dir.mkdir(parents=True, exist_ok=True)
    unpack_archives(raw_dir, files)
    record = json.loads(version_file.read_text()) if version_file.exists() else None
    version = record["version"] if record else None
    missing = [f for f in files if not (raw_dir / f).exists()]

    if not missing:
        if record is None:
            version_file.parent.mkdir(parents=True, exist_ok=True)
            version_file.write_text(json.dumps({
                "handle": handle, "version": None, "source": "placed in data/raw by hand",
                "recorded": datetime.date.today().isoformat()}, indent=2))
        return version

    try:
        if download is None:
            import kagglehub
            download = kagglehub.dataset_download
        for f in missing:
            h = handle if version is None else f"{handle}/versions/{version}"
            cached = Path(download(h, path=f))
            resolved = _version_in(cached)
            if version is None and resolved is not None:
                version = resolved
                version_file.parent.mkdir(parents=True, exist_ok=True)
                version_file.write_text(json.dumps({
                    "handle": handle, "version": version, "source": "kagglehub",
                    "recorded": datetime.date.today().isoformat()}, indent=2))
            elif None not in (version, resolved) and resolved != version:
                raise ValueError(f"{f}: received version {resolved}, pinned {version}")
            if cached.is_dir():
                cached = cached / f
            if cached.suffix == ".zip":
                with zipfile.ZipFile(cached) as zf:
                    zf.extract(f, raw_dir)
            else:
                shutil.copy2(cached, raw_dir / f)
            print(f"{f}: version {resolved}, copied to {raw_dir}")
    except ValueError:
        raise
    except Exception as e:  # authentication, network, missing package
        still = [f for f in files if not (raw_dir / f).exists()]
        raise RuntimeError(MANUAL_HELP.format(error=type(e).__name__, handle=handle,
                                              files="\n  ".join(still),
                                              raw_dir=raw_dir)) from None
    return version
