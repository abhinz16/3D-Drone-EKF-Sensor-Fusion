"""Small PX4 public-log downloader used by the project.

This is intentionally conservative. PX4 Flight Review is a public service, so
main.py only tries to download one matching log by default. If I want more logs,
I run download_px4_logs.py manually and raise --max-num.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import sys
import time
from typing import Iterable

import requests
from tqdm import tqdm

DB_INFO_API = "https://review.px4.io/dbinfo"
DOWNLOAD_API = "https://review.px4.io/download"
DEFAULT_DELAY_SECONDS = 6.0
WARN_THRESHOLD = 100


def _safe_lower(value):
    return str(value).lower() if value is not None else ""


def fetch_public_log_index(db_info_api: str = DB_INFO_API):
    response = requests.get(db_info_api, timeout=5 * 60)
    response.raise_for_status()
    return response.json()


def filter_logs(entries, mav_type=None, rating=None, source=None):
    filtered = list(entries)

    if mav_type:
        wanted = {_safe_lower(x) for x in mav_type}
        filtered = [e for e in filtered if _safe_lower(e.get("mav_type")) in wanted]

    if rating:
        wanted = {_safe_lower(x) for x in rating}
        filtered = [e for e in filtered if _safe_lower(e.get("rating")) in wanted]

    if source:
        filtered = [e for e in filtered if _safe_lower(e.get("source")) == _safe_lower(source)]

    def log_date(entry):
        try:
            return _dt.datetime.strptime(entry.get("log_date", "1900-01-01"), "%Y-%m-%d")
        except Exception:
            return _dt.datetime(1900, 1, 1)

    return sorted(filtered, key=log_date, reverse=True)


def _download_with_retry(download_api, log_id, max_retries=5):
    url = download_api + "?log=" + log_id

    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=10 * 60)

            if response.status_code == 503:
                wait_time = min(30 * (2 ** attempt), 300)
                if response.headers.get("Retry-After"):
                    wait_time = int(response.headers["Retry-After"])
                print(f"  Server is rate limiting. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code in (403, 444):
                raise RuntimeError(
                    f"PX4 server blocked the request with HTTP {response.status_code}. "
                    "Wait for a while before trying again."
                )

            if response.status_code == 404:
                print("  Log not found. Skipping.")
                return None

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            print(f"  Download attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(10 * (attempt + 1))

    return None


def download_public_logs(
    download_folder: str,
    max_num: int = 1,
    mav_type: Iterable[str] | None = ("Quadrotor",),
    rating: Iterable[str] | None = ("Good",),
    source: str | None = None,
    overwrite: bool = False,
    delay: float = DEFAULT_DELAY_SECONDS,
    db_info_api: str = DB_INFO_API,
    download_api: str = DOWNLOAD_API,
):
    os.makedirs(download_folder, exist_ok=True)

    print("Looking up public PX4 logs...")
    entries = fetch_public_log_index(db_info_api)
    entries = filter_logs(entries, mav_type=mav_type, rating=rating, source=source)

    if not entries:
        print("No PX4 logs matched the filter.")
        return []

    if max_num == -1:
        n_to_download = len(entries)
    else:
        n_to_download = min(len(entries), max_num)

    if n_to_download > WARN_THRESHOLD:
        raise RuntimeError(
            f"Refusing to automatically download {n_to_download} logs. "
            "Run download_px4_logs.py manually with --yes if you really want that."
        )

    existing = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(download_folder, "*.ulg"))
    }

    downloaded = []
    for i, entry in enumerate(entries[:n_to_download], start=1):
        log_id = entry.get("log_id")
        if not log_id:
            continue

        out_path = os.path.join(download_folder, log_id + ".ulg")
        if not overwrite and log_id in existing:
            downloaded.append(out_path)
            continue

        print(f"Downloading PX4 log {i}/{n_to_download}: {log_id}")
        response = _download_with_retry(download_api, log_id)
        if response is None:
            continue

        total = int(response.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="Writing .ulg",
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        downloaded.append(out_path)
        if i < n_to_download:
            time.sleep(delay)

    return downloaded
