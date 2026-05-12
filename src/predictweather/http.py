from __future__ import annotations

import ast
import json
import os
import time
import ssl
from contextlib import suppress
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request
from urllib.request import urlopen

import certifi


def build_ssl_context(allow_insecure: bool = False) -> ssl.SSLContext:
    if allow_insecure:
        return ssl._create_unverified_context()
    return ssl.create_default_context(cafile=certifi.where())


def env_allows_insecure_ssl() -> bool:
    return os.environ.get("PREDICTWEATHER_ALLOW_INSECURE_SSL", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _open_url_with_retry(
    url_or_request: str | Request,
    *,
    allow_insecure: bool = False,
    timeout: int = 120,
    retries: int = 4,
    backoff_seconds: float = 1.5,
):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return urlopen(url_or_request, context=build_ssl_context(allow_insecure=allow_insecure), timeout=timeout)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_HTTP_STATUS_CODES or attempt == retries - 1:
                raise
        except URLError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
        except TimeoutError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
        time.sleep(backoff_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected retry failure with no captured error.")


def parse_json_text(text: str, source: str = "response") -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                pass
        preview = stripped[:240].replace("\n", " ")
        raise ValueError(f"Invalid JSON from {source}: {exc}. Preview: {preview}") from exc


def fetch_json(url: str, allow_insecure: bool = False) -> dict:
    with _open_url_with_retry(url, allow_insecure=allow_insecure) as response:
        body = response.read().decode("utf-8", errors="replace")
    return parse_json_text(body, source=url)


def fetch_text(url: str, allow_insecure: bool = False) -> str:
    with _open_url_with_retry(url, allow_insecure=allow_insecure) as response:
        return response.read().decode("utf-8", errors="replace")


def url_exists(url: str, allow_insecure: bool = False) -> bool:
    with _open_url_with_retry(url, allow_insecure=allow_insecure, timeout=60, retries=2, backoff_seconds=0.5) as response:
        return response.status == 200


def download_url_to_file(
    url: str,
    destination: Path,
    allow_insecure: bool = False,
    headers: dict[str, str] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.partial")
    with suppress(FileNotFoundError):
        temp_path.unlink()

    try:
        request: str | Request = url if not headers else Request(url, headers=headers)
        with _open_url_with_retry(request, allow_insecure=allow_insecure, timeout=180, retries=5, backoff_seconds=2.0) as response:
            content_length_header = response.headers.get("Content-Length")
            expected_length = int(content_length_header) if content_length_header and content_length_header.isdigit() else None
            bytes_written = 0
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_written += len(chunk)

        if bytes_written <= 0:
            raise ValueError(f"Downloaded empty response from {url}")
        if expected_length is not None and bytes_written != expected_length:
            raise ValueError(f"Incomplete download from {url}: expected {expected_length} bytes, got {bytes_written}")

        temp_path.replace(destination)
        return destination
    except Exception:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise
