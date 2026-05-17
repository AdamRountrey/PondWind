from __future__ import annotations

import ast
import json
import os
import ssl
import time
from contextlib import suppress
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import datetime, timezone
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
DEFAULT_USER_AGENT = os.environ.get(
    "PONDWIND_USER_AGENT",
    "PondWind/0.1 (+https://github.com/AdamRountrey/PondWind)",
)


def _request_with_default_headers(url_or_request: str | Request) -> str | Request:
    if isinstance(url_or_request, Request):
        if not url_or_request.has_header("User-agent"):
            url_or_request.add_header("User-Agent", DEFAULT_USER_AGENT)
        return url_or_request
    return Request(url_or_request, headers={"User-Agent": DEFAULT_USER_AGENT})


def _retry_after_seconds(exc: HTTPError) -> float | None:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if not retry_after:
        return None
    retry_after = retry_after.strip()
    if retry_after.isdigit():
        return max(0.0, float(retry_after))
    try:
        retry_at = parsedate_to_datetime(retry_after)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


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
            request = _request_with_default_headers(url_or_request)
            return urlopen(request, context=build_ssl_context(allow_insecure=allow_insecure), timeout=timeout)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_HTTP_STATUS_CODES or attempt == retries - 1:
                raise
            delay = _retry_after_seconds(exc)
        except URLError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            delay = None
        except TimeoutError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            delay = None
        time.sleep(delay if delay is not None else backoff_seconds * (attempt + 1))
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
        request_headers = {"User-Agent": DEFAULT_USER_AGENT}
        if headers:
            request_headers.update(headers)
        request: str | Request = Request(url, headers=request_headers)
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
