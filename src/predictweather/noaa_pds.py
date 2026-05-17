from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

from predictweather.http import download_url_to_file, env_allows_insecure_ssl, fetch_text


@dataclass(frozen=True)
class IndexEntry:
    message_number: int
    offset: int
    variable: str
    level: str
    description: str


def parse_grib_index(text: str) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(":")
        if len(parts) < 5:
            continue
        try:
            message_number = int(parts[0])
            offset = int(parts[1])
        except ValueError:
            continue
        entries.append(
            IndexEntry(
                message_number=message_number,
                offset=offset,
                variable=parts[3].strip(),
                level=parts[4].strip(),
                description=stripped,
            )
        )
    return entries


def _matches(entry: IndexEntry, *, variable: str, level_text: str) -> bool:
    return entry.variable.upper() == variable.upper() and level_text.lower() in entry.level.lower()


def find_grib_message(entries: list[IndexEntry], *, variable: str, level_text: str) -> tuple[IndexEntry, int | None]:
    for index, entry in enumerate(entries):
        if _matches(entry, variable=variable, level_text=level_text):
            next_offset = entries[index + 1].offset if index + 1 < len(entries) else None
            return entry, next_offset
    raise KeyError(f"No {variable} message at {level_text!r} in GRIB index.")


def grib_index_url(grib_url: str) -> str:
    return f"{grib_url}.idx"


def load_grib_index(grib_url: str) -> list[IndexEntry]:
    return parse_grib_index(fetch_text(grib_index_url(grib_url), allow_insecure=env_allows_insecure_ssl()))


def has_indexed_message(grib_url: str, *, variable: str, level_text: str) -> bool:
    try:
        find_grib_message(load_grib_index(grib_url), variable=variable, level_text=level_text)
        return True
    except (HTTPError, URLError, TimeoutError, KeyError):
        return False


def has_indexed_messages(grib_url: str, specs: tuple[tuple[str, str], ...]) -> bool:
    try:
        entries = load_grib_index(grib_url)
        for variable, level_text in specs:
            find_grib_message(entries, variable=variable, level_text=level_text)
        return True
    except (HTTPError, URLError, TimeoutError, KeyError):
        return False


def download_indexed_message(
    grib_url: str,
    destination: Path,
    *,
    variable: str,
    level_text: str,
) -> Path:
    entries = load_grib_index(grib_url)
    entry, next_offset = find_grib_message(entries, variable=variable, level_text=level_text)
    headers = {"Range": f"bytes={entry.offset}-{next_offset - 1}"} if next_offset is not None else {"Range": f"bytes={entry.offset}-"}
    return download_url_to_file(
        grib_url,
        destination,
        allow_insecure=env_allows_insecure_ssl(),
        headers=headers,
    )
