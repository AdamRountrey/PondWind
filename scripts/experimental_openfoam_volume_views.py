"""Experimental OpenFOAM volume preview renderer for PondWind.

This script is intentionally standalone. It samples an existing PondWind
OpenFOAM case at several terrain-following heights and writes a couple of
static 3D-ish PNG previews for quick visual experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_REPORT_NAME = "20260517_1400_barton_pond3"
DEFAULT_HEIGHTS_M = (10.0, 25.0, 50.0, 100.0, 180.0, 300.0)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _local_app_data() -> Path:
    import os

    value = os.environ.get("LOCALAPPDATA")
    if not value:
        raise RuntimeError("LOCALAPPDATA is not set. Pass --case-dir explicitly.")
    return Path(value)


def _default_case_dir() -> Path:
    return (
        _local_app_data()
        / "PondWind"
        / "outputs"
        / "reports"
        / DEFAULT_REPORT_NAME
        / "report_temp"
        / "wind"
        / "openfoam_comparison"
        / "case"
    )


def _windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _parse_block_mesh_vertices(block_mesh_dict: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = block_mesh_dict.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\bvertices\s*\((.*?)\);\s*\bblocks\b", text, re.S)
    if not match:
        raise RuntimeError(f"Could not find vertices in {block_mesh_dict}")
    coords: list[tuple[float, float, float]] = []
    for line in match.group(1).splitlines():
        values = [float(v) for v in NUMBER_RE.findall(line)]
        if len(values) == 3:
            coords.append((values[0], values[1], values[2]))
    if not coords or len(coords) % 2 != 0:
        raise RuntimeError("Unexpected blockMeshDict vertex count.")

    arr = np.asarray(coords, dtype=np.float64)
    half = arr.shape[0] // 2
    bottom = arr[:half]
    x_values = np.unique(np.round(bottom[:, 0], 6))
    y_values = np.unique(np.round(bottom[:, 1], 6))
    nx = len(x_values) - 1
    ny = len(y_values) - 1
    if nx <= 0 or ny <= 0 or (nx + 1) * (ny + 1) != half:
        raise RuntimeError("Could not infer structured terrain dimensions from blockMeshDict.")
    terrain = bottom[:, 2].reshape(ny + 1, nx + 1)
    return x_values, y_values, terrain


def _cell_centers(x_vertices: np.ndarray, y_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (x_vertices[:-1] + x_vertices[1:]), 0.5 * (y_vertices[:-1] + y_vertices[1:])


def _terrain_cell_max(terrain_vertices: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(
        (
            terrain_vertices[:-1, :-1],
            terrain_vertices[1:, :-1],
            terrain_vertices[:-1, 1:],
            terrain_vertices[1:, 1:],
        )
    )


def _write_volume_sample_dict(
    case_dir: Path,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    terrain_max: np.ndarray,
    heights_m: tuple[float, ...],
) -> Path:
    system_dir = case_dir / "system"
    sample_dict = system_dir / "volumePreviewSampleDict"
    sets: list[str] = []
    for height_m in heights_m:
        name = f"h{int(round(height_m)):03d}m"
        points: list[str] = []
        for row in range(len(y_centers) - 1, -1, -1):
            for col in range(len(x_centers)):
                z = float(terrain_max[row, col] + height_m)
                points.append(f"                ({x_centers[col]:.6f} {y_centers[row]:.6f} {z:.6f})")
        sets.extend(
            [
                f"        {name}",
                "        {",
                "            type points;",
                "            ordered yes;",
                "            points",
                "            (",
                *points,
                "            );",
                "        }",
            ]
        )

    sample_dict.write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    version     2.0;",
                "    format      ascii;",
                "    class       dictionary;",
                "    object      volumePreviewSampleDict;",
                "}",
                "",
                "volumePreview",
                "{",
                "    type sets;",
                '    libs ("libsampling.so");',
                "    executeControl writeTime;",
                "    writeControl writeTime;",
                "    interpolationScheme cellPoint;",
                "    setFormat raw;",
                "    fields (U k);",
                "    sets",
                "    {",
                *sets,
                "    }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return sample_dict


def _run_openfoam_sampler(case_dir: Path, sample_dict: Path) -> None:
    case_wsl = _windows_to_wsl_path(case_dir)
    dict_wsl = _windows_to_wsl_path(sample_dict)
    command = (
        "source /opt/openfoam13/etc/bashrc && "
        f'postProcess -case "{case_wsl}" -latestTime -dict "{dict_wsl}"'
    )
    completed = subprocess.run(["wsl", "bash", "-lc", command], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenFOAM volume sampling failed.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def _find_sample_dir(case_dir: Path) -> Path:
    root = case_dir / "postProcessing" / "volumePreview"
    if not root.exists():
        raise RuntimeError(f"OpenFOAM did not create {root}")
    time_dirs = [child for child in root.iterdir() if child.is_dir()]
    if not time_dirs:
        raise RuntimeError(f"OpenFOAM did not create time output under {root}")
    return max(time_dirs, key=lambda p: float(p.name))


def _parse_xy(path: Path, nx: int, ny: int) -> dict[str, np.ndarray]:
    xs = np.full((ny, nx), np.nan, dtype=np.float64)
    ys = np.full((ny, nx), np.nan, dtype=np.float64)
    zs = np.full((ny, nx), np.nan, dtype=np.float64)
    u = np.full((ny, nx), np.nan, dtype=np.float64)
    v = np.full((ny, nx), np.nan, dtype=np.float64)
    w = np.full((ny, nx), np.nan, dtype=np.float64)
    k = np.full((ny, nx), np.nan, dtype=np.float64)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        values = [float(item) for item in NUMBER_RE.findall(line)]
        if len(values) >= 8:
            rows.append(values)
    if len(rows) != nx * ny:
        raise RuntimeError(f"Expected {nx * ny} samples in {path}, found {len(rows)}")
    for index, values in enumerate(rows):
        row = ny - 1 - (index // nx)
        col = index % nx
        xs[row, col] = values[1]
        ys[row, col] = values[2]
        zs[row, col] = values[3]
        u[row, col] = values[4]
        v[row, col] = values[5]
        w[row, col] = values[6]
        k[row, col] = values[7]
    return {"x": xs, "y": ys, "z": zs, "u": u, "v": v, "w": w, "k": k}


def _load_layers(sample_dir: Path, heights_m: tuple[float, ...], nx: int, ny: int) -> list[dict[str, np.ndarray | float]]:
    layers = []
    for height_m in heights_m:
        name = f"h{int(round(height_m)):03d}m"
        path = sample_dir / f"{name}.xy"
        if not path.exists():
            raise RuntimeError(f"Missing sampled layer: {path}")
        layer = _parse_xy(path, nx, ny)
        layer["height_m"] = height_m
        layer["speed"] = np.sqrt(layer["u"] ** 2 + layer["v"] ** 2 + layer["w"] ** 2)
        layers.append(layer)
    return layers


def _palette(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if not math.isfinite(value):
        return (190, 190, 190)
    t = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1.0e-6)))
    stops = [
        (38, 71, 134),
        (34, 139, 178),
        (70, 178, 143),
        (239, 196, 84),
        (217, 95, 76),
    ]
    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    return tuple(int(stops[i][c] * (1.0 - f) + stops[i + 1][c] * f) for c in range(3))


class Projector:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, width: int, height: int, azimuth_deg: float):
        self.width = width
        self.height = height
        self.cx = 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x)))
        self.cy = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
        self.zmin = float(np.nanmin(z))
        self.az = math.radians(azimuth_deg)
        self.elev = math.radians(34.0)
        domain = max(float(np.nanmax(x) - np.nanmin(x)), float(np.nanmax(y) - np.nanmin(y)))
        self.scale = 0.72 * min(width, height) / domain
        self.z_exag = 3.8

    def __call__(self, x: float, y: float, z: float) -> tuple[float, float]:
        dx = x - self.cx
        dy = y - self.cy
        xr = dx * math.cos(self.az) - dy * math.sin(self.az)
        yr = dx * math.sin(self.az) + dy * math.cos(self.az)
        zr = (z - self.zmin) * self.z_exag
        sx = self.width * 0.5 + xr * self.scale
        sy = self.height * 0.70 - (yr * math.sin(self.elev) + zr * math.cos(self.elev)) * self.scale
        return sx, sy


def _bilinear(grid: np.ndarray, x: float, y: float, x0: float, y0: float, dx: float, dy: float) -> float:
    fx = (x - x0) / dx
    fy = (y - y0) / dy
    i = int(math.floor(fx))
    j = int(math.floor(fy))
    if i < 0 or j < 0 or i >= grid.shape[1] - 1 or j >= grid.shape[0] - 1:
        return float("nan")
    tx = fx - i
    ty = fy - j
    return float(
        grid[j, i] * (1 - tx) * (1 - ty)
        + grid[j, i + 1] * tx * (1 - ty)
        + grid[j + 1, i] * (1 - tx) * ty
        + grid[j + 1, i + 1] * tx * ty
    )


def _trace_streamline(layer: dict[str, np.ndarray | float], seed_x: float, seed_y: float, step_m: float, max_steps: int) -> list[tuple[float, float, float, float]]:
    x_grid = layer["x"]
    y_grid = layer["y"]
    z_grid = layer["z"]
    u_grid = layer["u"]
    v_grid = layer["v"]
    s_grid = layer["speed"]
    x0 = float(np.nanmin(x_grid))
    y0 = float(np.nanmin(y_grid))
    dx = float(np.nanmedian(np.diff(np.unique(np.round(x_grid, 6)))))
    dy = float(np.nanmedian(np.diff(np.unique(np.round(y_grid, 6)))))
    x = seed_x
    y = seed_y
    points = []
    for _ in range(max_steps):
        u = _bilinear(u_grid, x, y, x0, y0, dx, dy)
        v = _bilinear(v_grid, x, y, x0, y0, dx, dy)
        z = _bilinear(z_grid, x, y, x0, y0, dx, dy)
        speed = _bilinear(s_grid, x, y, x0, y0, dx, dy)
        if not all(math.isfinite(item) for item in (u, v, z, speed)) or speed < 0.05:
            break
        points.append((x, y, z, speed))
        x += (u / speed) * step_m
        y += (v / speed) * step_m
    return points


def _streamline_seeds(layer: dict[str, np.ndarray | float], count: int) -> list[tuple[float, float]]:
    x_grid = layer["x"]
    y_grid = layer["y"]
    minx, maxx = float(np.nanmin(x_grid)), float(np.nanmax(x_grid))
    miny, maxy = float(np.nanmin(y_grid)), float(np.nanmax(y_grid))
    mean_u = float(np.nanmean(layer["u"]))
    mean_v = float(np.nanmean(layer["v"]))
    seeds = []
    ys = np.linspace(miny, maxy, count)
    xs = np.linspace(minx, maxx, count)
    if mean_u >= 0:
        seeds.extend((minx + 1.0, y) for y in ys)
    else:
        seeds.extend((maxx - 1.0, y) for y in ys)
    if mean_v >= 0:
        seeds.extend((x, miny + 1.0) for x in xs)
    else:
        seeds.extend((x, maxy - 1.0) for x in xs)
    return seeds


def _draw_title(draw: ImageDraw.ImageDraw, text: str, subtitle: str) -> None:
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 1500, 74), fill=(246, 249, 250, 238))
    draw.text((28, 18), text, fill=(21, 35, 42), font=font)
    draw.text((28, 42), subtitle, fill=(78, 96, 105), font=font)


def _draw_legend(draw: ImageDraw.ImageDraw, vmin: float, vmax: float, x: int, y: int) -> None:
    font = ImageFont.load_default()
    width = 260
    for i in range(width):
        color = _palette(vmin + (vmax - vmin) * i / max(width - 1, 1), vmin, vmax)
        draw.line((x + i, y, x + i, y + 14), fill=color)
    draw.rectangle((x, y, x + width, y + 14), outline=(80, 96, 104))
    draw.text((x, y + 20), f"{vmin:.1f} m/s", fill=(34, 43, 48), font=font)
    draw.text((x + width - 52, y + 20), f"{vmax:.1f} m/s", fill=(34, 43, 48), font=font)
    draw.text((x, y - 16), "Wind speed", fill=(34, 43, 48), font=font)


def _draw_terrain(draw: ImageDraw.ImageDraw, projector: Projector, x_vertices: np.ndarray, y_vertices: np.ndarray, terrain: np.ndarray) -> None:
    zmin = float(np.nanmin(terrain))
    zmax = float(np.nanmax(terrain))
    step = max(1, int(round(len(x_vertices) / 75)))
    cells = []
    for j in range(0, len(y_vertices) - 1, step):
        for i in range(0, len(x_vertices) - 1, step):
            poly = [
                projector(float(x_vertices[i]), float(y_vertices[j]), float(terrain[j, i])),
                projector(float(x_vertices[min(i + step, len(x_vertices) - 1)]), float(y_vertices[j]), float(terrain[j, min(i + step, len(x_vertices) - 1)])),
                projector(float(x_vertices[min(i + step, len(x_vertices) - 1)]), float(y_vertices[min(j + step, len(y_vertices) - 1)]), float(terrain[min(j + step, len(y_vertices) - 1), min(i + step, len(x_vertices) - 1)])),
                projector(float(x_vertices[i]), float(y_vertices[min(j + step, len(y_vertices) - 1)]), float(terrain[min(j + step, len(y_vertices) - 1), i])),
            ]
            avg_depth = sum(p[1] for p in poly) / len(poly)
            elev = float(np.nanmean(terrain[j : min(j + step + 1, terrain.shape[0]), i : min(i + step + 1, terrain.shape[1])]))
            t = (elev - zmin) / max(zmax - zmin, 1.0)
            color = (156 + int(50 * t), 171 + int(42 * t), 139 + int(35 * t))
            cells.append((avg_depth, poly, color))
    for _, poly, color in sorted(cells, key=lambda item: item[0]):
        draw.polygon(poly, fill=color, outline=(132, 147, 128))


def _draw_streamlines(
    draw: ImageDraw.ImageDraw,
    projector: Projector,
    layers: list[dict[str, np.ndarray | float]],
    vmin: float,
    vmax: float,
    seeds_per_edge: int,
    max_steps: int,
) -> None:
    step_m = 18.0
    all_lines = []
    for layer in layers:
        for seed_x, seed_y in _streamline_seeds(layer, seeds_per_edge):
            points = _trace_streamline(layer, seed_x, seed_y, step_m, max_steps)
            if len(points) >= 8:
                avg_depth = float(np.mean([projector(x, y, z)[1] for x, y, z, _ in points]))
                all_lines.append((avg_depth, points))
    for _, points in sorted(all_lines, key=lambda item: item[0]):
        projected = [projector(x, y, z) for x, y, z, _ in points]
        for a, b, point in zip(projected[:-1], projected[1:], points[:-1]):
            color = _palette(point[3], vmin, vmax)
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=3)
        if len(projected) > 2:
            a = projected[-2]
            b = projected[-1]
            draw.line((a[0], a[1], b[0], b[1]), fill=(25, 40, 48), width=2)


def _render_stream_volume(
    output_path: Path,
    x_vertices: np.ndarray,
    y_vertices: np.ndarray,
    terrain: np.ndarray,
    layers: list[dict[str, np.ndarray | float]],
) -> None:
    width, height = 1500, 1050
    image = Image.new("RGB", (width, height), (236, 241, 243))
    draw = ImageDraw.Draw(image, "RGBA")
    all_speed = np.concatenate([np.asarray(layer["speed"]).ravel() for layer in layers])
    vmin = float(np.nanpercentile(all_speed, 3))
    vmax = float(np.nanpercentile(all_speed, 97))
    projector = Projector(layers[0]["x"], layers[0]["y"], layers[-1]["z"], width, height, azimuth_deg=42.0)
    _draw_terrain(draw, projector, x_vertices, y_vertices, terrain)
    _draw_streamlines(draw, projector, layers[::2], vmin, vmax, seeds_per_edge=15, max_steps=260)
    _draw_title(
        draw,
        "OpenFOAM Volume Preview: Terrain-Following Streamlines",
        "Sampled from the saved 3D U field at multiple heights; experimental visualization only.",
    )
    _draw_legend(draw, vmin, vmax, width - 330, 28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _render_stacked_slices(
    output_path: Path,
    x_vertices: np.ndarray,
    y_vertices: np.ndarray,
    terrain: np.ndarray,
    layers: list[dict[str, np.ndarray | float]],
) -> None:
    width, height = 1500, 1050
    image = Image.new("RGB", (width, height), (238, 242, 244))
    draw = ImageDraw.Draw(image, "RGBA")
    all_speed = np.concatenate([np.asarray(layer["speed"]).ravel() for layer in layers])
    vmin = float(np.nanpercentile(all_speed, 3))
    vmax = float(np.nanpercentile(all_speed, 97))
    projector = Projector(layers[0]["x"], layers[0]["y"], layers[-1]["z"], width, height, azimuth_deg=-28.0)
    _draw_terrain(draw, projector, x_vertices, y_vertices, terrain)

    for layer in layers:
        x_grid = layer["x"]
        y_grid = layer["y"]
        z_grid = layer["z"]
        s_grid = layer["speed"]
        stride = max(4, int(round(x_grid.shape[1] / 35)))
        samples = []
        for j in range(0, x_grid.shape[0], stride):
            for i in range(0, x_grid.shape[1], stride):
                samples.append((projector(float(x_grid[j, i]), float(y_grid[j, i]), float(z_grid[j, i])), float(s_grid[j, i])))
        for (sx, sy), speed in samples:
            color = _palette(speed, vmin, vmax)
            draw.ellipse((sx - 4, sy - 3, sx + 4, sy + 3), fill=(*color, 92))
        _draw_streamlines(draw, projector, [layer], vmin, vmax, seeds_per_edge=8, max_steps=190)

    _draw_title(
        draw,
        "OpenFOAM Volume Preview: Stacked Wind Layers",
        "Colored points and streamlines show sampled wind slices above terrain.",
    )
    _draw_legend(draw, vmin, vmax, width - 330, 28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create experimental volume preview images from a PondWind OpenFOAM case.")
    parser.add_argument("--case-dir", type=Path, default=_default_case_dir())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--heights-m", default=",".join(str(int(h)) for h in DEFAULT_HEIGHTS_M))
    parser.add_argument("--skip-sample", action="store_true", help="Use existing postProcessing/volumePreview samples.")
    args = parser.parse_args()

    case_dir = args.case_dir
    if not case_dir.exists():
        raise RuntimeError(f"OpenFOAM case directory was not found: {case_dir}")
    output_dir = args.output_dir or case_dir.parent / "volume_views"
    heights_m = tuple(float(value.strip()) for value in args.heights_m.split(",") if value.strip())

    x_vertices, y_vertices, terrain = _parse_block_mesh_vertices(case_dir / "system" / "blockMeshDict")
    x_centers, y_centers = _cell_centers(x_vertices, y_vertices)
    terrain_max = _terrain_cell_max(terrain)
    sample_dict = _write_volume_sample_dict(case_dir, x_centers, y_centers, terrain_max, heights_m)
    if not args.skip_sample:
        _run_openfoam_sampler(case_dir, sample_dict)
    sample_dir = _find_sample_dir(case_dir)
    layers = _load_layers(sample_dir, heights_m, len(x_centers), len(y_centers))

    outputs = {
        "stream_volume": output_dir / "openfoam_volume_streamlines_isometric.png",
        "stacked_slices": output_dir / "openfoam_volume_stacked_slices.png",
    }
    _render_stream_volume(outputs["stream_volume"], x_vertices, y_vertices, terrain, layers)
    _render_stacked_slices(outputs["stacked_slices"], x_vertices, y_vertices, terrain, layers)

    manifest = {
        "case_dir": str(case_dir),
        "sample_dir": str(sample_dir),
        "heights_m": heights_m,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "volume_view_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
