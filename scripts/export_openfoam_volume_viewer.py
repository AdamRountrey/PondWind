"""Export an experimental interactive OpenFOAM volume viewer.

The output is a small local HTML app plus a compact JSON payload. It is meant
for quick visual inspection of a saved PondWind OpenFOAM case, not as a
production report product.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experimental_openfoam_volume_views import (
    DEFAULT_HEIGHTS_M,
    _default_case_dir,
    _find_sample_dir,
    _load_layers,
    _parse_block_mesh_vertices,
    _run_openfoam_sampler,
    _terrain_cell_max,
    _write_volume_sample_dict,
    _cell_centers,
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PondWind OpenFOAM Volume Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: #11181d;
      color: #eef4f6;
    }
    html, body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: #11181d;
    }
    #viewer {
      width: 100vw;
      height: 100vh;
      display: block;
      cursor: grab;
      background: linear-gradient(#18232a, #10171b);
    }
    #viewer.dragging {
      cursor: grabbing;
    }
    .panel {
      position: fixed;
      left: 16px;
      top: 16px;
      width: min(360px, calc(100vw - 32px));
      background: rgba(15, 22, 27, 0.88);
      border: 1px solid rgba(187, 214, 222, 0.24);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(8px);
    }
    h1 {
      margin: 0 0 4px;
      font-size: 19px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .subtle {
      margin: 0 0 12px;
      color: #a9bbc2;
      font-size: 13px;
      line-height: 1.35;
    }
    .row {
      display: grid;
      grid-template-columns: 116px 1fr;
      align-items: center;
      gap: 10px;
      margin: 9px 0;
      font-size: 13px;
    }
    input[type="range"] {
      width: 100%;
    }
    .checks {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-top: 8px;
    }
    label.check {
      display: flex;
      gap: 6px;
      align-items: center;
      color: #d7e4e8;
      font-size: 12px;
      white-space: nowrap;
    }
    button {
      border: 1px solid rgba(187, 214, 222, 0.32);
      border-radius: 6px;
      background: #22333b;
      color: #eef4f6;
      padding: 7px 10px;
      font: inherit;
      font-size: 13px;
    }
    .legend {
      position: fixed;
      right: 16px;
      top: 16px;
      width: 260px;
      background: rgba(15, 22, 27, 0.82);
      border: 1px solid rgba(187, 214, 222, 0.24);
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      color: #c8d8dd;
    }
    .bar {
      height: 12px;
      border-radius: 4px;
      background: linear-gradient(90deg, #264786, #228bb2, #46b28f, #efc454, #d95f4c);
      margin: 8px 0 6px;
    }
    .legend-scale {
      display: flex;
      justify-content: space-between;
    }
  </style>
</head>
<body>
  <canvas id="viewer"></canvas>
  <section class="panel">
    <h1>OpenFOAM Volume Viewer</h1>
    <p class="subtle">Drag to rotate, wheel to zoom, Shift+drag to pan. Experimental terrain-following samples from the saved 3D CFD field.</p>
    <div class="row">
      <span>Vertical scale</span>
      <input id="zScale" type="range" min="1" max="10" step="0.25" value="4">
    </div>
    <div class="row">
      <span>Particles</span>
      <input id="particleCount" type="range" min="0" max="850" step="25" value="425">
    </div>
    <div class="row">
      <span>Vectors</span>
      <input id="vectorDensity" type="range" min="0" max="5" step="1" value="2">
    </div>
    <div class="row">
      <span>Streamlines</span>
      <input id="streamDensity" type="range" min="0" max="5" step="1" value="3">
    </div>
    <div class="row">
      <span>Terrain</span>
      <button id="terrainToggle" type="button">Visible</button>
    </div>
    <div class="checks" id="layerChecks"></div>
  </section>
  <section class="legend">
    <strong>Wind speed</strong>
    <div class="bar"></div>
    <div class="legend-scale"><span id="speedMin"></span><span id="speedMax"></span></div>
  </section>
  <script id="volume-data" type="application/json">__VOLUME_DATA__</script>
  <script>
    const canvas = document.getElementById("viewer");
    const ctx = canvas.getContext("2d");
    const state = {
      azimuth: -0.72,
      elevation: 0.62,
      zoom: 1.0,
      panX: 0,
      panY: 0,
      zScale: 4.0,
      showTerrain: true,
      vectorDensity: 2,
      streamDensity: 3,
      particleCount: 425,
      layersVisible: new Set(),
      particles: [],
      dragging: false,
      lastX: 0,
      lastY: 0,
      frame: 0,
    };
    let data = null;
    let metrics = null;

    function resize() {
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.floor(window.innerWidth * ratio);
      canvas.height = Math.floor(window.innerHeight * ratio);
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function colorFor(value) {
      const t = Math.max(0, Math.min(1, (value - data.speed.p03) / Math.max(data.speed.p97 - data.speed.p03, 0.001)));
      const stops = [[38,71,134],[34,139,178],[70,178,143],[239,196,84],[217,95,76]];
      const pos = t * (stops.length - 1);
      const i = Math.min(Math.floor(pos), stops.length - 2);
      const f = pos - i;
      return stops[i].map((v, c) => Math.round(v * (1 - f) + stops[i + 1][c] * f));
    }

    function computeMetrics() {
      const xs = data.x;
      const ys = data.y;
      const zs = data.terrain.flat();
      metrics = {
        cx: (Math.min(...xs) + Math.max(...xs)) / 2,
        cy: (Math.min(...ys) + Math.max(...ys)) / 2,
        zMin: Math.min(...zs),
        domain: Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)),
      };
    }

    function project(x, y, z) {
      const dx = x - metrics.cx;
      const dy = y - metrics.cy;
      const ca = Math.cos(state.azimuth);
      const sa = Math.sin(state.azimuth);
      const xr = dx * ca - dy * sa;
      const yr = dx * sa + dy * ca;
      const zr = (z - metrics.zMin) * state.zScale;
      const scale = 0.74 * Math.min(window.innerWidth, window.innerHeight) / metrics.domain * state.zoom;
      const sx = window.innerWidth * 0.54 + state.panX + xr * scale;
      const sy = window.innerHeight * 0.68 + state.panY - (yr * Math.sin(state.elevation) + zr * Math.cos(state.elevation)) * scale;
      return [sx, sy, yr];
    }

    function drawTerrain(items) {
      if (!state.showTerrain) return;
      const nx = data.x.length;
      const ny = data.y.length;
      const stride = Math.max(1, Math.round(nx / 82));
      const zValues = data.terrain.flat();
      const zMin = Math.min(...zValues);
      const zMax = Math.max(...zValues);
      for (let j = 0; j < ny - 1; j += stride) {
        for (let i = 0; i < nx - 1; i += stride) {
          const i2 = Math.min(i + stride, nx - 1);
          const j2 = Math.min(j + stride, ny - 1);
          const pts = [
            project(data.x[i], data.y[j], data.terrain[j][i]),
            project(data.x[i2], data.y[j], data.terrain[j][i2]),
            project(data.x[i2], data.y[j2], data.terrain[j2][i2]),
            project(data.x[i], data.y[j2], data.terrain[j2][i]),
          ];
          const z = (data.terrain[j][i] + data.terrain[j2][i2]) * 0.5;
          const t = (z - zMin) / Math.max(zMax - zMin, 1);
          const shade = [90 + t * 52, 125 + t * 58, 104 + t * 34].map(Math.round);
          items.push({ depth: pts.reduce((a, p) => a + p[2], 0) / pts.length, kind: "poly", pts, color: `rgba(${shade[0]},${shade[1]},${shade[2]},0.92)` });
        }
      }
    }

    function layerByHeight(height) {
      return data.layers.find(layer => layer.height_m === height);
    }

    function drawVectors(items) {
      if (state.vectorDensity === 0) return;
      const layerStride = Math.max(1, Math.ceil(6 / Math.max(state.layersVisible.size, 1)));
      const pointStride = [0, 18, 14, 10, 8, 6][state.vectorDensity];
      for (const height of state.layersVisible) {
        const layer = layerByHeight(height);
        if (!layer) continue;
        const layerIndex = data.layers.indexOf(layer);
        if (layerIndex % layerStride !== 0) continue;
        for (let j = 0; j < layer.y.length; j += pointStride) {
          for (let i = 0; i < layer.x.length; i += pointStride) {
            const speed = layer.speed[j][i];
            const len = 20 + speed * 2.5;
            const mag = Math.max(speed, 0.001);
            const x0 = layer.x[i];
            const y0 = layer.y[j];
            const z0 = layer.z[j][i];
            const x1 = x0 + layer.u[j][i] / mag * len;
            const y1 = y0 + layer.v[j][i] / mag * len;
            const p0 = project(x0, y0, z0);
            const p1 = project(x1, y1, z0 + layer.w[j][i] / mag * len);
            const c = colorFor(speed);
            items.push({ depth: (p0[2] + p1[2]) / 2, kind: "line", a: p0, b: p1, color: `rgba(${c[0]},${c[1]},${c[2]},0.62)`, width: 1.4 });
          }
        }
      }
    }

    function bilinear(grid, x, y, layer) {
      const fx = (x - data.x[0]) / data.dx;
      const fy = (y - data.y[0]) / data.dy;
      const i = Math.floor(fx);
      const j = Math.floor(fy);
      if (i < 0 || j < 0 || i >= data.x.length - 1 || j >= data.y.length - 1) return NaN;
      const tx = fx - i;
      const ty = fy - j;
      return grid[j][i] * (1 - tx) * (1 - ty) + grid[j][i + 1] * tx * (1 - ty) + grid[j + 1][i] * (1 - tx) * ty + grid[j + 1][i + 1] * tx * ty;
    }

    function trace(layer, seedX, seedY, steps) {
      const points = [];
      let x = seedX;
      let y = seedY;
      for (let n = 0; n < steps; n += 1) {
        const u = bilinear(layer.u, x, y, layer);
        const v = bilinear(layer.v, x, y, layer);
        const z = bilinear(layer.z, x, y, layer);
        const speed = bilinear(layer.speed, x, y, layer);
        if (![u, v, z, speed].every(Number.isFinite) || speed < 0.05) break;
        points.push([x, y, z, speed]);
        const step = 22;
        x += u / speed * step;
        y += v / speed * step;
      }
      return points;
    }

    function drawStreamlines(items) {
      if (state.streamDensity === 0) return;
      const seedCount = [0, 5, 7, 10, 13, 16][state.streamDensity];
      for (const height of state.layersVisible) {
        const layer = layerByHeight(height);
        if (!layer) continue;
        const meanU = layer.u.flat().reduce((a, b) => a + b, 0) / layer.u.flat().length;
        const meanV = layer.v.flat().reduce((a, b) => a + b, 0) / layer.v.flat().length;
        const minX = data.x[0], maxX = data.x[data.x.length - 1];
        const minY = data.y[0], maxY = data.y[data.y.length - 1];
        const seeds = [];
        for (let n = 0; n < seedCount; n += 1) {
          const t = seedCount === 1 ? 0.5 : n / (seedCount - 1);
          seeds.push([meanU >= 0 ? minX + 1 : maxX - 1, minY + t * (maxY - minY)]);
          seeds.push([minX + t * (maxX - minX), meanV >= 0 ? minY + 1 : maxY - 1]);
        }
        for (const seed of seeds) {
          const traced = trace(layer, seed[0], seed[1], 160);
          if (traced.length < 8) continue;
          const projected = traced.map(p => project(p[0], p[1], p[2]));
          for (let i = 0; i < projected.length - 1; i += 1) {
            const c = colorFor(traced[i][3]);
            items.push({ depth: (projected[i][2] + projected[i + 1][2]) / 2, kind: "line", a: projected[i], b: projected[i + 1], color: `rgba(${c[0]},${c[1]},${c[2]},0.82)`, width: 2.0 });
          }
        }
      }
    }

    function resetParticles() {
      const heights = [...state.layersVisible];
      state.particles = [];
      if (!heights.length || state.particleCount === 0) return;
      const minX = data.x[0], maxX = data.x[data.x.length - 1];
      const minY = data.y[0], maxY = data.y[data.y.length - 1];
      for (let n = 0; n < state.particleCount; n += 1) {
        const height = heights[n % heights.length];
        state.particles.push({
          x: minX + Math.random() * (maxX - minX),
          y: minY + Math.random() * (maxY - minY),
          height,
          age: Math.random() * 200,
        });
      }
    }

    function drawParticles(items) {
      const minX = data.x[0], maxX = data.x[data.x.length - 1];
      const minY = data.y[0], maxY = data.y[data.y.length - 1];
      for (const p of state.particles) {
        const layer = layerByHeight(p.height);
        if (!layer || !state.layersVisible.has(p.height)) continue;
        const u = bilinear(layer.u, p.x, p.y, layer);
        const v = bilinear(layer.v, p.x, p.y, layer);
        const z = bilinear(layer.z, p.x, p.y, layer);
        const speed = bilinear(layer.speed, p.x, p.y, layer);
        if (![u, v, z, speed].every(Number.isFinite)) {
          p.x = minX + Math.random() * (maxX - minX);
          p.y = minY + Math.random() * (maxY - minY);
          p.age = 0;
          continue;
        }
        const p0 = project(p.x, p.y, z);
        p.x += u / Math.max(speed, 0.01) * 3.8;
        p.y += v / Math.max(speed, 0.01) * 3.8;
        p.age += 1;
        if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY || p.age > 360) {
          p.x = minX + Math.random() * (maxX - minX);
          p.y = minY + Math.random() * (maxY - minY);
          p.age = 0;
        }
        const p1 = project(p.x, p.y, z);
        const c = colorFor(speed);
        items.push({ depth: p0[2], kind: "point", p: p1, color: `rgba(${c[0]},${c[1]},${c[2]},0.88)`, radius: 2.2 });
      }
    }

    function draw() {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      const items = [];
      drawTerrain(items);
      drawVectors(items);
      drawStreamlines(items);
      drawParticles(items);
      items.sort((a, b) => a.depth - b.depth);
      for (const item of items) {
        ctx.strokeStyle = item.color;
        ctx.fillStyle = item.color;
        if (item.kind === "poly") {
          ctx.beginPath();
          ctx.moveTo(item.pts[0][0], item.pts[0][1]);
          for (let i = 1; i < item.pts.length; i += 1) ctx.lineTo(item.pts[i][0], item.pts[i][1]);
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = "rgba(8, 24, 24, 0.16)";
          ctx.lineWidth = 0.7;
          ctx.stroke();
        } else if (item.kind === "line") {
          ctx.lineWidth = item.width;
          ctx.beginPath();
          ctx.moveTo(item.a[0], item.a[1]);
          ctx.lineTo(item.b[0], item.b[1]);
          ctx.stroke();
        } else if (item.kind === "point") {
          ctx.beginPath();
          ctx.arc(item.p[0], item.p[1], item.radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      state.frame += 1;
      requestAnimationFrame(draw);
    }

    function wireControls() {
      const layerChecks = document.getElementById("layerChecks");
      for (const layer of data.layers) {
        state.layersVisible.add(layer.height_m);
        const label = document.createElement("label");
        label.className = "check";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = true;
        input.addEventListener("change", () => {
          if (input.checked) state.layersVisible.add(layer.height_m);
          else state.layersVisible.delete(layer.height_m);
          resetParticles();
        });
        label.append(input, `${layer.height_m} m`);
        layerChecks.append(label);
      }
      document.getElementById("zScale").addEventListener("input", event => { state.zScale = Number(event.target.value); });
      document.getElementById("vectorDensity").addEventListener("input", event => { state.vectorDensity = Number(event.target.value); });
      document.getElementById("streamDensity").addEventListener("input", event => { state.streamDensity = Number(event.target.value); });
      document.getElementById("particleCount").addEventListener("input", event => { state.particleCount = Number(event.target.value); resetParticles(); });
      document.getElementById("terrainToggle").addEventListener("click", event => {
        state.showTerrain = !state.showTerrain;
        event.target.textContent = state.showTerrain ? "Visible" : "Hidden";
      });
      document.getElementById("speedMin").textContent = `${data.speed.p03.toFixed(1)} m/s`;
      document.getElementById("speedMax").textContent = `${data.speed.p97.toFixed(1)} m/s`;
    }

    canvas.addEventListener("pointerdown", event => {
      state.dragging = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      canvas.classList.add("dragging");
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      if (event.shiftKey) {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.azimuth += dx * 0.006;
        state.elevation = Math.max(0.18, Math.min(1.26, state.elevation + dy * 0.004));
      }
    });
    canvas.addEventListener("pointerup", event => {
      state.dragging = false;
      canvas.classList.remove("dragging");
      canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      state.zoom = Math.max(0.35, Math.min(3.5, state.zoom * Math.exp(-event.deltaY * 0.001)));
    }, { passive: false });

    async function boot() {
      resize();
      window.addEventListener("resize", resize);
      data = JSON.parse(document.getElementById("volume-data").textContent);
      computeMetrics();
      wireControls();
      resetParticles();
      requestAnimationFrame(draw);
    }
    boot().catch(error => {
      document.body.innerHTML = `<pre style="padding:24px;color:#ffb8a8">${error.stack || error}</pre>`;
    });
  </script>
</body>
</html>
"""


def _downsample_grid(grid: np.ndarray, stride: int) -> list[list[float]]:
    small = np.asarray(grid[::stride, ::stride], dtype=np.float32)
    return [[round(float(value), 4) for value in row] for row in small]


def _downsample_vector(values: np.ndarray, stride: int) -> list[float]:
    return [round(float(value), 4) for value in np.asarray(values[::stride], dtype=np.float32)]


def _speed_percentiles(layers: list[dict[str, np.ndarray | float]]) -> dict[str, float]:
    speed = np.concatenate([np.asarray(layer["speed"]).ravel() for layer in layers])
    return {
        "min": round(float(np.nanmin(speed)), 4),
        "max": round(float(np.nanmax(speed)), 4),
        "p03": round(float(np.nanpercentile(speed, 3)), 4),
        "p97": round(float(np.nanpercentile(speed, 97)), 4),
    }


def _export_data(
    case_dir: Path,
    output_dir: Path,
    heights_m: tuple[float, ...],
    stride: int,
    skip_sample: bool,
) -> dict:
    x_vertices, y_vertices, terrain = _parse_block_mesh_vertices(case_dir / "system" / "blockMeshDict")
    x_centers, y_centers = _cell_centers(x_vertices, y_vertices)
    terrain_max = _terrain_cell_max(terrain)
    sample_dict = _write_volume_sample_dict(case_dir, x_centers, y_centers, terrain_max, heights_m)
    if not skip_sample:
        _run_openfoam_sampler(case_dir, sample_dict)
    sample_dir = _find_sample_dir(case_dir)
    layers = _load_layers(sample_dir, heights_m, len(x_centers), len(y_centers))

    data = {
        "source": {
            "case_dir": str(case_dir),
            "sample_dir": str(sample_dir),
            "note": "Experimental downsampled OpenFOAM volume data for local browser visualization.",
        },
        "stride": stride,
        "dx": round(float(np.nanmedian(np.diff(x_centers))) * stride, 6),
        "dy": round(float(np.nanmedian(np.diff(y_centers))) * stride, 6),
        "x": _downsample_vector(x_centers, stride),
        "y": _downsample_vector(y_centers, stride),
        "terrain": _downsample_grid(terrain_max, stride),
        "speed": _speed_percentiles(layers),
        "layers": [],
    }
    for layer in layers:
        data["layers"].append(
            {
                "height_m": float(layer["height_m"]),
                "x": _downsample_vector(np.asarray(layer["x"])[0, :], stride),
                "y": _downsample_vector(np.asarray(layer["y"])[:, 0], stride),
                "z": _downsample_grid(np.asarray(layer["z"]), stride),
                "u": _downsample_grid(np.asarray(layer["u"]), stride),
                "v": _downsample_grid(np.asarray(layer["v"]), stride),
                "w": _downsample_grid(np.asarray(layer["w"]), stride),
                "speed": _downsample_grid(np.asarray(layer["speed"]), stride),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "volume_data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an interactive local OpenFOAM volume viewer.")
    parser.add_argument("--case-dir", type=Path, default=_default_case_dir())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--heights-m", default=",".join(str(int(h)) for h in DEFAULT_HEIGHTS_M))
    parser.add_argument("--stride", type=int, default=3, help="Horizontal downsample stride for browser rendering.")
    parser.add_argument("--skip-sample", action="store_true", help="Use existing postProcessing/volumePreview samples.")
    args = parser.parse_args()

    case_dir = args.case_dir
    if not case_dir.exists():
        raise RuntimeError(f"OpenFOAM case directory was not found: {case_dir}")
    output_dir = args.output_dir or case_dir.parent / "volume_views" / "interactive"
    heights_m = tuple(float(value.strip()) for value in args.heights_m.split(",") if value.strip())
    stride = max(1, int(args.stride))

    data = _export_data(case_dir, output_dir, heights_m, stride, args.skip_sample)
    embedded_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (output_dir / "index.html").write_text(HTML.replace("__VOLUME_DATA__", embedded_json), encoding="utf-8")
    manifest = {
        "case_dir": str(case_dir),
        "output_dir": str(output_dir),
        "viewer": str(output_dir / "index.html"),
        "data": str(output_dir / "volume_data.json"),
        "heights_m": heights_m,
        "stride": stride,
        "grid": {
            "nx": len(data["x"]),
            "ny": len(data["y"]),
            "layers": len(data["layers"]),
        },
    }
    (output_dir / "viewer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
