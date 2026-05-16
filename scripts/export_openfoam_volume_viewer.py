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
      <span>Stream trails</span>
      <button id="streamTrailToggle" type="button">On</button>
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
      animateStreamTrails: true,
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
      const zs = data.terrain.flat();
      metrics = {
        cx: (Math.min(...data.terrain_x) + Math.max(...data.terrain_x)) / 2,
        cy: (Math.min(...data.terrain_y) + Math.max(...data.terrain_y)) / 2,
        zMin: Math.min(...zs),
        domain: Math.max(Math.max(...data.terrain_x) - Math.min(...data.terrain_x), Math.max(...data.terrain_y) - Math.min(...data.terrain_y)),
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
      const nx = data.terrain_x.length;
      const ny = data.terrain_y.length;
      const stride = Math.max(1, Math.round(nx / 130));
      const zValues = data.terrain.flat();
      const zMin = Math.min(...zValues);
      const zMax = Math.max(...zValues);
      for (let j = 0; j < ny - 1; j += stride) {
        for (let i = 0; i < nx - 1; i += stride) {
          const i2 = Math.min(i + stride, nx - 1);
          const j2 = Math.min(j + stride, ny - 1);
          const pts = [
            project(data.terrain_x[i], data.terrain_y[j], data.terrain[j][i]),
            project(data.terrain_x[i2], data.terrain_y[j], data.terrain[j][i2]),
            project(data.terrain_x[i2], data.terrain_y[j2], data.terrain[j2][i2]),
            project(data.terrain_x[i], data.terrain_y[j2], data.terrain[j2][i]),
          ];
          const z = (data.terrain[j][i] + data.terrain[j2][i2]) * 0.5;
          const t = (z - zMin) / Math.max(zMax - zMin, 1);
          const shade = Math.round(54 + t * 46);
          items.push({ depth: pts.reduce((a, p) => a + p[2], 0) / pts.length, kind: "poly", pts, color: `rgba(${shade},${shade + 3},${shade + 5},0.86)` });
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
      let streamIndex = 0;
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
          const trailHead = Math.floor(state.frame * 0.55 + streamIndex * 11) % (projected.length + 24);
          const trailTail = trailHead - 22;
          for (let i = 0; i < projected.length - 1; i += 1) {
            const c = colorFor(traced[i][3]);
            const depth = (projected[i][2] + projected[i + 1][2]) / 2;
            if (state.animateStreamTrails) {
              items.push({ depth, kind: "line", a: projected[i], b: projected[i + 1], color: `rgba(${c[0]},${c[1]},${c[2]},0.20)`, width: 1.5 });
              if (i >= trailTail && i <= trailHead) {
                const fade = 1 - Math.abs(i - trailHead) / 22;
                const alpha = 0.34 + 0.58 * Math.max(0, fade);
                const width = 2.6 + 2.4 * Math.max(0, fade);
                items.push({ depth: depth + 0.01, kind: "line", a: projected[i], b: projected[i + 1], color: `rgba(${c[0]},${c[1]},${c[2]},${alpha.toFixed(3)})`, width });
              }
            } else {
              items.push({ depth, kind: "line", a: projected[i], b: projected[i + 1], color: `rgba(${c[0]},${c[1]},${c[2]},0.90)`, width: 3.1 });
            }
          }
          streamIndex += 1;
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
          ctx.strokeStyle = "rgba(210, 224, 228, 0.08)";
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
      document.getElementById("streamTrailToggle").addEventListener("click", event => {
        state.animateStreamTrails = !state.animateStreamTrails;
        event.target.textContent = state.animateStreamTrails ? "On" : "Off";
      });
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


HTML_THREE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PondWind OpenFOAM Volume Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: #10171b;
      color: #eef4f6;
    }
    html, body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: #10171b;
    }
    #viewer {
      position: fixed;
      inset: 0;
    }
    .panel {
      position: fixed;
      left: 16px;
      top: 16px;
      width: min(360px, calc(100vw - 32px));
      background: rgba(13, 19, 23, 0.88);
      border: 1px solid rgba(187, 214, 222, 0.24);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(8px);
      z-index: 2;
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
      background: rgba(13, 19, 23, 0.82);
      border: 1px solid rgba(187, 214, 222, 0.24);
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      color: #c8d8dd;
      z-index: 2;
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
    .status {
      position: fixed;
      left: 16px;
      bottom: 14px;
      color: #8fa5ad;
      font-size: 12px;
      z-index: 2;
    }
  </style>
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.165.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.165.0/examples/jsm/"
      }
    }
  </script>
</head>
<body>
  <div id="viewer"></div>
  <section class="panel">
    <h1>OpenFOAM Volume Viewer</h1>
    <p class="subtle">Drag to rotate, wheel to zoom, Shift+drag to pan. WebGL/Three.js rendering from sampled OpenFOAM volume data.</p>
    <div class="row">
      <span>Vertical scale</span>
      <input id="zScale" type="range" min="1" max="10" step="0.25" value="4">
    </div>
    <div class="row">
      <span>Particles</span>
      <input id="particleCount" type="range" min="0" max="1200" step="25" value="650">
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
      <span>Stream trails</span>
      <button id="streamTrailToggle" type="button">On</button>
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
  <div class="status" id="status">Loading Three.js viewer...</div>
  <script id="volume-data" type="application/json">__VOLUME_DATA__</script>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
    import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
    import { LineMaterial } from "three/addons/lines/LineMaterial.js";

    const data = JSON.parse(document.getElementById("volume-data").textContent);
    const container = document.getElementById("viewer");
    const status = document.getElementById("status");
    const state = {
      zScale: 4,
      showTerrain: true,
      animateStreamTrails: true,
      vectorDensity: 2,
      streamDensity: 3,
      particleCount: 650,
      layersVisible: new Set(data.layers.map(layer => layer.height_m)),
      particles: [],
      frame: 0,
    };

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x10171b);
    scene.fog = new THREE.Fog(0x10171b, 3600, 8500);

    const camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 1, 14000);
    camera.position.set(-2500, 1450, 2500);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 170, 0);
    controls.update();

    const terrainGroup = new THREE.Group();
    const flowGroup = new THREE.Group();
    scene.add(terrainGroup, flowGroup);
    scene.add(new THREE.AmbientLight(0xffffff, 0.78));
    const sun = new THREE.DirectionalLight(0xffffff, 0.72);
    sun.position.set(-1200, 2200, 900);
    scene.add(sun);

    let terrainMesh = null;
    let terrainWire = null;
    let streamLines = null;
    let vectorLines = null;
    let trailLines = null;
    let particlePoints = null;
    let particlePositions = null;
    let particleColors = null;
    let streamCache = [];

    const terrainMinX = Math.min(...data.terrain_x);
    const terrainMaxX = Math.max(...data.terrain_x);
    const terrainMinY = Math.min(...data.terrain_y);
    const terrainMaxY = Math.max(...data.terrain_y);
    const centerX = 0.5 * (terrainMinX + terrainMaxX);
    const centerY = 0.5 * (terrainMinY + terrainMaxY);
    const zMin = Math.min(...data.terrain.flat());
    const zMax = Math.max(...data.terrain.flat());

    function worldX(x) {
      return x - centerX;
    }

    function worldZ(y) {
      return -(y - centerY);
    }

    function worldY(z) {
      return (z - zMin) * state.zScale;
    }

    function colorFor(value) {
      const t = Math.max(0, Math.min(1, (value - data.speed.p03) / Math.max(data.speed.p97 - data.speed.p03, 0.001)));
      const stops = [
        [0x26, 0x47, 0x86],
        [0x22, 0x8b, 0xb2],
        [0x46, 0xb2, 0x8f],
        [0xef, 0xc4, 0x54],
        [0xd9, 0x5f, 0x4c],
      ];
      const pos = t * (stops.length - 1);
      const i = Math.min(Math.floor(pos), stops.length - 2);
      const f = pos - i;
      return [
        (stops[i][0] * (1 - f) + stops[i + 1][0] * f) / 255,
        (stops[i][1] * (1 - f) + stops[i + 1][1] * f) / 255,
        (stops[i][2] * (1 - f) + stops[i + 1][2] * f) / 255,
      ];
    }

    function bilinear(grid, x, y) {
      const fx = (x - data.x[0]) / data.dx;
      const fy = (y - data.y[0]) / data.dy;
      const i = Math.floor(fx);
      const j = Math.floor(fy);
      if (i < 0 || j < 0 || i >= data.x.length - 1 || j >= data.y.length - 1) return NaN;
      const tx = fx - i;
      const ty = fy - j;
      return grid[j][i] * (1 - tx) * (1 - ty) + grid[j][i + 1] * tx * (1 - ty) + grid[j + 1][i] * (1 - tx) * ty + grid[j + 1][i + 1] * tx * ty;
    }

    function layerByHeight(height) {
      return data.layers.find(layer => layer.height_m === height);
    }

    function addLineSegments(name, positions, colors, lineWidth, opacity) {
      const geometry = new LineSegmentsGeometry();
      geometry.setPositions(positions);
      geometry.setColors(colors);
      const material = new LineMaterial({
        vertexColors: true,
        linewidth: lineWidth,
        transparent: opacity < 1,
        opacity,
        depthWrite: false,
      });
      material.resolution.set(window.innerWidth, window.innerHeight);
      const lines = new LineSegments2(geometry, material);
      lines.name = name;
      flowGroup.add(lines);
      return lines;
    }

    function disposeObject(object) {
      if (!object) return;
      object.removeFromParent();
      object.geometry?.dispose();
      object.material?.dispose();
    }

    function buildTerrain() {
      disposeObject(terrainMesh);
      disposeObject(terrainWire);
      const nx = data.terrain_x.length;
      const ny = data.terrain_y.length;
      const positions = new Float32Array(nx * ny * 3);
      const colors = new Float32Array(nx * ny * 3);
      const indices = [];
      let p = 0;
      let c = 0;
      for (let j = 0; j < ny; j += 1) {
        for (let i = 0; i < nx; i += 1) {
          const z = data.terrain[j][i];
          positions[p++] = worldX(data.terrain_x[i]);
          positions[p++] = worldY(z);
          positions[p++] = worldZ(data.terrain_y[j]);
          const t = (z - zMin) / Math.max(zMax - zMin, 1);
          const shade = (42 + t * 50) / 255;
          colors[c++] = shade;
          colors[c++] = shade + 0.012;
          colors[c++] = shade + 0.02;
        }
      }
      for (let j = 0; j < ny - 1; j += 1) {
        for (let i = 0; i < nx - 1; i += 1) {
          const a = j * nx + i;
          const b = a + 1;
          const d = (j + 1) * nx + i;
          const e = d + 1;
          indices.push(a, d, b, b, d, e);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      terrainMesh = new THREE.Mesh(
        geometry,
        new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.92 })
      );
      terrainGroup.add(terrainMesh);

      const wireGeometry = new THREE.WireframeGeometry(geometry);
      terrainWire = new THREE.LineSegments(
        wireGeometry,
        new THREE.LineBasicMaterial({ color: 0xc8d5d8, transparent: true, opacity: 0.055, depthWrite: false })
      );
      terrainGroup.add(terrainWire);
      terrainGroup.visible = state.showTerrain;
    }

    function trace(layer, seedX, seedY, steps) {
      const points = [];
      let x = seedX;
      let y = seedY;
      for (let n = 0; n < steps; n += 1) {
        const u = bilinear(layer.u, x, y);
        const v = bilinear(layer.v, x, y);
        const z = bilinear(layer.z, x, y);
        const speed = bilinear(layer.speed, x, y);
        if (![u, v, z, speed].every(Number.isFinite) || speed < 0.05) break;
        points.push({ x, y, z, speed });
        const step = 24;
        x += (u / speed) * step;
        y += (v / speed) * step;
      }
      return points;
    }

    function streamlineSeeds(layer, count) {
      const minX = data.x[0];
      const maxX = data.x[data.x.length - 1];
      const minY = data.y[0];
      const maxY = data.y[data.y.length - 1];
      const flatU = layer.u.flat();
      const flatV = layer.v.flat();
      const meanU = flatU.reduce((a, b) => a + b, 0) / flatU.length;
      const meanV = flatV.reduce((a, b) => a + b, 0) / flatV.length;
      const seeds = [];
      for (let n = 0; n < count; n += 1) {
        const t = count === 1 ? 0.5 : n / (count - 1);
        seeds.push([meanU >= 0 ? minX + 1 : maxX - 1, minY + t * (maxY - minY)]);
        seeds.push([minX + t * (maxX - minX), meanV >= 0 ? minY + 1 : maxY - 1]);
      }
      return seeds;
    }

    function buildStreams() {
      disposeObject(streamLines);
      disposeObject(vectorLines);
      disposeObject(trailLines);
      streamCache = [];

      const streamPositions = [];
      const streamColors = [];
      const vectorPositions = [];
      const vectorColors = [];
      const seedCount = [0, 5, 7, 10, 13, 16][state.streamDensity];

      if (seedCount > 0) {
        for (const height of state.layersVisible) {
          const layer = layerByHeight(height);
          if (!layer) continue;
          for (const seed of streamlineSeeds(layer, seedCount)) {
            const traced = trace(layer, seed[0], seed[1], 170);
            if (traced.length < 8) continue;
            const points = traced.map(point => new THREE.Vector3(worldX(point.x), worldY(point.z), worldZ(point.y)));
            const speeds = traced.map(point => point.speed);
            streamCache.push({ points, speeds });
            for (let i = 0; i < points.length - 1; i += 1) {
              const c = colorFor(speeds[i]);
              streamPositions.push(points[i].x, points[i].y, points[i].z, points[i + 1].x, points[i + 1].y, points[i + 1].z);
              streamColors.push(...c, ...c);
            }
          }
        }
      }
      if (streamPositions.length) {
        streamLines = addLineSegments("streamlines", streamPositions, streamColors, state.animateStreamTrails ? 1.8 : 3.4, state.animateStreamTrails ? 0.34 : 0.92);
      }

      const pointStride = [0, 18, 14, 10, 8, 6][state.vectorDensity];
      if (pointStride > 0) {
        for (const height of state.layersVisible) {
          const layer = layerByHeight(height);
          if (!layer) continue;
          for (let j = 0; j < layer.y.length; j += pointStride) {
            for (let i = 0; i < layer.x.length; i += pointStride) {
              const speed = layer.speed[j][i];
              const mag = Math.max(speed, 0.001);
              const len = 36 + speed * 3.0;
              const x0 = layer.x[i];
              const y0 = layer.y[j];
              const z0 = layer.z[j][i];
              const x1 = x0 + (layer.u[j][i] / mag) * len;
              const y1 = y0 + (layer.v[j][i] / mag) * len;
              const z1 = z0 + (layer.w[j][i] / mag) * len;
              const c = colorFor(speed);
              vectorPositions.push(worldX(x0), worldY(z0), worldZ(y0), worldX(x1), worldY(z1), worldZ(y1));
              vectorColors.push(...c, ...c);
            }
          }
        }
      }
      if (vectorPositions.length) {
        vectorLines = addLineSegments("vectors", vectorPositions, vectorColors, 1.45, 0.62);
      }
      trailLines = addLineSegments("stream-trails", [], [], 4.8, 0.96);
      trailLines.visible = state.animateStreamTrails;
    }

    function resetParticles() {
      disposeObject(particlePoints);
      state.particles = [];
      if (!state.layersVisible.size || state.particleCount <= 0) return;
      const minX = data.x[0];
      const maxX = data.x[data.x.length - 1];
      const minY = data.y[0];
      const maxY = data.y[data.y.length - 1];
      const heights = [...state.layersVisible];
      particlePositions = new Float32Array(state.particleCount * 3);
      particleColors = new Float32Array(state.particleCount * 3);
      for (let n = 0; n < state.particleCount; n += 1) {
        state.particles.push({
          x: minX + Math.random() * (maxX - minX),
          y: minY + Math.random() * (maxY - minY),
          height: heights[n % heights.length],
          age: Math.random() * 300,
        });
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3).setUsage(THREE.DynamicDrawUsage));
      geometry.setAttribute("color", new THREE.BufferAttribute(particleColors, 3).setUsage(THREE.DynamicDrawUsage));
      particlePoints = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ size: 7.5, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.92, depthWrite: false })
      );
      flowGroup.add(particlePoints);
    }

    function updateParticles() {
      if (!particlePoints || !particlePositions || !particleColors) return;
      const minX = data.x[0];
      const maxX = data.x[data.x.length - 1];
      const minY = data.y[0];
      const maxY = data.y[data.y.length - 1];
      const heights = [...state.layersVisible];
      for (let n = 0; n < state.particles.length; n += 1) {
        const particle = state.particles[n];
        let layer = layerByHeight(particle.height);
        if (!layer || !state.layersVisible.has(particle.height)) {
          particle.height = heights[n % Math.max(heights.length, 1)];
          layer = layerByHeight(particle.height);
        }
        if (!layer) continue;
        const u = bilinear(layer.u, particle.x, particle.y);
        const v = bilinear(layer.v, particle.x, particle.y);
        const z = bilinear(layer.z, particle.x, particle.y);
        const speed = bilinear(layer.speed, particle.x, particle.y);
        if (![u, v, z, speed].every(Number.isFinite)) {
          particle.x = minX + Math.random() * (maxX - minX);
          particle.y = minY + Math.random() * (maxY - minY);
          particle.age = 0;
        } else {
          particle.x += (u / Math.max(speed, 0.01)) * 5.5;
          particle.y += (v / Math.max(speed, 0.01)) * 5.5;
          particle.age += 1;
          if (particle.x < minX || particle.x > maxX || particle.y < minY || particle.y > maxY || particle.age > 420) {
            particle.x = minX + Math.random() * (maxX - minX);
            particle.y = minY + Math.random() * (maxY - minY);
            particle.age = 0;
          }
        }
        const safeZ = Number.isFinite(z) ? z : zMin;
        particlePositions[n * 3] = worldX(particle.x);
        particlePositions[n * 3 + 1] = worldY(safeZ) + 2;
        particlePositions[n * 3 + 2] = worldZ(particle.y);
        const c = colorFor(Number.isFinite(speed) ? speed : data.speed.p03);
        particleColors[n * 3] = c[0];
        particleColors[n * 3 + 1] = c[1];
        particleColors[n * 3 + 2] = c[2];
      }
      particlePoints.geometry.attributes.position.needsUpdate = true;
      particlePoints.geometry.attributes.color.needsUpdate = true;
    }

    function updateTrailLines() {
      if (!trailLines || !state.animateStreamTrails) return;
      const positions = [];
      const colors = [];
      for (let s = 0; s < streamCache.length; s += 1) {
        const stream = streamCache[s];
        const head = Math.floor(state.frame * 0.7 + s * 13) % (stream.points.length + 26);
        const tail = head - 24;
        for (let i = Math.max(0, tail); i < Math.min(stream.points.length - 1, head); i += 1) {
          const fade = 1 - Math.abs(i - head) / 24;
          const c = colorFor(stream.speeds[i]);
          const lift = 3 + 6 * Math.max(0, fade);
          const a = stream.points[i];
          const b = stream.points[i + 1];
          positions.push(a.x, a.y + lift, a.z, b.x, b.y + lift, b.z);
          colors.push(...c, ...c);
        }
      }
      trailLines.geometry.dispose();
      const geometry = new LineSegmentsGeometry();
      geometry.setPositions(positions);
      geometry.setColors(colors);
      trailLines.geometry = geometry;
      trailLines.visible = true;
    }

    function rebuildScene() {
      buildTerrain();
      buildStreams();
      resetParticles();
    }

    function wireControls() {
      const layerChecks = document.getElementById("layerChecks");
      for (const layer of data.layers) {
        const label = document.createElement("label");
        label.className = "check";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = true;
        input.addEventListener("change", () => {
          if (input.checked) state.layersVisible.add(layer.height_m);
          else state.layersVisible.delete(layer.height_m);
          buildStreams();
          resetParticles();
        });
        label.append(input, `${layer.height_m} m`);
        layerChecks.append(label);
      }
      document.getElementById("zScale").addEventListener("input", event => {
        state.zScale = Number(event.target.value);
        rebuildScene();
      });
      document.getElementById("vectorDensity").addEventListener("input", event => {
        state.vectorDensity = Number(event.target.value);
        buildStreams();
      });
      document.getElementById("streamDensity").addEventListener("input", event => {
        state.streamDensity = Number(event.target.value);
        buildStreams();
      });
      document.getElementById("particleCount").addEventListener("input", event => {
        state.particleCount = Number(event.target.value);
        resetParticles();
      });
      document.getElementById("streamTrailToggle").addEventListener("click", event => {
        state.animateStreamTrails = !state.animateStreamTrails;
        event.target.textContent = state.animateStreamTrails ? "On" : "Off";
        buildStreams();
      });
      document.getElementById("terrainToggle").addEventListener("click", event => {
        state.showTerrain = !state.showTerrain;
        event.target.textContent = state.showTerrain ? "Visible" : "Hidden";
        terrainGroup.visible = state.showTerrain;
      });
      document.getElementById("speedMin").textContent = `${data.speed.p03.toFixed(1)} m/s`;
      document.getElementById("speedMax").textContent = `${data.speed.p97.toFixed(1)} m/s`;
    }

    function resize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      for (const lines of [streamLines, vectorLines, trailLines]) {
        if (lines?.material?.resolution) lines.material.resolution.set(window.innerWidth, window.innerHeight);
      }
    }

    function animate() {
      state.frame += 1;
      controls.update();
      updateParticles();
      updateTrailLines();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }

    try {
      wireControls();
      rebuildScene();
      status.textContent = "Three.js / BufferGeometry viewer";
      window.addEventListener("resize", resize);
      resize();
      animate();
    } catch (error) {
      status.textContent = error.stack || String(error);
      status.style.color = "#ffb8a8";
      throw error;
    }
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
    terrain_stride: int,
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
        "terrain_stride": terrain_stride,
        "dx": round(float(np.nanmedian(np.diff(x_centers))) * stride, 6),
        "dy": round(float(np.nanmedian(np.diff(y_centers))) * stride, 6),
        "x": _downsample_vector(x_centers, stride),
        "y": _downsample_vector(y_centers, stride),
        "terrain_x": _downsample_vector(x_centers, terrain_stride),
        "terrain_y": _downsample_vector(y_centers, terrain_stride),
        "terrain": _downsample_grid(terrain_max, terrain_stride),
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
    parser.add_argument("--stride", type=int, default=3, help="Horizontal downsample stride for wind-field browser rendering.")
    parser.add_argument("--terrain-stride", type=int, default=1, help="Horizontal downsample stride for terrain rendering.")
    parser.add_argument("--skip-sample", action="store_true", help="Use existing postProcessing/volumePreview samples.")
    args = parser.parse_args()

    case_dir = args.case_dir
    if not case_dir.exists():
        raise RuntimeError(f"OpenFOAM case directory was not found: {case_dir}")
    output_dir = args.output_dir or case_dir.parent / "volume_views" / "interactive"
    heights_m = tuple(float(value.strip()) for value in args.heights_m.split(",") if value.strip())
    stride = max(1, int(args.stride))
    terrain_stride = max(1, int(args.terrain_stride))

    data = _export_data(case_dir, output_dir, heights_m, stride, terrain_stride, args.skip_sample)
    embedded_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (output_dir / "index.html").write_text(HTML_THREE.replace("__VOLUME_DATA__", embedded_json), encoding="utf-8")
    manifest = {
        "case_dir": str(case_dir),
        "output_dir": str(output_dir),
        "viewer": str(output_dir / "index.html"),
        "data": str(output_dir / "volume_data.json"),
        "heights_m": heights_m,
        "stride": stride,
        "terrain_stride": terrain_stride,
        "grid": {
            "nx": len(data["x"]),
            "ny": len(data["y"]),
            "terrain_nx": len(data["terrain_x"]),
            "terrain_ny": len(data["terrain_y"]),
            "layers": len(data["layers"]),
        },
    }
    (output_dir / "viewer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
