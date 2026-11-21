#!/usr/bin/env python3
"""
StatMath (Estilo Photomath) — Aplicativo Aberto e Portátil para Reconhecimento de Distribuições Estatísticas.

Principais Características:
1. Interface Limpa Estilo Photomath: Inicia com o visor de câmera/upload limpo (sem exemplos pré-carregados
   poluindo a tela e sem selos de porcentagem de acertos/confiança no topo).
2. Três Formas de Entrada Totalmente Funcionais:
   - 📷 Tirar Foto pela Câmera (solicita autorização explícita ao usuário antes de abrir o vídeo e recorta a área do retângulo de foco).
   - 🖼️ Colocar Imagem já tirada pelo celular/dispositivo (PNG, JPG, JPEG, WEBP).
   - ✍️ Escrever/Digitar a Distribuição Diretamente (ex.: "X_1, ..., X_n ~ i.i.d. Pois(4)", "X ~ Gama(3, 1.5)",
     "X ~ Binomial(20, 0.3)", "X ~ Uniforme(-2, 5)", "X ~ Exp(2)") — analisa diretamente o que foi escrito
     mesmo sem nenhuma foto e retorna todas as informações!
3. Card Principal no Topo em LaTeX:
   - Mostra de imediato QUAL É A DISTRIBUIÇÃO, COMO DEVERIA SER ESCRITO e a FUNÇÃO DENSIDADE/MASSA DE PROBABILIDADE (FDP/FMP) + ACUMULADA (FDA).
4. Sub-regiões Retráteis em LaTeX (Passos 1 a 5 que abrem e minimizam para poupar espaço):
   - Passo 1: Como Deveria Ser Escrito Formalmente & Dicas de Escrita
   - Passo 2: Qual é a Variável Aleatória, Independência (X ⊥ Y) e Amostragem i.i.d.
   - Passo 3: Parâmetros, Média, Variância, Mediana, Moda e Curtose
   - Passo 4: Fórmula Geral de Momentos (Momentos Brutos, Centrais e FGM)
   - Passo 5: Todas as Parametrizações (FDP, Acumulada e Como Interpretar Cada Parâmetro)
"""

import base64
import html
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List

from stat_vision.analyzer import StatisticalImageAnalyzer, DistributionReport
from stat_vision.knowledge_base import DISTRIBUTIONS_DB
from stat_vision.symbolic_engine import evaluate_curve_points
from stat_vision.real_evaluator import (
    evaluate_real_test_folder,
    get_cached_real_evaluation,
    resolve_real_test_dir,
)


def _running_inside_streamlit() -> bool:
    try:
        import streamlit as st  # type: ignore
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _compute_nice_ticks(v_min: float, v_max: float, target_count: int = 7, integer_only: bool = False) -> List[float]:
    span = v_max - v_min
    if span <= 1e-12:
        return [v_min]
    raw_step = span / max(target_count, 2)
    mag = 10.0 ** math.floor(math.log10(raw_step))
    residual = raw_step / mag
    if residual <= 1.2:
        step = 1.0 * mag
    elif residual <= 2.2:
        step = 2.0 * mag
    elif residual <= 2.8 and not integer_only:
        step = 2.5 * mag
    elif residual <= 5.5:
        step = 5.0 * mag
    else:
        step = 10.0 * mag
    if integer_only:
        step = max(1.0, math.ceil(step))
    first = math.ceil((v_min - 1e-9) / step) * step
    ticks: List[float] = []
    cur = first
    limit = v_max + 1e-9 * step
    count = 0
    while cur <= limit and count < 40:
        val = 0.0 if abs(cur) < 1e-11 else round(cur, 8)
        ticks.append(val)
        cur += step
        count += 1
    return ticks


def _format_axis_val(val: float, step_hint: float = 1.0, is_int: bool = False) -> str:
    if abs(val) < 1e-11:
        val = 0.0
    if is_int and abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    if abs(val) >= 1000 or (0 < abs(val) < 0.001):
        return f"{val:.2e}"
    if abs(val - round(val)) < 1e-7 and step_hint >= 1.0:
        return str(int(round(val)))
    if step_hint >= 0.5:
        return f"{val:.2f}".rstrip("0").rstrip(".")
    if step_hint >= 0.05:
        return f"{val:.3f}".rstrip("0").rstrip(".")
    return f"{val:.4f}".rstrip("0").rstrip(".")


def _render_svg_curve(
    dist_id: str,
    params: Dict[str, float],
    rv_symbol: str = "x",
    mean_val: Any = None,
) -> str:
    xs, ys, is_discrete = evaluate_curve_points(dist_id, params, num_points=220)
    if not xs or not ys:
        return ""

    spec = DISTRIBUTIONS_DB.get(dist_id)
    used_params = dict(spec.default_params) if spec else {}
    used_params.update({k: float(v) for k, v in params.items() if v is not None})

    # Calcula a probabilidade acumulada F(x) em cada ponto para exibição interativa
    cdfs: List[float] = []
    if is_discrete:
        acc = 0.0
        for y_v in ys:
            acc = min(1.0, acc + y_v)
            cdfs.append(round(acc, 6))
    else:
        acc = 0.0
        cdfs.append(0.0)
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            acc += 0.5 * (ys[i] + ys[i - 1]) * max(dx, 0.0)
            cdfs.append(acc)
        total_area = max(cdfs[-1], 1.0)
        cdfs = [round(min(1.0, v / total_area), 6) for v in cdfs]

    w, h = 680, 320
    pad_l, pad_r, pad_t, pad_b = 72, 28, 30, 52
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    raw_x_min, raw_x_max = min(xs), max(xs)
    raw_y_max = max(max(ys), 1e-6)
    if is_discrete:
        x_min, x_max = raw_x_min - 0.65, raw_x_max + 0.65
    else:
        x_min, x_max = raw_x_min, raw_x_max
    if x_max - x_min < 1e-6:
        x_min -= 1.0
        x_max += 1.0

    y_min = 0.0
    y_max = raw_y_max * 1.18

    x_span = x_max - x_min
    y_span = max(y_max - y_min, 1e-9)

    def to_sx(xv: float) -> float:
        return pad_l + ((xv - x_min) / x_span) * pw

    def to_sy(yv: float) -> float:
        return (h - pad_b) - ((yv - y_min) / y_span) * ph

    x_ticks = _compute_nice_ticks(x_min, x_max, target_count=8, integer_only=is_discrete)
    y_ticks = _compute_nice_ticks(y_min, y_max, target_count=6, integer_only=False)
    x_step_hint = (x_ticks[1] - x_ticks[0]) if len(x_ticks) > 1 else x_span / 6.0
    y_step_hint = (y_ticks[1] - y_ticks[0]) if len(y_ticks) > 1 else y_span / 5.0

    var_sym = rv_symbol if rv_symbol else ("k" if is_discrete else "x")
    y_axis_label = f"P({var_sym.upper()} = {var_sym})" if is_discrete else f"f({var_sym})"
    x_axis_title = f"Eixo X — Valores de {var_sym}"
    y_axis_title = f"Eixo Y — {'Probabilidade ' + y_axis_label if is_discrete else 'Densidade ' + y_axis_label}"

    # Elementos estáticos iniciais do SVG (grade, marcas e valores dos eixos X e Y, curva/barras e rótulos)
    grid_svg: List[str] = []
    axes_svg: List[str] = []

    for xt in x_ticks:
        sx = to_sx(xt)
        if pad_l - 1 <= sx <= w - pad_r + 1:
            lbl = _format_axis_val(xt, x_step_hint, is_int=is_discrete)
            grid_svg.append(
                f'<line x1="{sx:.1f}" y1="{pad_t}" x2="{sx:.1f}" y2="{h - pad_b}" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            axes_svg.append(
                f'<line x1="{sx:.1f}" y1="{h - pad_b}" x2="{sx:.1f}" y2="{h - pad_b + 6}" stroke="#94a3b8" stroke-width="1.5"/>'
            )
            axes_svg.append(
                f'<text x="{sx:.1f}" y="{h - pad_b + 20}" fill="#cbd5e1" font-size="11.5" font-weight="600" text-anchor="middle" font-family="monospace">{lbl}</text>'
            )

    for yt in y_ticks:
        sy = to_sy(yt)
        if pad_t - 1 <= sy <= h - pad_b + 1:
            lbl = _format_axis_val(yt, y_step_hint, is_int=False)
            grid_svg.append(
                f'<line x1="{pad_l}" y1="{sy:.1f}" x2="{w - pad_r}" y2="{sy:.1f}" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            axes_svg.append(
                f'<line x1="{pad_l - 6}" y1="{sy:.1f}" x2="{pad_l}" y2="{sy:.1f}" stroke="#94a3b8" stroke-width="1.5"/>'
            )
            axes_svg.append(
                f'<text x="{pad_l - 10}" y="{sy + 4:.1f}" fill="#cbd5e1" font-size="11.5" font-weight="600" text-anchor="end" font-family="monospace">{lbl}</text>'
            )

    # Títulos dos Eixos X e Y
    axes_svg.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{h - 8}" fill="#94a3b8" font-size="12" font-weight="700" text-anchor="middle">{html.escape(x_axis_title)}</text>'
    )
    axes_svg.append(
        f'<text x="18" y="{pad_t + ph / 2:.1f}" fill="#94a3b8" font-size="12" font-weight="700" text-anchor="middle" transform="rotate(-90 18 {pad_t + ph / 2:.1f})">{html.escape(y_axis_title)}</text>'
    )

    # Curva ou Barras Discretas + Valores nos Pontos
    plot_svg: List[str] = []
    labels_svg: List[str] = []
    y_zero_sy = max(pad_t, min(h - pad_b, to_sy(0.0)))

    if is_discrete:
        bar_w = max(4.0, min(18.0, (pw / max(len(xs) + 2, 4)) * 0.55))
        label_stride = max(1, len(xs) // 10)
        for idx, (x_val, y_val) in enumerate(zip(xs, ys)):
            cx = to_sx(x_val)
            cy = to_sy(y_val)
            plot_svg.append(
                f'<line x1="{cx:.1f}" y1="{y_zero_sy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="#ef4444" stroke-width="{bar_w:.1f}" stroke-linecap="round"/>'
            )
            plot_svg.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#ffffff" stroke="#ef4444" stroke-width="2"/>'
            )
            if (idx % label_stride == 0 or y_val == raw_y_max) and y_val > raw_y_max * 0.03:
                labels_svg.append(
                    f'<text x="{cx:.1f}" y="{max(pad_t + 12, cy - 8):.1f}" fill="#fde047" font-size="10.5" font-weight="700" text-anchor="middle" font-family="monospace">{y_val:.3f}</text>'
                )
    else:
        pts = [f"{to_sx(xv):.1f},{to_sy(yv):.1f}" for xv, yv in zip(xs, ys)]
        poly_pts = f"{to_sx(xs[0]):.1f},{y_zero_sy:.1f} " + " ".join(pts) + f" {to_sx(xs[-1]):.1f},{y_zero_sy:.1f}"
        plot_svg.append(f'<polygon points="{poly_pts}" fill="rgba(239, 68, 68, 0.20)"/>')
        plot_svg.append(f'<polyline fill="none" stroke="#ef4444" stroke-width="3" points="{" ".join(pts)}"/>')

        # Destaca os pontos principais da curva com seus valores (x, f(x))
        peak_idx = max(range(len(ys)), key=lambda i: ys[i])
        px_val, py_val = xs[peak_idx], ys[peak_idx]
        psx, psy = to_sx(px_val), to_sy(py_val)
        labels_svg.append(f'<circle cx="{psx:.1f}" cy="{psy:.1f}" r="4.5" fill="#fde047" stroke="#090d16" stroke-width="1.5"/>')
        labels_svg.append(
            f'<text x="{min(w - pad_r - 55, max(pad_l + 55, psx)):.1f}" y="{max(pad_t + 14, psy - 9):.1f}" fill="#fde047" font-size="11" font-weight="700" text-anchor="middle" font-family="monospace">'
            f'Pico: ({px_val:.3g}, {py_val:.4g})</text>'
        )

    # Linha vertical tracejada da Média E[X]
    mean_num: Optional[float] = float(mean_val) if isinstance(mean_val, (int, float)) else None
    if mean_num is not None and x_min <= mean_num <= x_max:
        msx = to_sx(mean_num)
        plot_svg.append(
            f'<line x1="{msx:.1f}" y1="{pad_t}" x2="{msx:.1f}" y2="{h - pad_b}" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="5,4"/>'
        )
        labels_svg.append(
            f'<text x="{min(w - pad_r - 45, max(pad_l + 45, msx + 6)):.1f}" y="{pad_t + 14}" fill="#38bdf8" font-size="11" font-weight="700" font-family="monospace">E[X]={mean_num:.4g}</text>'
        )

    # Serializa os dados matemáticos da distribuição para o controlador JavaScript de Zoom/Pan/Crosshair
    chart_payload = json.dumps(
        {
            "distId": dist_id,
            "params": used_params,
            "isDiscrete": is_discrete,
            "varSym": var_sym,
            "yLabel": y_axis_label,
            "xAxisTitle": x_axis_title,
            "yAxisTitle": y_axis_title,
            "xs": [round(v, 6) for v in xs],
            "ys": [round(v, 6) for v in ys],
            "cdfs": cdfs,
            "initXMin": round(x_min, 6),
            "initXMax": round(x_max, 6),
            "initYMin": round(y_min, 6),
            "initYMax": round(y_max, 6),
            "rawYMax": round(raw_y_max, 6),
            "meanVal": round(mean_num, 6) if mean_num is not None else None,
        }
    )

    peak_idx_init = max(range(len(ys)), key=lambda i: ys[i])
    init_x_disp = _format_axis_val(xs[peak_idx_init], x_step_hint, is_int=is_discrete)
    init_y_disp = f"{ys[peak_idx_init]:.5g}"
    init_cdf_disp = f"{cdfs[peak_idx_init]:.4f}"

    return f"""
    <div style="background:#090d16;border:1.5px solid #26334d;border-radius:16px;padding:14px;margin-top:12px;">
      <!-- Cabeçalho e Leitor Interativo de Valores (x, y, Acumulada) -->
      <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px;">
        <div style="font-weight:800;font-size:0.95rem;color:#f8fafc;display:flex;align-items:center;gap:6px;">
          📊 Gráfico Interativo da Distribuição
          <span style="font-size:0.75rem;color:#94a3b8;font-weight:500;">(Role o mouse para Zoom • Arraste para Mover • Passe o cursor para ver os valores)</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;font-family:monospace;font-size:0.84rem;">
          <span style="background:#111827;border:1px solid #334155;padding:4px 10px;border-radius:8px;color:#fda4af;">
            <strong>{html.escape(var_sym)} =</strong> <span id="chartHoverX">{init_x_disp}</span>
          </span>
          <span style="background:#111827;border:1px solid #334155;padding:4px 10px;border-radius:8px;color:#fde047;">
            <strong>{html.escape(y_axis_label)} =</strong> <span id="chartHoverY">{init_y_disp}</span>
          </span>
          <span style="background:#111827;border:1px solid #334155;padding:4px 10px;border-radius:8px;color:#38bdf8;">
            <strong>F({html.escape(var_sym)}) =</strong> <span id="chartHoverCDF">{init_cdf_disp}</span>
          </span>
        </div>
      </div>

      <!-- Barra de Ferramentas de Zoom e Navegação nos Eixos X e Y -->
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;background:#111827;border:1px solid #1e293b;border-radius:10px;padding:8px 10px;margin-bottom:10px;">
        <span style="font-size:0.78rem;color:#cbd5e1;font-weight:700;">🔍 Zoom no Gráfico:</span>
        <button type="button" class="mini-ctrl-btn" onclick="distChartZoomStep(1.25)" title="Diminuir Zoom no Gráfico">➖</button>
        <input type="range" id="distChartZoomSlider" class="zoom-slider" min="100" max="1000" step="10" value="100" style="max-width:140px;" oninput="distChartSetZoomPct(this.value)"/>
        <button type="button" class="mini-ctrl-btn" onclick="distChartZoomStep(0.80)" title="Aumentar Zoom no Gráfico">➕</button>
        <span id="distChartZoomBadge" style="min-width:46px;text-align:center;font-weight:800;color:#fb7185;font-size:0.80rem;">100%</span>

        <span style="color:#334155;margin:0 2px;">|</span>
        <button type="button" class="mini-ctrl-btn" onclick="distChartPanStep(-0.18, 0)" title="Mover Eixo X para Esquerda">⬅️</button>
        <button type="button" class="mini-ctrl-btn" onclick="distChartPanStep(0.18, 0)" title="Mover Eixo X para Direita">➡️</button>
        <button type="button" class="mini-ctrl-btn" onclick="distChartPanStep(0, 0.18)" title="Mover Eixo Y para Cima">⬆️</button>
        <button type="button" class="mini-ctrl-btn" onclick="distChartPanStep(0, -0.18)" title="Mover Eixo Y para Baixo">⬇️</button>

        <div style="margin-left:auto;display:flex;gap:6px;">
          <button type="button" id="distChartToggleLabelsBtn" class="mini-ctrl-btn" onclick="distChartTogglePointLabels()" style="background:#1e3a8a;border-color:#3b82f6;">
            🏷️ Valores: ON
          </button>
          <button type="button" class="mini-ctrl-btn" onclick="distChartResetView()" style="background:#be123c;border-color:#fb7185;">
            🎯 Resetar Eixos
          </button>
        </div>
      </div>

      <!-- Área Gráfica SVG com Eixos X/Y Graduados, Zoom, Pan e Mira (Crosshair) -->
      <div style="position:relative;user-select:none;">
        <svg id="distInteractiveSvg" viewBox="0 0 {w} {h}" width="100%" height="310"
             style="background:#060911;border-radius:12px;border:1px solid #1e293b;cursor:crosshair;touch-action:none;display:block;">
          <defs>
            <clipPath id="distPlotClip">
              <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}"/>
            </clipPath>
          </defs>
          <g id="distGridLayer">{"".join(grid_svg)}</g>
          <g id="distPlotLayer" clip-path="url(#distPlotClip)">{"".join(plot_svg)}</g>
          <g id="distLabelsLayer" clip-path="url(#distPlotClip)">{"".join(labels_svg)}</g>
          <!-- Eixos Principais X e Y -->
          <line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="#64748b" stroke-width="1.8"/>
          <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="#64748b" stroke-width="1.8"/>
          <g id="distAxesTicksLayer">{"".join(axes_svg)}</g>
          <g id="distCrosshairLayer"></g>
        </svg>
      </div>
      <script>
      (function() {{
        const CFG = {chart_payload};
        const W = {w}, H = {h}, PAD_L = {pad_l}, PAD_R = {pad_r}, PAD_T = {pad_t}, PAD_B = {pad_b};
        const PW = W - PAD_L - PAD_R, PH = H - PAD_T - PAD_B;

        let xMin = CFG.initXMin, xMax = CFG.initXMax;
        let yMin = CFG.initYMin, yMax = CFG.initYMax;
        let showPointLabels = true;
        let isPanning = false, panStartX = 0, panStartY = 0;
        let panInitXMin = 0, panInitXMax = 0, panInitYMin = 0, panInitYMax = 0;
        let lastHoverMathX = null;

        const svgEl = document.getElementById('distInteractiveSvg');
        const gridLayer = document.getElementById('distGridLayer');
        const plotLayer = document.getElementById('distPlotLayer');
        const labelsLayer = document.getElementById('distLabelsLayer');
        const axesLayer = document.getElementById('distAxesTicksLayer');
        const crossLayer = document.getElementById('distCrosshairLayer');

        // Aproximação de Lanczos para ln(Gamma(z)) em JS para reamostragem precisa ao dar Zoom profundo
        function lgammaJS(z) {{
          if (z <= 0) return 0;
          const g = 7;
          const p = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
          ];
          if (z < 0.5) {{
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - lgammaJS(1 - z);
          }}
          z -= 1;
          let x = p[0];
          for (let i = 1; i < g + 2; i++) x += p[i] / (z + i);
          const t = z + g + 0.5;
          return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
        }}

        function evalContinuousPdf(x) {{
          const p = CFG.params || {{}};
          const d = CFG.distId;
          if (d === 'normal') {{
            const mu = p.mu !== undefined ? p.mu : 0;
            const sig = Math.max(p.sigma !== undefined ? p.sigma : 1, 1e-6);
            return (1 / (sig * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mu) / sig, 2));
          }} else if (d === 'exponential') {{
            if (x < 0) return 0;
            const lam = Math.max(p.lam !== undefined ? p.lam : 1, 1e-6);
            return lam * Math.exp(-lam * x);
          }} else if (d === 'uniform') {{
            const a = p.a !== undefined ? p.a : 0;
            const b = p.b !== undefined ? p.b : 1;
            const L = Math.max(b - a, 1e-6);
            return (x >= a && x <= b) ? (1 / L) : 0;
          }} else if (d === 'gamma') {{
            if (x < 0) return 0;
            const a = Math.max(p.alpha !== undefined ? p.alpha : 3, 0.05);
            const b = Math.max(p.beta !== undefined ? p.beta : 1, 1e-6);
            if (x === 0) return Math.abs(a - 1) < 1e-9 ? b : 0;
            return Math.exp(a * Math.log(b) - lgammaJS(a) + (a - 1) * Math.log(x) - b * x);
          }} else if (d === 'beta') {{
            if (x < 0 || x > 1) return 0;
            const a = Math.max(p.alpha !== undefined ? p.alpha : 2, 0.05);
            const b = Math.max(p.beta !== undefined ? p.beta : 5, 0.05);
            const logB = lgammaJS(a) + lgammaJS(b) - lgammaJS(a + b);
            if (x === 0) return Math.abs(a - 1) < 1e-9 ? Math.exp(-logB) : 0;
            if (x === 1) return Math.abs(b - 1) < 1e-9 ? Math.exp(-logB) : 0;
            return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
          }} else if (d === 'chi_square') {{
            if (x < 0) return 0;
            const k = Math.max(p.k !== undefined ? p.k : 4, 0.5);
            if (x === 0) return Math.abs(k - 2) < 1e-9 ? 0.5 : 0;
            return Math.exp(-0.5 * k * Math.log(2) - lgammaJS(0.5 * k) + (0.5 * k - 1) * Math.log(x) - 0.5 * x);
          }} else if (d === 'student_t') {{
            const nu = Math.max(p.nu !== undefined ? p.nu : 5, 0.5);
            const nc = Math.exp(lgammaJS(0.5 * (nu + 1)) - lgammaJS(0.5 * nu)) / Math.sqrt(nu * Math.PI);
            return nc * Math.pow(1 + (x * x) / nu, -0.5 * (nu + 1));
          }} else if (d === 'lognormal') {{
            if (x <= 0) return 0;
            const mu = p.mu !== undefined ? p.mu : 0;
            const sig = Math.max(p.sigma !== undefined ? p.sigma : 0.6, 1e-6);
            return (1 / (x * sig * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((Math.log(x) - mu) / sig, 2));
          }} else if (d === 'weibull') {{
            if (x <= 0) return 0;
            const k = Math.max(p.k !== undefined ? p.k : 2, 0.1);
            const lam = Math.max(p.lam !== undefined ? p.lam : 1, 1e-6);
            return (k / lam) * Math.pow(x / lam, k - 1) * Math.exp(-Math.pow(x / lam, k));
          }} else if (d === 'cauchy') {{
            const x0 = p.x0 !== undefined ? p.x0 : 0;
            const gamma = Math.max(p.gamma !== undefined ? p.gamma : 1, 1e-6);
            return 1 / (Math.PI * gamma * (1 + Math.pow((x - x0) / gamma, 2)));
          }}
          return 0;
        }}

        function interpCdf(xVal) {{
          const xs = CFG.xs, cdfs = CFG.cdfs;
          if (!xs.length) return 0;
          if (xVal <= xs[0]) return CFG.isDiscrete ? (xVal < xs[0] ? 0 : cdfs[0]) : 0;
          if (xVal >= xs[xs.length - 1]) return 1;
          for (let i = 0; i < xs.length - 1; i++) {{
            if (xVal >= xs[i] && xVal <= xs[i + 1]) {{
              if (CFG.isDiscrete) return cdfs[i];
              const t = (xVal - xs[i]) / Math.max(xs[i + 1] - xs[i], 1e-9);
              return cdfs[i] + t * (cdfs[i + 1] - cdfs[i]);
            }}
          }}
          return cdfs[cdfs.length - 1];
        }}

        function computeNiceTicksJS(vMin, vMax, targetCount, intOnly) {{
          const span = vMax - vMin;
          if (span <= 1e-12) return [vMin];
          const rawStep = span / Math.max(targetCount, 2);
          const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
          const res = rawStep / mag;
          let step;
          if (res <= 1.2) step = 1 * mag;
          else if (res <= 2.2) step = 2 * mag;
          else if (res <= 2.8 && !intOnly) step = 2.5 * mag;
          else if (res <= 5.5) step = 5 * mag;
          else step = 10 * mag;
          if (intOnly) step = Math.max(1, Math.ceil(step));
          const first = Math.ceil((vMin - 1e-9) / step) * step;
          const out = [];
          let cur = first, cnt = 0;
          while (cur <= vMax + 1e-9 * step && cnt < 40) {{
            out.push(Math.abs(cur) < 1e-11 ? 0 : Number(cur.toPrecision(10)));
            cur += step;
            cnt++;
          }}
          return {{ ticks: out, step: step }};
        }}

        function fmtAxisNum(v, step, intOnly) {{
          if (Math.abs(v) < 1e-11) v = 0;
          if (intOnly && Math.abs(v - Math.round(v)) < 1e-6) return String(Math.round(v));
          if (Math.abs(v) >= 10000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) return v.toExponential(2);
          if (step >= 1 && Math.abs(v - Math.round(v)) < 1e-6) return String(Math.round(v));
          let dec = step >= 0.5 ? 2 : (step >= 0.05 ? 3 : (step >= 0.005 ? 4 : 5));
          return Number(v.toFixed(dec)).toString();
        }}

        function toSX(xv) {{ return PAD_L + ((xv - xMin) / Math.max(xMax - xMin, 1e-12)) * PW; }}
        function toSY(yv) {{ return (H - PAD_B) - ((yv - yMin) / Math.max(yMax - yMin, 1e-12)) * PH; }}
        function fromSX(sx) {{ return xMin + ((sx - PAD_L) / PW) * (xMax - xMin); }}
        function fromSY(sy) {{ return yMin + (((H - PAD_B) - sy) / PH) * (yMax - yMin); }}

        function redrawChart() {{
          const xInfo = computeNiceTicksJS(xMin, xMax, 8, CFG.isDiscrete);
          const yInfo = computeNiceTicksJS(yMin, yMax, 6, false);

          let gHtml = '', aHtml = '';
          xInfo.ticks.forEach(xt => {{
            const sx = toSX(xt);
            if (sx >= PAD_L - 1 && sx <= W - PAD_R + 1) {{
              const lbl = fmtAxisNum(xt, xInfo.step, CFG.isDiscrete);
              gHtml += `<line x1="${{sx.toFixed(1)}}" y1="${{PAD_T}}" x2="${{sx.toFixed(1)}}" y2="${{H - PAD_B}}" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>`;
              aHtml += `<line x1="${{sx.toFixed(1)}}" y1="${{H - PAD_B}}" x2="${{sx.toFixed(1)}}" y2="${{H - PAD_B + 6}}" stroke="#94a3b8" stroke-width="1.5"/>`;
              aHtml += `<text x="${{sx.toFixed(1)}}" y="${{H - PAD_B + 20}}" fill="#cbd5e1" font-size="11.5" font-weight="600" text-anchor="middle" font-family="monospace">${{lbl}}</text>`;
            }}
          }});

          yInfo.ticks.forEach(yt => {{
            const sy = toSY(yt);
            if (sy >= PAD_T - 1 && sy <= H - PAD_B + 1) {{
              const lbl = fmtAxisNum(yt, yInfo.step, false);
              gHtml += `<line x1="${{PAD_L}}" y1="${{sy.toFixed(1)}}" x2="${{W - PAD_R}}" y2="${{sy.toFixed(1)}}" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>`;
              aHtml += `<line x1="${{PAD_L - 6}}" y1="${{sy.toFixed(1)}}" x2="${{PAD_L}}" y2="${{sy.toFixed(1)}}" stroke="#94a3b8" stroke-width="1.5"/>`;
              aHtml += `<text x="${{PAD_L - 10}}" y="${{(sy + 4).toFixed(1)}}" fill="#cbd5e1" font-size="11.5" font-weight="600" text-anchor="end" font-family="monospace">${{lbl}}</text>`;
            }}
          }});

          aHtml += `<text x="${{(PAD_L + PW / 2).toFixed(1)}}" y="${{H - 8}}" fill="#94a3b8" font-size="12" font-weight="700" text-anchor="middle">${{CFG.xAxisTitle}}</text>`;
          aHtml += `<text x="18" y="${{(PAD_T + PH / 2).toFixed(1)}}" fill="#94a3b8" font-size="12" font-weight="700" text-anchor="middle" transform="rotate(-90 18 ${{(PAD_T + PH / 2).toFixed(1)}})">${{CFG.yAxisTitle}}</text>`;

          let pHtml = '', lHtml = '';
          const yZeroSY = Math.max(PAD_T, Math.min(H - PAD_B, toSY(0)));

          if (CFG.isDiscrete) {{
            const visibleIdx = [];
            for (let i = 0; i < CFG.xs.length; i++) {{
              if (CFG.xs[i] >= xMin - 0.5 && CFG.xs[i] <= xMax + 0.5) visibleIdx.push(i);
            }}
            const barW = Math.max(4, Math.min(22, (PW / Math.max(visibleIdx.length + 2, 4)) * 0.55));
            const stride = visibleIdx.length <= 14 ? 1 : Math.ceil(visibleIdx.length / 12);
            visibleIdx.forEach((idx, vPos) => {{
              const xv = CFG.xs[idx], yv = CFG.ys[idx];
              const cx = toSX(xv), cy = toSY(yv);
              pHtml += `<line x1="${{cx.toFixed(1)}}" y1="${{yZeroSY.toFixed(1)}}" x2="${{cx.toFixed(1)}}" y2="${{cy.toFixed(1)}}" stroke="#ef4444" stroke-width="${{barW.toFixed(1)}}" stroke-linecap="round"/>`;
              pHtml += `<circle cx="${{cx.toFixed(1)}}" cy="${{cy.toFixed(1)}}" r="4" fill="#ffffff" stroke="#ef4444" stroke-width="2"/>`;
              if (showPointLabels && (vPos % stride === 0 || yv === CFG.rawYMax)) {{
                lHtml += `<text x="${{cx.toFixed(1)}}" y="${{Math.max(PAD_T + 12, cy - 8).toFixed(1)}}" fill="#fde047" font-size="10.5" font-weight="700" text-anchor="middle" font-family="monospace">${{yv.toFixed(3)}}</text>`;
              }}
            }});
          }} else {{
            const N = 220;
            const pts = [];
            let sampleXs = [];
            for (let i = 0; i < N; i++) {{
              sampleXs.push(xMin + ((xMax - xMin) * i) / (N - 1));
            }}
            if (CFG.distId === 'uniform') {{
              const a = CFG.params.a !== undefined ? CFG.params.a : 0;
              const b = CFG.params.b !== undefined ? CFG.params.b : 1;
              const eps = 1e-6 * Math.max(b - a, 1);
              sampleXs.push(a - eps, a, b, b + eps);
              sampleXs.sort((u, v) => u - v);
            }}
            let bestX = sampleXs[0], bestY = -1;
            sampleXs.forEach(xv => {{
              const yv = evalContinuousPdf(xv);
              if (yv > bestY) {{ bestY = yv; bestX = xv; }}
              pts.push(`${{toSX(xv).toFixed(1)}},${{toSY(yv).toFixed(1)}}`);
            }});
            const polyPts = `${{toSX(sampleXs[0]).toFixed(1)}},${{yZeroSY.toFixed(1)}} ` + pts.join(' ') + ` ${{toSX(sampleXs[sampleXs.length - 1]).toFixed(1)}},${{yZeroSY.toFixed(1)}}`;
            pHtml += `<polygon points="${{polyPts}}" fill="rgba(239, 68, 68, 0.20)"/>`;
            pHtml += `<polyline fill="none" stroke="#ef4444" stroke-width="3" points="${{pts.join(' ')}}"/>`;

            if (showPointLabels) {{
              // Marca 5 pontos distribuídos na janela de zoom atual com seus valores (x, y)
              const markCount = 5;
              for (let m = 1; m <= markCount; m++) {{
                const mx = xMin + ((xMax - xMin) * m) / (markCount + 1);
                const my = evalContinuousPdf(mx);
                if (my >= yMin && my <= yMax) {{
                  const msx = toSX(mx), msy = toSY(my);
                  lHtml += `<circle cx="${{msx.toFixed(1)}}" cy="${{msy.toFixed(1)}}" r="3.5" fill="#fde047" stroke="#090d16" stroke-width="1.2"/>`;
                  lHtml += `<text x="${{msx.toFixed(1)}}" y="${{Math.max(PAD_T + 13, msy - 8).toFixed(1)}}" fill="#fde047" font-size="10" font-weight="700" text-anchor="middle" font-family="monospace">(${{Number(mx.toFixed(2))}}, ${{Number(my.toFixed(3))}})</text>`;
                }}
              }}
            }}
          }}

          if (CFG.meanVal !== null && CFG.meanVal >= xMin && CFG.meanVal <= xMax) {{
            const msx = toSX(CFG.meanVal);
            pHtml += `<line x1="${{msx.toFixed(1)}}" y1="${{PAD_T}}" x2="${{msx.toFixed(1)}}" y2="${{H - PAD_B}}" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="5,4"/>`;
            lHtml += `<text x="${{Math.min(W - PAD_R - 45, Math.max(PAD_L + 45, msx + 6)).toFixed(1)}}" y="${{PAD_T + 14}}" fill="#38bdf8" font-size="11" font-weight="700" font-family="monospace">E[X]=${{Number(CFG.meanVal.toFixed(4))}}</text>`;
          }}

          gridLayer.innerHTML = gHtml;
          plotLayer.innerHTML = pHtml;
          labelsLayer.innerHTML = lHtml;
          axesLayer.innerHTML = aHtml;

          const initSpan = Math.max(CFG.initXMax - CFG.initXMin, 1e-9);
          const curSpan = Math.max(xMax - xMin, 1e-9);
          const zoomPct = Math.max(25, Math.min(2000, Math.round((initSpan / curSpan) * 100)));
          const badge = document.getElementById('distChartZoomBadge');
          if (badge) badge.innerText = zoomPct + '%';
          const slider = document.getElementById('distChartZoomSlider');
          if (slider) slider.value = Math.max(100, Math.min(1000, zoomPct));

          if (lastHoverMathX !== null) renderCrosshairAtMathX(lastHoverMathX);
        }}

        function renderCrosshairAtMathX(mathX) {{
          let snapX = mathX, snapY = 0, snapCdf = 0;
          if (CFG.isDiscrete) {{
            let bestDist = Infinity, bestIdx = 0;
            for (let i = 0; i < CFG.xs.length; i++) {{
              const d = Math.abs(CFG.xs[i] - mathX);
              if (d < bestDist) {{ bestDist = d; bestIdx = i; }}
            }}
            snapX = CFG.xs[bestIdx];
            snapY = CFG.ys[bestIdx];
            snapCdf = CFG.cdfs[bestIdx];
          }} else {{
            snapX = Math.max(xMin, Math.min(xMax, mathX));
            snapY = evalContinuousPdf(snapX);
            snapCdf = interpCdf(snapX);
          }}

          const cx = toSX(snapX);
          const cy = toSY(snapY);
          const xStr = CFG.isDiscrete ? String(Math.round(snapX)) : Number(snapX.toFixed(4)).toString();
          const yStr = Number(snapY.toFixed(5)).toString();
          const cdfStr = Number(snapCdf.toFixed(4)).toString();

          const hx = document.getElementById('chartHoverX');
          const hy = document.getElementById('chartHoverY');
          const hcdf = document.getElementById('chartHoverCDF');
          if (hx) hx.innerText = xStr;
          if (hy) hy.innerText = yStr;
          if (hcdf) hcdf.innerText = cdfStr;

          if (cx < PAD_L || cx > W - PAD_R) {{
            crossLayer.innerHTML = '';
            return;
          }}
          const clampedCY = Math.max(PAD_T, Math.min(H - PAD_B, cy));
          const tipW = 168, tipH = 44;
          const tipX = (cx + 14 + tipW > W - PAD_R) ? (cx - tipW - 14) : (cx + 14);
          const tipY = Math.max(PAD_T + 4, Math.min(H - PAD_B - tipH - 4, clampedCY - tipH - 6));

          crossLayer.innerHTML = `
            <!-- Linhas Tracejadas de Projeção até os Eixos X e Y -->
            <line x1="${{cx.toFixed(1)}}" y1="${{PAD_T}}" x2="${{cx.toFixed(1)}}" y2="${{H - PAD_B}}" stroke="#fb7185" stroke-width="1.4" stroke-dasharray="4,3"/>
            <line x1="${{PAD_L}}" y1="${{clampedCY.toFixed(1)}}" x2="${{cx.toFixed(1)}}" y2="${{clampedCY.toFixed(1)}}" stroke="#fde047" stroke-width="1.4" stroke-dasharray="4,3"/>
            <!-- Etiqueta Destacada no Eixo X -->
            <rect x="${{(cx - 28).toFixed(1)}}" y="${{H - PAD_B + 3}}" width="56" height="19" rx="5" fill="#e11d48"/>
            <text x="${{cx.toFixed(1)}}" y="${{H - PAD_B + 16}}" fill="#ffffff" font-size="11" font-weight="800" text-anchor="middle" font-family="monospace">${{xStr}}</text>
            <!-- Etiqueta Destacada no Eixo Y -->
            <rect x="${{PAD_L - 64}}" y="${{(clampedCY - 10).toFixed(1)}}" width="60" height="20" rx="5" fill="#ca8a04"/>
            <text x="${{PAD_L - 34}}" y="${{(clampedCY + 4).toFixed(1)}}" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle" font-family="monospace">${{Number(snapY.toFixed(4))}}</text>
            <!-- Ponto Brilhante na Curva/Barra -->
            <circle cx="${{cx.toFixed(1)}}" cy="${{clampedCY.toFixed(1)}}" r="6" fill="#fb7185" stroke="#ffffff" stroke-width="2"/>
            <!-- Tooltip Flutuante com (x, y) -->
            <rect x="${{tipX.toFixed(1)}}" y="${{tipY.toFixed(1)}}" width="${{tipW}}" height="${{tipH}}" rx="8" fill="rgba(15, 23, 42, 0.94)" stroke="#fb7185" stroke-width="1.5"/>
            <text x="${{(tipX + 10).toFixed(1)}}" y="${{(tipY + 17).toFixed(1)}}" fill="#fda4af" font-size="11" font-weight="700" font-family="monospace">${{CFG.varSym}} = ${{xStr}}</text>
            <text x="${{(tipX + 10).toFixed(1)}}" y="${{(tipY + 34).toFixed(1)}}" fill="#fde047" font-size="11" font-weight="700" font-family="monospace">${{CFG.yLabel}} = ${{yStr}}</text>
          `;
        }}

        function getSvgCoords(evt) {{
          const rect = svgEl.getBoundingClientRect();
          const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
          const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
          return {{
            sx: ((clientX - rect.left) / Math.max(rect.width, 1)) * W,
            sy: ((clientY - rect.top) / Math.max(rect.height, 1)) * H
          }};
        }}

        svgEl.addEventListener('mousemove', function(e) {{
          const pt = getSvgCoords(e);
          if (isPanning) {{
            const dxPx = pt.sx - panStartX;
            const dyPx = pt.sy - panStartY;
            const spanX = panInitXMax - panInitXMin;
            const spanY = panInitYMax - panInitYMin;
            xMin = panInitXMin - (dxPx / PW) * spanX;
            xMax = panInitXMax - (dxPx / PW) * spanX;
            const newYMin = Math.max(0, panInitYMin + (dyPx / PH) * spanY);
            yMin = newYMin;
            yMax = newYMin + spanY;
            redrawChart();
            return;
          }}
          if (pt.sx >= PAD_L && pt.sx <= W - PAD_R) {{
            lastHoverMathX = fromSX(pt.sx);
            renderCrosshairAtMathX(lastHoverMathX);
          }}
        }});

        svgEl.addEventListener('mousedown', function(e) {{
          const pt = getSvgCoords(e);
          isPanning = true;
          panStartX = pt.sx;
          panStartY = pt.sy;
          panInitXMin = xMin; panInitXMax = xMax;
          panInitYMin = yMin; panInitYMax = yMax;
        }});
        window.addEventListener('mouseup', function() {{ isPanning = false; }});

        svgEl.addEventListener('mouseleave', function() {{
          if (!isPanning) {{
            lastHoverMathX = null;
            crossLayer.innerHTML = '';
          }}
        }});

        svgEl.addEventListener('wheel', function(e) {{
          e.preventDefault();
          const pt = getSvgCoords(e);
          const factor = e.deltaY < 0 ? 0.82 : 1.22;
          const anchorX = (pt.sx >= PAD_L && pt.sx <= W - PAD_R) ? fromSX(pt.sx) : 0.5 * (xMin + xMax);
          const anchorY = (pt.sy >= PAD_T && pt.sy <= H - PAD_B) ? fromSY(pt.sy) : 0.5 * (yMin + yMax);
          const newSpanX = Math.max(1e-4, Math.min((CFG.initXMax - CFG.initXMin) * 4, (xMax - xMin) * factor));
          const newSpanY = Math.max(1e-5, Math.min((CFG.initYMax - CFG.initYMin) * 4, (yMax - yMin) * factor));
          const rx = (anchorX - xMin) / Math.max(xMax - xMin, 1e-12);
          const ry = (anchorY - yMin) / Math.max(yMax - yMin, 1e-12);
          xMin = anchorX - rx * newSpanX;
          xMax = xMin + newSpanX;
          yMin = Math.max(0, anchorY - ry * newSpanY);
          yMax = yMin + newSpanY;
          redrawChart();
        }}, {{ passive: false }});

        window.distChartZoomStep = function(factor) {{
          const midX = 0.5 * (xMin + xMax);
          const newSpanX = Math.max(1e-4, Math.min((CFG.initXMax - CFG.initXMin) * 4, (xMax - xMin) * factor));
          const newSpanY = Math.max(1e-5, Math.min((CFG.initYMax - CFG.initYMin) * 4, (yMax - yMin) * factor));
          xMin = midX - 0.5 * newSpanX;
          xMax = midX + 0.5 * newSpanX;
          yMax = yMin + newSpanY;
          redrawChart();
        }};

        window.distChartSetZoomPct = function(pctVal) {{
          const pct = Math.max(25, Math.min(1000, parseFloat(pctVal) || 100));
          const scale = 100 / pct;
          const midX = 0.5 * (xMin + xMax);
          const newSpanX = (CFG.initXMax - CFG.initXMin) * scale;
          const newSpanY = (CFG.initYMax - CFG.initYMin) * scale;
          xMin = midX - 0.5 * newSpanX;
          xMax = midX + 0.5 * newSpanX;
          yMax = yMin + newSpanY;
          redrawChart();
        }};

        window.distChartPanStep = function(fracX, fracY) {{
          const dx = (xMax - xMin) * fracX;
          const dy = (yMax - yMin) * fracY;
          xMin += dx; xMax += dx;
          const nextYMin = Math.max(0, yMin + dy);
          const spanY = yMax - yMin;
          yMin = nextYMin;
          yMax = nextYMin + spanY;
          redrawChart();
        }};

        window.distChartTogglePointLabels = function() {{
          showPointLabels = !showPointLabels;
          const btn = document.getElementById('distChartToggleLabelsBtn');
          if (btn) {{
            btn.innerText = showPointLabels ? '🏷️ Valores: ON' : '🏷️ Valores: OFF';
            btn.style.background = showPointLabels ? '#1e3a8a' : '#1e293b';
          }}
          redrawChart();
        }};

        window.distChartResetView = function() {{
          xMin = CFG.initXMin; xMax = CFG.initXMax;
          yMin = CFG.initYMin; yMax = CFG.initYMax;
          lastHoverMathX = null;
          crossLayer.innerHTML = '';
          redrawChart();
        }};
      }})();
      </script>
    </div>
    """


class StatVisionHTTPHandler(BaseHTTPRequestHandler):
    analyzer = StatisticalImageAnalyzer()

    def log_message(self, format: str, *args: Any) -> None:
        pass

    @staticmethod
    def _read_macos_system_clipboard_image_b64() -> str:
        """
        Lê diretamente uma imagem da Área de Transferência do sistema (NSPasteboard no macOS
        ou ImageGrab no Windows/Linux) e retorna em Base64 PNG.
        """
        # 1. Tenta Pillow ImageGrab se instalado (funciona em Windows, macOS e Linux)
        try:
            from PIL import ImageGrab  # type: ignore
            import io
            grabbed = ImageGrab.grabclipboard()
            if grabbed is not None:
                buf = io.BytesIO()
                grabbed.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass

        # 2. Nativo do macOS via AppKit NSPasteboard (sem dependências externas)
        if sys.platform == "darwin" and shutil.which("swift"):
            swift_clip_code = """
import AppKit
import Foundation
let pb = NSPasteboard.general
if let img = NSImage(pasteboard: pb),
   let tiff = img.tiffRepresentation,
   let rep = NSBitmapImageRep(data: tiff),
   let png = rep.representation(using: .png, properties: [:]) {
    print(png.base64EncodedString())
}
"""
            try:
                proc = subprocess.run(
                    ["swift", "-e", swift_clip_code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except Exception:
                pass

        # 3. Nativo do Windows via PowerShell System.Windows.Forms.Clipboard (sem dependências externas)
        if sys.platform == "win32" and shutil.which("powershell"):
            ps_clip = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "Add-Type -AssemblyName System.Drawing; "
                "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
                "if ($null -ne $img) { "
                "$ms = New-Object System.IO.MemoryStream; "
                "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); "
                "[Convert]::ToBase64String($ms.ToArray()) }"
            )
            try:
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-STA", "-Command", ps_clip],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except Exception:
                pass

        # 4. Nativo do Linux via wl-paste (Wayland) ou xclip (X11)
        if sys.platform.startswith("linux"):
            for cmd in (
                ["wl-paste", "--type", "image/png"],
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
            ):
                if shutil.which(cmd[0]):
                    try:
                        proc = subprocess.run(cmd, capture_output=True, timeout=6)
                        if proc.returncode == 0 and proc.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                            return base64.b64encode(proc.stdout).decode("ascii")
                    except Exception:
                        pass
        return ""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if parsed.path == "/manifest.json":
            manifest = {
                "name": "StatMath — Leitor de Distribuições Estatísticas",
                "short_name": "StatMath",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#070b14",
                "theme_color": "#e11d48",
            }
            payload = str(manifest).replace("'", '"').encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/real-accuracy":
            eval_data = get_cached_real_evaluation()
            payload = json.dumps(eval_data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        qs = urllib.parse.parse_qs(parsed.query)
        written_query = qs.get("hint", [""])[0].strip()

        if written_query:
            report = self.analyzer.analyze(None, extra_text_hint=written_query)
            self._send_photomath_html(report=report, hint=written_query, prev_data_url="")
        else:
            # Inicia o aplicativo limpo, sem nenhum exemplo pré-carregado
            self._send_photomath_html(report=None, hint="", prev_data_url="")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/real-accuracy/run":
            eval_data = evaluate_real_test_folder(analyzer=self.analyzer, force=True)
            payload = json.dumps(eval_data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8", errors="ignore")
        params = urllib.parse.parse_qs(body)

        hint = params.get("hint", [""])[0].strip()
        data_url = params.get("image_data_url", [""])[0].strip()
        orig_data_url = params.get("original_image_data_url", [""])[0].strip()
        pan_x = params.get("pan_x", ["0"])[0].strip()
        pan_y = params.get("pan_y", ["0"])[0].strip()
        zoom_scale = params.get("zoom_scale", ["1"])[0].strip()
        action = params.get("action", ["analyze"])[0]

        if action == "clear":
            self._send_photomath_html(report=None, hint="", prev_data_url="")
            return

        if action == "paste_system_clipboard":
            clip_b64 = self._read_macos_system_clipboard_image_b64()
            if clip_b64:
                data_url = f"data:image/png;base64,{clip_b64}"
                orig_data_url = data_url
                pan_x, pan_y, zoom_scale = "0", "0", "1"

        if not orig_data_url.startswith("data:image/"):
            orig_data_url = data_url

        report: Optional[DistributionReport] = None

        if data_url.startswith("data:image/") and "," in data_url:
            header_part, b64_part = data_url.split(",", 1)
            ext = ".jpg" if ("jpeg" in header_part or "jpg" in header_part) else (".webp" if "webp" in header_part else ".png")
            raw_bytes = base64.b64decode(b64_part)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(raw_bytes)
            tmp.close()
            try:
                report = self.analyzer.analyze(tmp.name, extra_text_hint=hint)
                # Quando o usuário moveu ou deu zoom na imagem dentro do retângulo de análise,
                # avalia também a imagem original completa para nunca perder parâmetros ou símbolos
                # que tenham ficado parcialmente fora do recorte:
                if orig_data_url.startswith("data:image/") and "," in orig_data_url and orig_data_url != data_url:
                    orig_header, orig_b64 = orig_data_url.split(",", 1)
                    orig_ext = (
                        ".jpg"
                        if ("jpeg" in orig_header or "jpg" in orig_header)
                        else (".webp" if "webp" in orig_header else ".png")
                    )
                    tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=orig_ext)
                    tmp_orig.write(base64.b64decode(orig_b64))
                    tmp_orig.close()
                    try:
                        rep_orig = self.analyzer.analyze(tmp_orig.name, extra_text_hint=hint)
                        crop_max_score = max(report.ocr_result.distribution_scores.values() or [0.0])
                        orig_max_score = max(rep_orig.ocr_result.distribution_scores.values() or [0.0])
                        crop_params = len(report.ocr_result.statistical_context.extracted_parameters)
                        orig_params = len(rep_orig.ocr_result.statistical_context.extracted_parameters)
                        if orig_max_score > crop_max_score or (orig_params > crop_params):
                            report = rep_orig
                    finally:
                        try:
                            os.remove(tmp_orig.name)
                        except OSError:
                            pass
            finally:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
        elif hint:
            # Quando o usuário escreve/digita a distribuição diretamente (sem enviar imagem)
            report = self.analyzer.analyze(None, extra_text_hint=hint)

        self._send_photomath_html(
            report=report,
            hint=hint,
            prev_data_url=orig_data_url,
            pan_x=pan_x,
            pan_y=pan_y,
            zoom_scale=zoom_scale,
        )

    def _build_results_column_html(self, report: Optional[DistributionReport]) -> str:
        if report is None:
            return """
            <div class="empty-results-card">
              <div style="font-size:3rem;margin-bottom:12px;">📐</div>
              <h2 style="margin:0 0 10px 0;color:#fff;">Pronto para Analisar sua Distribuição</h2>
              <p style="color:#94a3b8;max-width:560px;margin:0 auto 20px auto;line-height:1.6;">
                Utilize o painel ao lado para <strong>tirar uma foto pela câmera</strong>,
                <strong>enviar uma imagem do seu dispositivo</strong> ou <strong>escrever a distribuição diretamente no campo de texto</strong>.
              </p>
              <div class="didactic-box" style="text-align:left;max-width:580px;margin:0 auto;">
                <strong>✍️ Exemplos do que você pode escrever ou fotografar:</strong>
                <ul style="margin:8px 0 0 0;color:#cbd5e1;line-height:1.7;">
                  <li><code>X_1, ..., X_n ~ i.i.d. N(0, 1)</code> &nbsp;(Distribuição Normal com amostra i.i.d.)</li>
                  <li><code>X ~ Exponencial(lambda=2)</code> &nbsp;(Distribuição Exponencial)</li>
                  <li><code>X ~ Poisson(lambda=4)</code> &nbsp;(Distribuição de Poisson)</li>
                  <li><code>X ~ Binomial(n=10, p=0.4)</code> &nbsp;(Distribuição Binomial)</li>
                  <li><code>X ~ Uniforme(a=-2, b=5)</code> &nbsp;(Distribuição Uniforme Contínua)</li>
                  <li><code>X ~ Gama(alpha=3, beta=1.5)</code> ou <code>X ~ Beta(2, 5)</code> ou <code>T ~ Student(5)</code></li>
                </ul>
              </div>
            </div>
            """

        spec = report.identified_distribution
        num = report.numeric_summary
        ctx = report.ocr_result.statistical_context
        rv_sym = "x" if (ctx.is_iid or ctx.is_indexed_sample) else (ctx.primary_random_variable or "X").lower()
        if len(rv_sym) > 2 or "_" in rv_sym:
            rv_sym = "x"
        svg_overlay = _render_svg_curve(
            spec.id,
            num.get("parameters_used", {}),
            rv_symbol=rv_sym,
            mean_val=num.get("mean"),
        )

        tips_html = "".join(f"<li>{tip}</li>" for tip in ctx.handwriting_feedback)

        param_accordions = []
        for idx, p_spec in enumerate(spec.parameterizations):
            open_attr = "open" if idx == 0 else ""
            interp_items = "".join(
                f"<li><strong>{html.escape(p_sym)}:</strong> {html.escape(p_desc)}</li>"
                for p_sym, p_desc in p_spec.parameter_interpretations.items()
            )
            param_accordions.append(f"""
            <details class="nested-accordion" {open_attr}>
                <summary>
                    <span>📌 {html.escape(p_spec.name)}</span>
                    <span class="summary-math">\\({p_spec.canonical_writing_latex}\\)</span>
                </summary>
                <div class="accordion-body">
                    <p><strong>Como Escrever nesta Parametrização:</strong> \\({p_spec.canonical_writing_latex}\\) &nbsp;|&nbsp; <strong>Parâmetros:</strong> \\({p_spec.parameters}\\)</p>
                    <p><strong>{html.escape(spec.pdf_or_pmf_name)}:</strong></p>
                    <div class="math-block">\\[{p_spec.pdf_or_pmf_latex}\\]</div>
                    <p><strong>Função de Distribuição Acumulada (FDA / CDF — \\(F_X(x) = \\mathbb{{P}}(X \\le x)\\)):</strong></p>
                    <div class="math-block">\\[{p_spec.cdf_latex}\\]</div>
                    <p><strong>Média:</strong> \\({p_spec.mean_latex}\\) &nbsp;|&nbsp; <strong>Variância:</strong> \\({p_spec.variance_latex}\\) &nbsp;|&nbsp; <strong>Curtose:</strong> \\({p_spec.kurtosis_latex}\\)</p>
                    <div class="didactic-box">
                        <strong>🔍 Como interpretar cada parâmetro nesta parametrização:</strong>
                        <ul style="margin:6px 0 0 0;">{interp_items}</ul>
                    </div>
                    <p class="muted" style="margin-top:8px;"><em>Onde é usada / Conversão:</em> {html.escape(p_spec.conversion_notes)}</p>
                </div>
            </details>
            """)

        iid_items = "".join(
            f"<li><strong>{html.escape(k_title)}:</strong> {k_desc}</li>"
            for k_title, k_desc in spec.iid_and_statistical_properties.items()
        )

        img_props = report.curve_analysis.image_properties
        quality_warning_html = ""
        if getattr(img_props, "is_low_quality", False) and getattr(img_props, "quality_warnings", None):
            q_items = "".join(
                f"<li style='margin-bottom:4px;'>{html.escape(w_msg)}</li>"
                for w_msg in img_props.quality_warnings
            )
            quality_warning_html = f"""
            <div style="background:rgba(239, 68, 68, 0.13);border:1.5px solid #ef4444;border-radius:14px;padding:12px 16px;margin-bottom:16px;">
              <div style="font-size:0.82rem;text-transform:uppercase;letter-spacing:0.6px;color:#fca5a5;font-weight:800;margin-bottom:6px;">
                ⚠️ Aviso: A qualidade da imagem está prejudicando a análise
              </div>
              <ul style="margin:0;padding-left:18px;font-size:0.88rem;color:#fee2e2;line-height:1.5;">
                {q_items}
              </ul>
            </div>
            """

        corrections_html = ""
        if ctx.notation_corrections:
            corr_items = "".join(f"<li style='margin-bottom:6px;'>{c_item}</li>" for c_item in ctx.notation_corrections)
            corrections_html = f"""
            <div style="background:rgba(245, 158, 11, 0.12);border:1.5px solid #f59e0b;border-radius:14px;padding:12px 16px;margin-bottom:16px;">
              <div style="font-size:0.80rem;text-transform:uppercase;letter-spacing:0.7px;color:#fbbf24;font-weight:800;margin-bottom:6px;">
                🛠️ Correções Matemáticas de Notação/Intervalo Identificadas na Sua Imagem
              </div>
              <ul style="margin:0;padding-left:18px;font-size:0.89rem;color:#fef3c7;line-height:1.55;">
                {corr_items}
              </ul>
            </div>
            """

        return f"""
        <!-- CARD DE RESULTADO IMEDIATO (SEMPRE VISÍVEL NO COMEÇO) -->
        <div class="pm-result-hero">
          {quality_warning_html}
          <div class="pm-scanned-box">
            <div class="pm-scanned-label">✍️ Como Deveria Ser Escrito Formalmente (em LaTeX)</div>
            <div style="font-size:1.22rem;overflow-x:auto;">
              \\[{ctx.canonical_how_to_write_latex}\\]
            </div>
          </div>
          {corrections_html}

          <span class="pm-dist-badge">DISTRIBUIÇÃO IDENTIFICADA</span>
          <h2 style="margin:0 0 6px 0;font-size:1.65rem;color:#fff;">{html.escape(spec.name)}</h2>
          <p style="margin:0 0 14px 0;color:#cbd5e1;font-size:0.93rem;">
            <strong>Família:</strong> Distribuição {spec.family} &nbsp;•&nbsp;
            <strong>Suporte:</strong> \\({spec.support_latex}\\) &nbsp;•&nbsp;
            <strong>Variável Aleatória:</strong> \\({ctx.primary_random_variable}\\) &nbsp;•&nbsp;
            <strong>Amostragem:</strong> {"i.i.d. (Independente e Identicamente Distribuída)" if ctx.is_iid else ("Independente" if ctx.is_independent else "Variável Aleatória")}
          </p>

          <div style="font-weight:700;color:#fda4af;font-size:0.92rem;margin-bottom:4px;">
            📐 {html.escape(spec.pdf_or_pmf_name)} Principal (Forma Geral):
          </div>
          <div class="math-block">\\[{spec.pdf_or_pmf_latex}\\]</div>

          <div style="font-weight:700;color:#38bdf8;font-size:0.92rem;margin-top:10px;margin-bottom:4px;">
            📈 Função de Distribuição Acumulada — FDA / CDF Geral (\\(F_X(x) = \\mathbb{{P}}(X \\le x)\\)):
          </div>
          <div class="math-block">\\[{spec.cdf_latex}\\]</div>

          <!-- NOVA SEÇÃO: FDP E FDA COM OS PARÂMETROS CAPTURADOS DA IMAGEM -->
          <div style="background:rgba(16, 185, 129, 0.08);border:1.5px solid #10b981;border-radius:14px;padding:14px 16px;margin-top:14px;margin-bottom:14px;">
            <div style="font-size:0.80rem;text-transform:uppercase;letter-spacing:0.7px;color:#34d399;font-weight:800;margin-bottom:6px;">
              🎯 FDP / FMP e Acumulada (FDA) com os Parâmetros da Sua Imagem
            </div>
            <p style="margin:0 0 8px 0;font-size:0.90rem;color:#e2e8f0;">
              <strong>Parâmetros Identificados na Imagem:</strong> \\({ctx.image_parameters_latex}\\) &nbsp;|&nbsp;
              <strong>Variável Aleatória:</strong> \\({ctx.primary_random_variable}\\)
            </p>
            <div style="font-weight:700;color:#6ee7b7;font-size:0.88rem;margin-bottom:4px;">
              🔹 {html.escape(spec.pdf_or_pmf_name)} Aplicada aos Parâmetros da Imagem:
            </div>
            <div class="math-block" style="border-color:#065f46;">\\[{ctx.image_pdf_latex}\\]</div>

            <div style="font-weight:700;color:#7dd3fc;font-size:0.88rem;margin-top:10px;margin-bottom:4px;">
              🔹 Função de Distribuição Acumulada (FDA / CDF) com os Parâmetros da Imagem:
            </div>
            <div class="math-block" style="border-color:#0c4a6e;">\\[{ctx.image_cdf_latex}\\]</div>
            <div style="font-size:0.82rem;color:#94a3b8;margin-top:4px;">
              ℹ️ <em>{html.escape(ctx.image_cdf_note)}</em>
            </div>
          </div>

          <div class="didactic-box" style="margin-top:12px;margin-bottom:12px;">
            <strong>💡 Intuição Rápida:</strong> {html.escape(spec.didactic_overview)}
          </div>

          {svg_overlay}
        </div>

        <!-- CABEÇALHO DOS PASSOS DA SOLUÇÃO (SUB-REGIÕES RETRÁTEIS) -->
        <div class="steps-header">
          <div style="font-weight:800;font-size:1.08rem;color:#fff;">
            📖 Detalhes da Distribuição (Clique nas sub-regiões para Expandir ou Minimizar)
          </div>
          <div style="display:flex;gap:8px;">
            <button type="button" class="dock-btn" onclick="toggleAllSteps(true)">➕ Abrir Todas</button>
            <button type="button" class="dock-btn" onclick="toggleAllSteps(false)">➖ Minimizar Todas</button>
          </div>
        </div>

        <!-- PASSO 1: COMO DEVERIA SER ESCRITO FORMALMENTE -->
        <details class="section-accordion" open>
          <summary>
            <span style="display:flex;align-items:center;"><span class="step-number">1</span> Como Deveria Ser Escrito Formalmente & Dicas de Escrita</span>
          </summary>
          <div class="accordion-body">
            <p>Notação matemática formal em LaTeX para a expressão analisada:</p>
            <div class="math-block">\\[{ctx.canonical_how_to_write_latex}\\]</div>
            <table>
              <tr>
                <th>Para uma Única Variável Aleatória</th>
                <td>\\[{spec.canonical_single_rv_latex}\\]</td>
              </tr>
              <tr>
                <th>Para uma Amostra Aleatória i.i.d. de tamanho \\(n\\)</th>
                <td>\\[{spec.canonical_iid_rv_latex}\\]</td>
              </tr>
            </table>
            <div class="didactic-box">
              <strong>✍️ Dicas Educativas de Escrita:</strong>
              <ul style="margin:6px 0 0 0;">{tips_html}</ul>
            </div>
            <p class="muted"><strong>Texto Lido ({html.escape(report.ocr_result.backend_used)}):</strong> <code>{html.escape(report.ocr_result.raw_text)}</code></p>
          </div>
        </details>

        <!-- PASSO 2: VARIÁVEL ALEATÓRIA, INDEPENDÊNCIA E I.I.D. -->
        <details class="section-accordion">
          <summary>
            <span style="display:flex;align-items:center;"><span class="step-number">2</span> Variável Aleatória (\\({ctx.primary_random_variable}\\)), Independência (\\(X \\perp Y\\)) e i.i.d.</span>
          </summary>
          <div class="accordion-body">
            <table>
              <tr>
                <th style="width:30%;">Qual é a Variável Aleatória?</th>
                <td><strong>\\({ctx.primary_random_variable}\\)</strong> — {html.escape(spec.random_variable_meaning)}</td>
              </tr>
              <tr>
                <th>É Independente (\\(X_i \\perp\\!\\!\\!\\perp X_j\\))?</th>
                <td><span class="{'pill-yes' if ctx.is_independent else 'pill-no'}">{'SIM — Detectado!' if ctx.is_independent else 'Não escrito explicitamente'}</span> {html.escape('; '.join(ctx.independence_evidence))}</td>
              </tr>
              <tr>
                <th>É Independente e Identicamente Distribuída (i.i.d.)?</th>
                <td><span class="{'pill-yes' if ctx.is_iid else 'pill-no'}">{'SIM — Amostra i.i.d. detectada!' if ctx.is_iid else 'Variável única ou i.i.d. não explícito'}</span> {html.escape('; '.join(ctx.iid_evidence))}</td>
              </tr>
            </table>
            <div class="didactic-box">
              <strong>🧠 Definição Matemática em LaTeX:</strong>
              <ul style="margin:6px 0 0 0;">
                <li><strong>Independência Estatística (\\(X_1 \\perp\\!\\!\\!\\perp \\dots \\perp\\!\\!\\!\\perp X_n\\)):</strong> \\[f_{{X_1, \\dots, X_n}}(x_1, \\dots, x_n) = \\prod_{{i=1}}^n f_{{X_i}}(x_i)\\]</li>
                <li><strong>Identicamente Distribuídas (i.d.):</strong> \\[F_{{X_1}}(x) = F_{{X_2}}(x) = \\dots = F_{{X_n}}(x) = F(x; \\theta)\\]</li>
                <li><strong>Amostra i.i.d.:</strong> Permite escrever a Função de Verossimilhança como \\(L(\\theta) = \\prod_{{i=1}}^n f(x_i; \\theta)\\).</li>
              </ul>
            </div>
            <h4>Propriedades Teóricas sob Independência e i.i.d.</h4>
            <ul>{iid_items}</ul>
          </div>
        </details>

        <!-- PASSO 3: PARÂMETROS, MÉDIA, VARIÂNCIA E CURTOSE -->
        <details class="section-accordion">
          <summary>
            <span style="display:flex;align-items:center;"><span class="step-number">3</span> Parâmetros, Média, Variância e Curtose (em LaTeX)</span>
          </summary>
          <div class="accordion-body">
            <p><strong>Parâmetros da Distribuição:</strong> \\({spec.primary_parameters_latex}\\)</p>
            <p><strong>Parâmetros Utilizados no Cálculo:</strong> <code>{html.escape(str(num.get('parameters_used')))}</code></p>
            <table>
              <thead>
                <tr>
                  <th>Medida Estatística</th>
                  <th>Fórmula em LaTeX</th>
                  <th>Valor</th>
                  <th>Explicação Didática</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Média (\\(\\mathbb{{E}}[X]\\))</strong></td>
                  <td>\\[{spec.mean_latex}\\]</td>
                  <td><code>{num.get('mean')}</code></td>
                  <td>{html.escape(spec.mean_didactic)}</td>
                </tr>
                <tr>
                  <td><strong>Variância (\\(\\operatorname{{Var}}(X)\\))</strong></td>
                  <td>\\[{spec.variance_latex}\\]</td>
                  <td><code>{num.get('variance')}</code></td>
                  <td>{html.escape(spec.variance_didactic)}</td>
                </tr>
                <tr>
                  <td><strong>Mediana</strong></td>
                  <td>\\[{spec.median_latex}\\]</td>
                  <td><code>{num.get('median')}</code></td>
                  <td>Valor central onde \\(F_X(\\text{{Mediana}}) = 0.5\\).</td>
                </tr>
                <tr>
                  <td><strong>Moda</strong></td>
                  <td>\\[{spec.mode_latex}\\]</td>
                  <td><code>{num.get('mode')}</code></td>
                  <td>Valor mais provável (pico máximo da densidade/massa).</td>
                </tr>
                <tr>
                  <td><strong>Assimetria (\\(\\gamma_1\\))</strong></td>
                  <td>\\[{spec.skewness_latex}\\]</td>
                  <td><code>{num.get('skewness')}</code></td>
                  <td>Mede a simetria da distribuição (\\(\\gamma_1 = 0\\) indica simetria).</td>
                </tr>
                <tr>
                  <td><strong>Excesso de Curtose (\\(\\gamma_2\\))</strong></td>
                  <td>\\[{spec.excess_kurtosis_latex}\\]</td>
                  <td><code>{num.get('excess_kurtosis')}</code></td>
                  <td rowspan="2">{html.escape(spec.kurtosis_didactic)}</td>
                </tr>
                <tr>
                  <td><strong>Curtose Bruta (\\(\\beta_2\\))</strong></td>
                  <td>\\[{spec.raw_kurtosis_latex}\\]</td>
                  <td><code>{num.get('raw_kurtosis')}</code></td>
                </tr>
              </tbody>
            </table>
          </div>
        </details>

        <!-- PASSO 4: FÓRMULA DE MOMENTOS -->
        <details class="section-accordion">
          <summary>
            <span style="display:flex;align-items:center;"><span class="step-number">4</span> Fórmula de Momentos (Brutos, Centrais e Geradora de Momentos)</span>
          </summary>
          <div class="accordion-body">
            <p><strong>1. \\(k\\)-ésimo Momento Bruto (\\(\\mu'_k = \\mathbb{{E}}[X^k]\\)):</strong></p>
            <div class="math-block">\\[{spec.raw_moment_k_latex}\\]</div>

            <p><strong>2. \\(k\\)-ésimo Momento Central (\\(\\mu_k = \\mathbb{{E}}[(X - \\mu)^k]\\)):</strong></p>
            <div class="math-block">\\[{spec.central_moment_k_latex}\\]</div>

            <p><strong>3. Função Geradora de Momentos — FGM (\\(M_X(t) = \\mathbb{{E}}[e^{{tX}}]\\)):</strong></p>
            <div class="math-block">\\[{spec.mgf_latex}\\]</div>

            <p><strong>4. Função Característica (\\(\\varphi_X(t) = \\mathbb{{E}}[e^{{itX}}]\\)):</strong></p>
            <div class="math-block">\\[{spec.cf_latex}\\]</div>

            <div class="didactic-box">
              <strong>🎓 Como deduzir a fórmula de momentos passo a passo:</strong><br/>
              {html.escape(spec.moments_didactic_guide)}
            </div>
          </div>
        </details>

        <!-- PASSO 5: TODAS AS PARAMETRIZAÇÕES (FDP, FDA ACUMULADA E INTERPRETAÇÃO) -->
        <details class="section-accordion">
          <summary>
            <span style="display:flex;align-items:center;"><span class="step-number">5</span> Todas as Parametrizações: FDP, Acumulada (FDA) e Interpretação dos Parâmetros</span>
          </summary>
          <div class="accordion-body">
            <p class="muted">Clique em cada parametrização para abrir sua <strong>FDP/FMP</strong>, sua <strong>Função Acumulada (FDA)</strong> e <strong>como interpretar cada parâmetro</strong>:</p>
            {"".join(param_accordions)}
          </div>
        </details>
        """

    def _send_photomath_html(
        self,
        report: Optional[DistributionReport],
        hint: str,
        prev_data_url: str = "",
        pan_x: str = "0",
        pan_y: str = "0",
        zoom_scale: str = "1",
    ) -> None:
        has_user_image = prev_data_url.startswith("data:image/") and "," in prev_data_url
        img_src = prev_data_url if has_user_image else ""
        img_display = "block" if has_user_image else "none"
        controls_display = "block" if has_user_image else "none"
        placeholder_display = "none" if has_user_image else "flex"
        try:
            init_pan_x = float(pan_x)
        except ValueError:
            init_pan_x = 0.0
        try:
            init_pan_y = float(pan_y)
        except ValueError:
            init_pan_y = 0.0
        try:
            init_zoom = max(0.25, min(4.0, float(zoom_scale)))
        except ValueError:
            init_zoom = 1.0
        init_zoom_pct = int(round(init_zoom * 100))

        results_column_html = self._build_results_column_html(report)

        real_eval = get_cached_real_evaluation()
        real_eval_acc = real_eval.get("accuracy_pct")
        real_eval_pts = real_eval.get("total_points", real_eval.get("correct", 0))
        real_eval_total = real_eval.get("evaluated", 0)
        real_eval_folder_imgs = real_eval.get("image_count_in_folder", 0)
        real_eval_fresh = bool(real_eval.get("is_fresh", True))

        if real_eval_folder_imgs == 0:
            real_acc_pill_style = "background:rgba(51,65,85,0.6);border:1px solid #475569;color:#cbd5e1;"
            real_acc_pill_text = "🧪 Teste Real: 0 imagens em real_test_data/"
        elif real_eval_acc is not None and real_eval_fresh:
            if real_eval_acc >= 80.0:
                real_acc_pill_style = "background:rgba(6,95,70,0.45);border:1px solid #10b981;color:#6ee7b7;"
            elif real_eval_acc >= 50.0:
                real_acc_pill_style = "background:rgba(146,64,14,0.45);border:1px solid #f59e0b;color:#fde68a;"
            else:
                real_acc_pill_style = "background:rgba(159,18,57,0.45);border:1px solid #f43f5e;color:#fda4af;"
            real_acc_pill_text = f"🎯 Acurácia Real da IA: {real_eval_acc:.1f}% ({real_eval_pts:g}/{real_eval_total} pts)"
        else:
            real_acc_pill_style = "background:rgba(30,58,138,0.45);border:1px solid #38bdf8;color:#bae6fd;"
            real_acc_pill_text = f"🧪 Avaliando Teste Real ({real_eval_folder_imgs} imagens)..."

        real_eval_needs_poll_js = "true" if (real_eval_folder_imgs > 0 and not real_eval_fresh) else "false"
        real_eval_dir_escaped = html.escape(str(real_eval.get("test_dir") or ""))

        page = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
<link rel="manifest" href="/manifest.json"/>
<title>StatMath — Analisador de Distribuições Estatísticas</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [{{left: '\\\\[', right: '\\\\]', display: true}}, {{left: '$$', right: '$$', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}, {{left: '$', right: '$', display: false}}]}});"></script>
<style>
  :root {{
    --pm-red: #e11d48;
    --pm-red-dark: #be123c;
    --pm-blue: #38bdf8;
    --bg: #070b14;
    --card: #111827;
    --border: #26334d;
    --text: #f8fafc;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
  }}

  /* Barra Superior Limpa */
  .app-navbar {{
    background: rgba(15, 23, 42, 0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 12px 22px;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: -0.3px;
  }}
  .brand-icon {{
    background: var(--pm-red);
    color: #fff;
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(225, 29, 72, 0.4);
  }}

  .app-layout {{
    max-width: 1380px;
    margin: 0 auto;
    padding: 20px;
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 24px;
    align-items: start;
  }}
  @media (max-width: 980px) {{
    .app-layout {{ grid-template-columns: 1fr; padding: 12px; }}
  }}

  /* MÓDULO SCANNER LIMPO (Coluna Esquerda) */
  .scanner-shell {{
    background: #0d1322;
    border: 1px solid var(--border);
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 16px 40px rgba(0,0,0,0.45);
    position: relative;
  }}
  .scanner-topbar {{
    padding: 12px 16px;
    background: #111827;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e293b;
    font-size: 0.84rem;
    color: var(--muted);
  }}

  .viewfinder {{
    position: relative;
    width: 100%;
    height: 280px;
    background: #04070d;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    touch-action: none;
  }}
  .viewfinder video {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }}
  .viewfinder img#previewImage {{
    position: absolute;
    top: 50%;
    left: 50%;
    max-width: 86%;
    max-height: 62%;
    width: auto;
    height: auto;
    transform-origin: center center;
    cursor: grab;
    user-select: none;
    -webkit-user-drag: none;
    z-index: 1;
  }}
  .viewfinder img#previewImage:active {{
    cursor: grabbing;
  }}
  .zoom-pan-bar {{
    background: #0b111e;
    border-top: 1px solid #1e293b;
    padding: 10px 14px;
    font-size: 0.80rem;
    color: #cbd5e1;
  }}
  .zoom-controls-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }}
  .mini-ctrl-btn {{
    background: #1e293b;
    border: 1px solid #334155;
    color: #f8fafc;
    padding: 5px 9px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
  }}
  .mini-ctrl-btn:hover {{
    background: #334155;
  }}
  input[type="range"].zoom-slider {{
    flex: 1;
    accent-color: var(--pm-red);
    cursor: pointer;
  }}
  .viewfinder-placeholder {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 24px;
    color: #64748b;
    z-index: 2;
  }}

  /* Moldura Retangular de Foco */
  .crop-reticle {{
    position: absolute;
    width: 86%;
    height: 62%;
    border: 1.5px dashed rgba(255, 255, 255, 0.25);
    border-radius: 14px;
    pointer-events: none;
    box-shadow: 0 0 0 9999px rgba(4, 7, 13, 0.38);
    display: flex;
    align-items: flex-start;
    justify-content: center;
  }}
  .corner {{
    position: absolute;
    width: 22px;
    height: 22px;
    border-color: var(--pm-red);
    border-style: solid;
  }}
  .corner.tl {{ top: -2px; left: -2px; border-width: 4px 0 0 4px; border-top-left-radius: 12px; }}
  .corner.tr {{ top: -2px; right: -2px; border-width: 4px 4px 0 0; border-top-right-radius: 12px; }}
  .corner.bl {{ bottom: -2px; left: -2px; border-width: 0 0 4px 4px; border-bottom-left-radius: 12px; }}
  .corner.br {{ bottom: -2px; right: -2px; border-width: 0 4px 4px 0; border-bottom-right-radius: 12px; }}

  .scanner-Dock {{
    background: #111827;
    padding: 16px;
    border-top: 1px solid #1e293b;
  }}
  .shutter-row {{
    display: flex;
    justify-content: space-around;
    align-items: center;
    margin-bottom: 14px;
  }}
  .dock-btn {{
    background: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    padding: 9px 14px;
    border-radius: 12px;
    font-size: 0.83rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .dock-btn:hover {{ background: #334155; }}
  .shutter-btn {{
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: var(--pm-red);
    border: 4px solid #fff;
    box-shadow: 0 0 0 3px var(--pm-red), 0 8px 20px rgba(225,29,72,0.5);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    color: #fff;
    transition: transform 0.15s;
  }}
  .shutter-btn:hover {{ transform: scale(1.06); background: var(--pm-red-dark); }}

  /* Modal de Permissão de Câmera */
  #permissionModal {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(3, 7, 18, 0.82);
    backdrop-filter: blur(6px);
    z-index: 999;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }}
  .modal-card {{
    background: #111827;
    border: 2px solid var(--pm-red);
    border-radius: 20px;
    max-width: 420px;
    width: 100%;
    padding: 24px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0,0,0,0.7);
  }}

  /* Card Inicial Vazio e Cards de Solução */
  .empty-results-card {{
    background: var(--card);
    border: 1px dashed #334155;
    border-radius: 20px;
    padding: 48px 24px;
    text-align: center;
  }}
  .pm-result-hero {{
    background: linear-gradient(155deg, #111827 0%, #1e1b4b 100%);
    border: 2px solid var(--pm-red);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 12px 32px rgba(225, 29, 72, 0.16);
  }}
  .pm-scanned-box {{
    background: #090d16;
    border: 1px solid #26334d;
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 16px;
  }}
  .pm-scanned-label {{
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #fb7185;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .pm-dist-badge {{
    display: inline-block;
    background: var(--pm-red);
    color: #fff;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .math-block {{
    background: #090d16;
    padding: 12px 16px;
    border-radius: 12px;
    overflow-x: auto;
    margin: 8px 0;
    border: 1px solid #1e293b;
    font-size: 1.05rem;
  }}
  .didactic-box {{
    background: rgba(56, 189, 248, 0.09);
    border-left: 4px solid var(--pm-blue);
    padding: 12px 14px;
    border-radius: 8px;
    margin: 10px 0;
    line-height: 1.55;
  }}

  /* Passos da Solução (Sub-regiões Retráteis) */
  .steps-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 16px 0 10px 0;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .step-number {{
    background: var(--pm-red);
    color: #fff;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 800;
    margin-right: 10px;
    flex-shrink: 0;
  }}
  details.section-accordion {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 12px;
    overflow: hidden;
  }}
  details.section-accordion[open] {{ border-color: #fb7185; }}
  details.section-accordion > summary {{
    list-style: none;
    padding: 15px 18px;
    font-weight: 700;
    font-size: 1.0rem;
    color: #f8fafc;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #131c2e;
    user-select: none;
  }}
  details.section-accordion > summary::-webkit-details-marker {{ display: none; }}
  details.section-accordion > summary::after {{
    content: "Mostrar detalhes ▾";
    font-size: 0.78rem;
    color: #fb7185;
    background: rgba(225, 29, 72, 0.14);
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 600;
    flex-shrink: 0;
  }}
  details.section-accordion[open] > summary::after {{
    content: "Minimizar ▴";
    color: #94a3b8;
    background: rgba(255,255,255,0.07);
  }}
  .accordion-body {{ padding: 16px 18px; border-top: 1px solid var(--border); line-height: 1.6; }}

  /* Sub-acordeões de Parametrizações */
  details.nested-accordion {{ background: #090d16; border: 1px solid #26334d; border-radius: 12px; margin-bottom: 10px; }}
  details.nested-accordion > summary {{ list-style: none; padding: 12px 15px; font-weight: 600; font-size: 0.93rem; color: var(--pm-blue); cursor: pointer; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; }}
  details.nested-accordion > summary::-webkit-details-marker {{ display: none; }}
  .summary-math {{ font-size: 0.86rem; color: #e2e8f0; background: #111827; padding: 3px 9px; border-radius: 6px; border: 1px solid #1e293b; }}

  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.89rem; }}
  th, td {{ border-bottom: 1px solid #1e293b; padding: 9px 8px; text-align: left; vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; }}
  .pill-yes {{ background: #065f46; color: #6ee7b7; padding: 3px 9px; border-radius: 6px; font-size: 0.76rem; font-weight: 700; }}
  .pill-no {{ background: #334155; color: #cbd5e1; padding: 3px 9px; border-radius: 6px; font-size: 0.76rem; }}
  input[type="text"] {{ width: 100%; padding: 11px 12px; background: #090d16; border: 1px solid #334155; color: #f8fafc; border-radius: 10px; font-size: 0.92rem; }}
  input[type="text"]:focus {{ outline: none; border-color: var(--pm-red); }}
  .muted {{ color: var(--muted); font-size: 0.86rem; }}
</style>
</head>
<body>

<!-- MODAL DE AUTORIZAÇÃO DE CÂMERA (Aparece antes de ligar a câmera) -->
<div id="permissionModal">
  <div class="modal-card">
    <div style="font-size:2.4rem;margin-bottom:8px;">📷</div>
    <h3 style="margin:0 0 8px 0;color:#fff;">Permitir Acesso à Câmera?</h3>
    <p style="color:#cbd5e1;font-size:0.92rem;line-height:1.5;margin-bottom:18px;">
      Para fotografar sua distribuição estatística escrita à mão, autorize o acesso à câmera do seu dispositivo.
    </p>
    <div style="display:flex;gap:10px;justify-content:center;">
      <button type="button" class="dock-btn" style="background:var(--pm-red);border-color:var(--pm-red);color:#fff;padding:11px 18px;" onclick="confirmAndOpenCamera()">
        ✅ Autorizar e Abrir Câmera
      </button>
      <button type="button" class="dock-btn" onclick="closePermissionModal()">
        Cancelar
      </button>
    </div>
  </div>
</div>

<!-- MODAL DE ACURÁCIA REAL (TESTE CEGO EM real_test_data/) -->
<div id="realAccuracyModal" style="display:none;position:fixed;inset:0;background:rgba(3,7,18,0.84);backdrop-filter:blur(6px);z-index:1000;align-items:center;justify-content:center;padding:20px;">
  <div style="background:#111827;border:1.5px solid #334155;border-radius:20px;max-width:780px;width:100%;max-height:86vh;display:flex;flex-direction:column;box-shadow:0 24px 60px rgba(0,0,0,0.75);overflow:hidden;">
    <div style="padding:18px 22px;border-bottom:1px solid #1e293b;display:flex;justify-content:space-between;align-items:center;gap:12px;background:#0f172a;">
      <div>
        <div style="font-weight:800;font-size:1.08rem;color:#fff;">🎯 Acurácia Real da IA — Teste Cego (real_test_data/)</div>
        <div style="font-size:0.78rem;color:#94a3b8;margin-top:2px;">
          A IA recebe apenas a imagem (sem ver o nome do arquivo). O nome serve exclusivamente como gabarito.
        </div>
      </div>
      <button type="button" class="dock-btn" onclick="closeRealAccuracyModal()" style="padding:6px 12px;">✕ Fechar</button>
    </div>
    <div id="realAccuracyModalBody" style="padding:18px 22px;overflow-y:auto;flex:1;font-size:0.88rem;color:#e2e8f0;">
      Carregando relatório do teste real...
    </div>
    <div style="padding:14px 22px;border-top:1px solid #1e293b;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;background:#0f172a;">
      <div style="font-size:0.75rem;color:#94a3b8;word-break:break-all;">
        📁 Pasta: <code>{real_eval_dir_escaped}</code>
      </div>
      <button type="button" id="runRealEvalBtn" class="dock-btn" style="background:var(--pm-red);border-color:var(--pm-red);color:#fff;padding:9px 16px;" onclick="runRealAccuracyEval()">
        🔄 Reavaliar Pasta Agora
      </button>
    </div>
  </div>
</div>

<!-- NAVBAR SUPERIOR LIMPA -->
<nav class="app-navbar">
  <div class="brand">
    <div class="brand-icon">∑</div>
    <div>
      <span>StatMath</span>
      <span style="font-size:0.78rem;color:#fb7185;display:block;font-weight:600;">Leitor e Tutor de Distribuições Estatísticas</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <button type="button" id="realAccuracyPill" onclick="openRealAccuracyModal()"
            title="Clique para ver o relatório detalhado do Teste Real (real_test_data/)"
            style="{real_acc_pill_style}padding:7px 14px;border-radius:999px;font-size:0.82rem;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:6px;transition:filter 0.15s;">
      {real_acc_pill_text}
    </button>
  </div>
</nav>

<div class="app-layout">
  <!-- COLUNA ESQUERDA: CÂMERA, ARQUIVO OU DIGITAÇÃO DIRETA -->
  <div>
    <div class="scanner-shell">
      <div class="scanner-topbar">
        <span>📷 Fotografe ou envie a imagem da distribuição</span>
        <span id="scannerStatusBadge" style="color:#fb7185;font-weight:700;">Pronto</span>
      </div>

      <!-- VISOR LIMPO + SUPORTE A ARRASTAR E SOLTAR E ÁREA DE TRANSFERÊNCIA -->
      <div class="viewfinder" id="dropZone">
        <div id="emptyPlaceholder" class="viewfinder-placeholder" style="display:{placeholder_display};">
          <div style="font-size:2.2rem;margin-bottom:8px;opacity:0.7;">📷</div>
          <div style="font-size:0.86rem;color:#94a3b8;max-width:270px;line-height:1.45;">
            Clique em <strong>📁 Arquivo</strong>, <strong>📋 Colar Print</strong> (ou pressione <kbd style="background:#1e293b;padding:2px 6px;border-radius:4px;border:1px solid #334155;">Ctrl+V</kbd> / <kbd style="background:#1e293b;padding:2px 6px;border-radius:4px;border:1px solid #334155;">⌘V</kbd>), arraste uma foto ou use a <strong>Câmera</strong>
          </div>
        </div>
        <img id="previewImage" src="{img_src}" alt="Imagem Enviada" draggable="false"
             style="display:{img_display};transform:translate(calc(-50% + {init_pan_x:.1f}px), calc(-50% + {init_pan_y:.1f}px)) scale({init_zoom:.3f});"/>
        <video id="cameraVideo" autoplay playsinline style="display:none;"></video>

        <div class="crop-reticle" id="reticleBox" style="z-index:3;">
          <div class="corner tl"></div>
          <div class="corner tr"></div>
          <div class="corner bl"></div>
          <div class="corner br"></div>
        </div>
      </div>

      <!-- BARRA INTERATIVA DE ZOOM E MOVIMENTO DA IMAGEM DENTRO DO RETÂNGULO -->
      <div class="zoom-pan-bar" id="zoomPanControls" style="display:{controls_display};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="color:#94a3b8;font-size:0.76rem;">
            🖐️ <strong>Arraste a imagem</strong> para encaixar no retângulo e ajuste o <strong>Zoom</strong>:
          </span>
          <span id="zoomValueLabel" style="color:#fb7185;font-weight:800;font-size:0.80rem;">{init_zoom_pct}%</span>
        </div>
        <div class="zoom-controls-row">
          <button type="button" class="mini-ctrl-btn" onclick="changeZoomBy(-0.15)" title="Diminuir imagem (Zoom -)">➖</button>
          <input type="range" id="zoomSlider" class="zoom-slider" min="0.25" max="4.0" step="0.05" value="{init_zoom:.2f}" oninput="onZoomSliderInput(this.value)"/>
          <button type="button" class="mini-ctrl-btn" onclick="changeZoomBy(0.15)" title="Aumentar imagem (Zoom +)">➕</button>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
          <div style="display:flex;gap:4px;">
            <button type="button" class="mini-ctrl-btn" onclick="nudgePan(-18, 0)" title="Mover para a esquerda">⬅️</button>
            <button type="button" class="mini-ctrl-btn" onclick="nudgePan(0, -18)" title="Mover para cima">⬆️</button>
            <button type="button" class="mini-ctrl-btn" onclick="nudgePan(0, 18)" title="Mover para baixo">⬇️</button>
            <button type="button" class="mini-ctrl-btn" onclick="nudgePan(18, 0)" title="Mover para a direita">➡️</button>
            <button type="button" class="mini-ctrl-btn" onclick="resetPanZoom()" title="Centralizar e restaurar tamanho padrão">🎯 Centralizar</button>
          </div>
          <button type="button" class="mini-ctrl-btn" style="background:var(--pm-red);border-color:var(--pm-red);color:#fff;padding:6px 12px;" onclick="applyPanZoomAndAnalyze()">
            🔍 Analisar Enquadramento
          </button>
        </div>
      </div>

      <canvas id="cameraCanvas" style="display:none;"></canvas>

      <!-- CONTROLES DO SCANNER E CAMPO DE ESCRITA DIRETA -->
      <div class="scanner-Dock">
        <div class="shutter-row" style="gap:8px;flex-wrap:wrap;">
          <label for="fileInput" class="dock-btn" style="margin:0;cursor:pointer;" title="Selecionar arquivo de imagem do computador">
            📁 Arquivo
          </label>

          <button type="button" class="dock-btn" onclick="pasteFromClipboard()" title="Colar imagem ou captura de tela da Área de Transferência (Ctrl+V / Cmd+V)">
            📋 Colar Print
          </button>

          <button type="button" class="shutter-btn" id="shutterButton" onclick="handleShutterClick()" title="Tirar Foto com a Câmera">
            📷
          </button>

          <button type="button" class="dock-btn" id="stopCamBtn" style="display:none;" onclick="stopCameraStream()">
            ✖️ Fechar
          </button>
          <button type="button" class="dock-btn" onclick="clearAllInput()" title="Limpar tudo">
            🗑️ Limpar
          </button>
        </div>

        <form method="POST" id="analyzeForm" onsubmit="prepareFormBeforeSubmit()">
          <input type="hidden" name="action" id="formAction" value="analyze"/>
          <input type="hidden" name="image_data_url" id="imageDataUrl" value="{html.escape(prev_data_url)}"/>
          <input type="hidden" name="original_image_data_url" id="originalImageDataUrl" value="{html.escape(prev_data_url)}"/>
          <input type="hidden" name="pan_x" id="panXInput" value="{init_pan_x:.1f}"/>
          <input type="hidden" name="pan_y" id="panYInput" value="{init_pan_y:.1f}"/>
          <input type="hidden" name="zoom_scale" id="zoomScaleInput" value="{init_zoom:.3f}"/>
          <input type="file" id="fileInput" accept="image/png,image/jpeg,image/webp,image/*" style="position:absolute;width:1px;height:1px;opacity:0;overflow:hidden;z-index:-1;"/>

          <div style="margin-top:8px;padding-top:12px;border-top:1px solid #1e293b;">
            <label style="display:block;font-size:0.82rem;color:#cbd5e1;font-weight:600;margin-bottom:6px;">
              ✍️ Ou escreva a distribuição diretamente abaixo:
            </label>
            <div style="display:flex;gap:8px;">
              <input type="text" name="hint" id="hintInput" value="{html.escape(hint)}" placeholder="Ex.: X_1, ..., X_n ~ i.i.d. N(0, 1) ou Pois(4)"/>
              <button type="submit" class="dock-btn" style="background:var(--pm-red);border-color:var(--pm-red);color:#fff;white-space:nowrap;padding:0 18px;">
                Analisar
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  </div>

  <!-- COLUNA DIREITA: RESULTADO EM LATEX E SUB-REGIÕES RETRÁTEIS -->
  <div>
    {results_column_html}
  </div>
</div>

<script>
let activeStream = null;
let cameraOpen = false;
let panX = {init_pan_x:.1f};
let panY = {init_pan_y:.1f};
let zoomScale = {init_zoom:.3f};
let isDraggingImg = false;
let dragStartX = 0;
let dragStartY = 0;
let dragInitPanX = 0;
let dragInitPanY = 0;

function updateImageTransformUI() {{
  const img = document.getElementById('previewImage');
  if (img) {{
    img.style.transform = `translate(calc(-50% + ${{panX.toFixed(1)}}px), calc(-50% + ${{panY.toFixed(1)}}px)) scale(${{zoomScale.toFixed(3)}})`;
  }}
  const slider = document.getElementById('zoomSlider');
  if (slider) slider.value = zoomScale;
  const lbl = document.getElementById('zoomValueLabel');
  if (lbl) lbl.innerText = Math.round(zoomScale * 100) + '%';
  const pxIn = document.getElementById('panXInput');
  const pyIn = document.getElementById('panYInput');
  const zsIn = document.getElementById('zoomScaleInput');
  if (pxIn) pxIn.value = panX.toFixed(1);
  if (pyIn) pyIn.value = panY.toFixed(1);
  if (zsIn) zsIn.value = zoomScale.toFixed(3);
}}

function onZoomSliderInput(val) {{
  zoomScale = Math.max(0.25, Math.min(4.0, parseFloat(val) || 1.0));
  updateImageTransformUI();
}}

function changeZoomBy(delta) {{
  zoomScale = Math.max(0.25, Math.min(4.0, zoomScale + delta));
  updateImageTransformUI();
}}

function nudgePan(dx, dy) {{
  panX += dx;
  panY += dy;
  updateImageTransformUI();
}}

function resetPanZoom() {{
  panX = 0;
  panY = 0;
  zoomScale = 1.0;
  updateImageTransformUI();
}}

function cropFramedRegionToDataUrl() {{
  const img = document.getElementById('previewImage');
  const reticle = document.getElementById('reticleBox');
  const origInput = document.getElementById('originalImageDataUrl');
  if (!img || img.style.display === 'none' || !img.naturalWidth || !img.naturalHeight) {{
    return origInput ? origInput.value : '';
  }}
  if (Math.abs(panX) < 1.0 && Math.abs(panY) < 1.0 && Math.abs(zoomScale - 1.0) < 0.02) {{
    return origInput ? origInput.value : '';
  }}
  const retRect = reticle.getBoundingClientRect();
  const imgRect = img.getBoundingClientRect();
  if (imgRect.width < 4 || imgRect.height < 4) {{
    return origInput ? origInput.value : '';
  }}

  const ix0 = Math.max(retRect.left, imgRect.left);
  const iy0 = Math.max(retRect.top, imgRect.top);
  const ix1 = Math.min(retRect.right, imgRect.right);
  const iy1 = Math.min(retRect.bottom, imgRect.bottom);

  if (ix1 - ix0 < 12 || iy1 - iy0 < 12) {{
    return origInput ? origInput.value : '';
  }}

  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  const sx = Math.max(0, Math.floor(((ix0 - imgRect.left) / imgRect.width) * natW));
  const sy = Math.max(0, Math.floor(((iy0 - imgRect.top) / imgRect.height) * natH));
  const sw = Math.min(natW - sx, Math.ceil(((ix1 - ix0) / imgRect.width) * natW));
  const sh = Math.min(natH - sy, Math.ceil(((iy1 - iy0) / imgRect.height) * natH));

  if (sw < 10 || sh < 10) {{
    return origInput ? origInput.value : '';
  }}

  const canvas = document.getElementById('cameraCanvas');
  canvas.width = sw;
  canvas.height = sh;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, sw, sh);
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  return canvas.toDataURL('image/png');
}}

function prepareFormBeforeSubmit() {{
  const action = document.getElementById('formAction').value;
  if (action === 'analyze') {{
    const origUrl = document.getElementById('originalImageDataUrl').value;
    if (origUrl && origUrl.startsWith('data:image/')) {{
      const croppedUrl = cropFramedRegionToDataUrl();
      if (croppedUrl && croppedUrl.startsWith('data:image/')) {{
        document.getElementById('imageDataUrl').value = croppedUrl;
      }}
    }}
  }}
}}

function applyPanZoomAndAnalyze() {{
  document.getElementById('formAction').value = 'analyze';
  prepareFormBeforeSubmit();
  document.getElementById('analyzeForm').submit();
}}

function handleShutterClick() {{
  if (!cameraOpen) {{
    document.getElementById('permissionModal').style.display = 'flex';
  }} else {{
    captureAndCropFromScanner();
  }}
}}

function closePermissionModal() {{
  document.getElementById('permissionModal').style.display = 'none';
}}

async function confirmAndOpenCamera() {{
  closePermissionModal();
  try {{
    activeStream = await navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode: 'environment', width: {{ ideal: 1280 }}, height: {{ ideal: 720 }} }},
      audio: false
    }});
    const video = document.getElementById('cameraVideo');
    video.srcObject = activeStream;
    video.style.display = 'block';
    document.getElementById('previewImage').style.display = 'none';
    document.getElementById('emptyPlaceholder').style.display = 'none';
    cameraOpen = true;
    document.getElementById('stopCamBtn').style.display = 'inline-flex';
    document.getElementById('scannerStatusBadge').innerText = '🔴 Câmera Ativa';
  }} catch (err) {{
    alert('Não foi possível acessar a câmera. Verifique as permissões do navegador/sistema: ' + err.message);
  }}
}}

function stopCameraStream() {{
  if (activeStream) {{
    activeStream.getTracks().forEach(t => t.stop());
    activeStream = null;
  }}
  cameraOpen = false;
  document.getElementById('cameraVideo').style.display = 'none';
  const hasImg = document.getElementById('imageDataUrl').value.startsWith('data:image/');
  document.getElementById('previewImage').style.display = hasImg ? 'block' : 'none';
  document.getElementById('emptyPlaceholder').style.display = hasImg ? 'none' : 'flex';
  document.getElementById('stopCamBtn').style.display = 'none';
  document.getElementById('scannerStatusBadge').innerText = 'Pronto';
}}

function captureAndCropFromScanner() {{
  const video = document.getElementById('cameraVideo');
  const canvas = document.getElementById('cameraCanvas');
  const vw = video.videoWidth || 920;
  const vh = video.videoHeight || 520;

  const cropW = Math.floor(vw * 0.86);
  const cropH = Math.floor(vh * 0.62);
  const cropX = Math.floor((vw - cropW) / 2);
  const cropY = Math.floor((vh - cropH) / 2);

  canvas.width = cropW;
  canvas.height = cropH;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

  const dataUrl = canvas.toDataURL('image/png');
  stopCameraStream();
  resetPanZoom();
  document.getElementById('originalImageDataUrl').value = dataUrl;
  document.getElementById('imageDataUrl').value = dataUrl;
  document.getElementById('formAction').value = 'analyze';
  document.getElementById('analyzeForm').submit();
}}

function handleSelectedImageFile(file) {{
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {{
    resetPanZoom();
    document.getElementById('originalImageDataUrl').value = evt.target.result;
    document.getElementById('imageDataUrl').value = evt.target.result;
    document.getElementById('formAction').value = 'analyze';
    document.getElementById('analyzeForm').submit();
  }};
  reader.readAsDataURL(file);
}}

document.getElementById('fileInput').addEventListener('change', function(e) {{
  if (e.target.files && e.target.files.length > 0) {{
    handleSelectedImageFile(e.target.files[0]);
  }}
}});

const dropZone = document.getElementById('dropZone');
if (dropZone) {{
  // Arrastar a imagem com o mouse dentro do visor (Pan)
  dropZone.addEventListener('mousedown', function(e) {{
    const img = document.getElementById('previewImage');
    if (!img || img.style.display === 'none' || cameraOpen) return;
    isDraggingImg = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragInitPanX = panX;
    dragInitPanY = panY;
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!isDraggingImg) return;
    panX = dragInitPanX + (e.clientX - dragStartX);
    panY = dragInitPanY + (e.clientY - dragStartY);
    updateImageTransformUI();
  }});
  window.addEventListener('mouseup', function() {{
    isDraggingImg = false;
  }});

  // Arrastar a imagem com o dedo em telas touch (Pan)
  dropZone.addEventListener('touchstart', function(e) {{
    const img = document.getElementById('previewImage');
    if (!img || img.style.display === 'none' || cameraOpen) return;
    if (e.touches.length === 1) {{
      isDraggingImg = true;
      dragStartX = e.touches[0].clientX;
      dragStartY = e.touches[0].clientY;
      dragInitPanX = panX;
      dragInitPanY = panY;
    }}
  }}, {{ passive: true }});
  dropZone.addEventListener('touchmove', function(e) {{
    if (!isDraggingImg || e.touches.length !== 1) return;
    panX = dragInitPanX + (e.touches[0].clientX - dragStartX);
    panY = dragInitPanY + (e.touches[0].clientY - dragStartY);
    updateImageTransformUI();
  }}, {{ passive: true }});
  dropZone.addEventListener('touchend', function() {{
    isDraggingImg = false;
  }});

  // Roda do mouse / gesto de pinça no trackpad para aumentar ou diminuir a imagem (Zoom)
  dropZone.addEventListener('wheel', function(e) {{
    const img = document.getElementById('previewImage');
    if (!img || img.style.display === 'none' || cameraOpen) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 0.10 : -0.10;
    changeZoomBy(factor);
  }}, {{ passive: false }});

  dropZone.addEventListener('dragover', function(e) {{
    e.preventDefault();
    e.stopPropagation();
    dropZone.style.borderColor = '#e11d48';
  }});
  dropZone.addEventListener('dragleave', function(e) {{
    e.preventDefault();
    e.stopPropagation();
    dropZone.style.borderColor = '';
  }});
  dropZone.addEventListener('drop', function(e) {{
    e.preventDefault();
    e.stopPropagation();
    dropZone.style.borderColor = '';
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {{
      handleSelectedImageFile(e.dataTransfer.files[0]);
    }}
  }});
}}

// 1. Suporte global a Ctrl+V / Cmd+V (colar imagem direto da Área de Transferência)
window.addEventListener('paste', function(e) {{
  const items = (e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData) || {{}}).items;
  if (!items) return;
  for (let i = 0; i < items.length; i++) {{
    if (items[i].type && items[i].type.indexOf('image') !== -1) {{
      e.preventDefault();
      const blob = items[i].getAsFile();
      if (blob) {{
        document.getElementById('scannerStatusBadge').innerText = '📋 Colando imagem...';
        handleSelectedImageFile(blob);
        return;
      }}
    }}
  }}
}});

// 2. Botão "📋 Colar Print" (tenta API do navegador; se bloqueado pelo WebKit, usa leitor nativo da Área de Transferência do SO)
async function pasteFromClipboard() {{
  document.getElementById('scannerStatusBadge').innerText = '📋 Lendo Área de Transferência...';
  if (navigator.clipboard && navigator.clipboard.read) {{
    try {{
      const clipboardItems = await navigator.clipboard.read();
      for (const item of clipboardItems) {{
        const imgType = item.types.find(type => type.startsWith('image/'));
        if (imgType) {{
          const blob = await item.getType(imgType);
          handleSelectedImageFile(blob);
          return;
        }}
      }}
    }} catch (err) {{
      // Continua para o fallback nativo do sistema operacional no servidor Python
    }}
  }}
  // Fallback garantido: pede ao backend Python para ler a Área de Transferência do Sistema (NSPasteboard / ImageGrab)
  document.getElementById('formAction').value = 'paste_system_clipboard';
  document.getElementById('analyzeForm').submit();
}}

function clearAllInput() {{
  stopCameraStream();
  resetPanZoom();
  document.getElementById('imageDataUrl').value = '';
  document.getElementById('originalImageDataUrl').value = '';
  document.getElementById('hintInput').value = '';
  document.getElementById('formAction').value = 'clear';
  document.getElementById('analyzeForm').submit();
}}

function toggleAllSteps(expand) {{
  document.querySelectorAll('details.section-accordion').forEach(el => {{
    el.open = expand;
  }});
}}

function escapeHtmlBasic(str) {{
  return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function updateRealAccuracyPillUI(data) {{
  const pill = document.getElementById('realAccuracyPill');
  if (!pill || !data) return;
  const folderImgs = data.image_count_in_folder || 0;
  const acc = data.accuracy_pct;
  const pts = (data.total_points !== undefined && data.total_points !== null) ? data.total_points : (data.correct || 0);
  const total = data.evaluated || 0;

  if (folderImgs === 0) {{
    pill.style.background = 'rgba(51,65,85,0.6)';
    pill.style.borderColor = '#475569';
    pill.style.color = '#cbd5e1';
    pill.textContent = '🧪 Teste Real: 0 imagens em real_test_data/';
  }} else if (acc !== null && acc !== undefined && data.is_fresh) {{
    if (acc >= 80) {{
      pill.style.background = 'rgba(6,95,70,0.45)';
      pill.style.borderColor = '#10b981';
      pill.style.color = '#6ee7b7';
    }} else if (acc >= 50) {{
      pill.style.background = 'rgba(146,64,14,0.45)';
      pill.style.borderColor = '#f59e0b';
      pill.style.color = '#fde68a';
    }} else {{
      pill.style.background = 'rgba(159,18,57,0.45)';
      pill.style.borderColor = '#f43f5e';
      pill.style.color = '#fda4af';
    }}
    pill.textContent = '🎯 Acurácia Real da IA: ' + Number(acc).toFixed(1) + '% (' + pts + '/' + total + ' pts)';
  }} else {{
    pill.style.background = 'rgba(30,58,138,0.45)';
    pill.style.borderColor = '#38bdf8';
    pill.style.color = '#bae6fd';
    pill.textContent = '🧪 Avaliando Teste Real (' + folderImgs + ' imagens)...';
  }}
}}

function renderRealAccuracyModalBody(data) {{
  const body = document.getElementById('realAccuracyModalBody');
  if (!body || !data) return;
  const items = data.items || [];
  const folderImgs = data.image_count_in_folder || 0;
  const acc = data.accuracy_pct;
  const pts = (data.total_points !== undefined && data.total_points !== null) ? data.total_points : (data.correct || 0);
  const total = data.evaluated || 0;
  const fullHits = data.correct || 0;
  const partialHits = data.partial || 0;
  const misses = data.wrong || 0;

  if (folderImgs === 0) {{
    body.innerHTML =
      '<div style="background:#090d16;border:1px dashed #334155;border-radius:14px;padding:22px;text-align:center;">' +
      '<div style="font-size:2rem;margin-bottom:8px;">📂</div>' +
      '<div style="font-weight:700;color:#fff;margin-bottom:6px;">Nenhuma imagem encontrada em <code>real_test_data/</code></div>' +
      '<p style="color:#94a3b8;margin:0 0 12px 0;line-height:1.5;">Coloque suas imagens de teste na pasta <code>real_test_data/</code> com o nome da distribuição no arquivo (ex.: <code>normal_1.jpg</code>, <code>poisson_2.png</code>, <code>cauchy.jpg</code>) e clique em <strong>Reavaliar Pasta Agora</strong>.</p>' +
      '<p style="color:#6ee7b7;font-size:0.82rem;margin:0;">🔒 Pontuação: 1.0 pt (Dist + Parâmetros) • 0.5 pt (Apenas um) • 0.0 pt (Errou ambos).</p>' +
      '</div>';
    return;
  }}

  let summaryHtml = '';
  if (acc !== null && acc !== undefined) {{
    const accColor = acc >= 80 ? '#34d399' : (acc >= 50 ? '#fbbf24' : '#fb7185');
    summaryHtml =
      '<div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;background:#090d16;border:1px solid #1e293b;border-radius:14px;padding:14px 18px;margin-bottom:14px;">' +
      '<div><div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Acurácia Real da IA — sᵢ = (dᵢ + pᵢ) / 2</div>' +
      '<div style="font-size:1.75rem;font-weight:900;color:' + accColor + ';">' + Number(acc).toFixed(1) + '% <span style="font-size:0.95rem;color:#cbd5e1;font-weight:600;">(' + pts + ' / ' + total + ' pts)</span></div>' +
      '<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">✅ 1.0 pt: <strong>' + fullHits + '</strong> &nbsp;•&nbsp; ⚠️ 0.5 pt: <strong>' + partialHits + '</strong> &nbsp;•&nbsp; ❌ 0.0 pt: <strong>' + misses + '</strong></div></div>' +
      '<div style="font-size:0.78rem;color:#94a3b8;text-align:right;">Última avaliação:<br/><strong style="color:#e2e8f0;">' + escapeHtmlBasic(data.evaluated_at || 'Agora') + '</strong></div>' +
      '</div>';
  }} else {{
    summaryHtml =
      '<div style="background:#090d16;border:1px solid #334155;border-radius:12px;padding:12px 14px;margin-bottom:14px;color:#bae6fd;">⏳ Clique em <strong>Reavaliar Pasta Agora</strong> para testar as ' + folderImgs + ' imagens encontradas.</div>';
  }}

  let rowsHtml = '';
  for (const it of items) {{
    const sc = it.sample_score;
    let badge = '<span class="pill-no">⚠️ SEM GABARITO</span>';
    if (sc === 1.0) {{
      badge = '<span class="pill-yes">✅ 1.0 pt</span>';
    }} else if (sc === 0.5) {{
      badge = '<span style="background:#78350f;color:#fde68a;padding:3px 9px;border-radius:6px;font-size:0.76rem;font-weight:700;">⚠️ 0.5 pt</span>';
    }} else if (sc === 0.0) {{
      badge = '<span style="background:#881337;color:#fda4af;padding:3px 9px;border-radius:6px;font-size:0.76rem;font-weight:700;">❌ 0.0 pt</span>';
    }}
    const dMark = it.dist_ok ? '✅' : '❌';
    const pMark = it.params_ok ? '✅' : '❌';
    const pObj = it.extracted_parameters || {{}};
    const pStr = Object.keys(pObj).length > 0 ? Object.entries(pObj).map(([k, v]) => k + '=' + v).join(', ') : '-';
    rowsHtml +=
      '<tr>' +
      '<td><code>' + escapeHtmlBasic(it.filename) + '</code></td>' +
      '<td>' + escapeHtmlBasic(it.expected_name) + '</td>' +
      '<td>' + dMark + ' ' + escapeHtmlBasic(it.predicted_name) + '</td>' +
      '<td>' + pMark + ' <code>' + escapeHtmlBasic(pStr) + '</code></td>' +
      '<td>' + badge + '</td>' +
      '</tr>';
  }}

  body.innerHTML = summaryHtml + (items.length > 0 ?
    '<table><thead><tr><th>Arquivo</th><th>Gabarito</th><th>Distribuição</th><th>Parâmetros</th><th>Nota</th></tr></thead><tbody>' + rowsHtml + '</tbody></table>'
    : '');
}}

async function openRealAccuracyModal() {{
  const modal = document.getElementById('realAccuracyModal');
  if (modal) modal.style.display = 'flex';
  try {{
    const resp = await fetch('/api/real-accuracy', {{ cache: 'no-store' }});
    const data = await resp.json();
    updateRealAccuracyPillUI(data);
    renderRealAccuracyModalBody(data);
  }} catch (e) {{
    const body = document.getElementById('realAccuracyModalBody');
    if (body) body.innerHTML = '<div style="color:#fda4af;">Erro ao carregar dados do teste real.</div>';
  }}
}}

function closeRealAccuracyModal() {{
  const modal = document.getElementById('realAccuracyModal');
  if (modal) modal.style.display = 'none';
}}

async function runRealAccuracyEval() {{
  const btn = document.getElementById('runRealEvalBtn');
  const body = document.getElementById('realAccuracyModalBody');
  if (btn) {{ btn.disabled = true; btn.textContent = '⏳ Avaliando imagens em modo cego...'; }}
  if (body) body.innerHTML = '<div style="padding:24px;text-align:center;color:#bae6fd;">🧪 Executando avaliação cega nas imagens de <code>real_test_data/</code>... Aguarde alguns segundos.</div>';
  try {{
    const resp = await fetch('/api/real-accuracy/run', {{ method: 'POST' }});
    const data = await resp.json();
    updateRealAccuracyPillUI(data);
    renderRealAccuracyModalBody(data);
  }} catch (e) {{
    if (body) body.innerHTML = '<div style="color:#fda4af;">Erro ao executar a avaliação cega.</div>';
  }} finally {{
    if (btn) {{ btn.disabled = false; btn.textContent = '🔄 Reavaliar Pasta Agora'; }}
  }}
}}

if ({real_eval_needs_poll_js}) {{
  (function pollRealEval(attempts) {{
    if (attempts <= 0) return;
    setTimeout(async () => {{
      try {{
        const resp = await fetch('/api/real-accuracy', {{ cache: 'no-store' }});
        const data = await resp.json();
        updateRealAccuracyPillUI(data);
        if (!data.is_fresh && (data.image_count_in_folder || 0) > 0) {{
          pollRealEval(attempts - 1);
        }}
      }} catch (e) {{}}
    }}, 1500);
  }})(20);
}}
</script>
</body>
</html>"""
        encoded = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_standalone_server(port: int = 8501) -> None:
    import threading
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", port), StatVisionHTTPHandler)

    def _background_warmup_and_eval() -> None:
        try:
            StatVisionHTTPHandler.analyzer.ocr_engine._load_learned_data()
            cached = get_cached_real_evaluation()
            if (cached.get("image_count_in_folder") or 0) > 0 and not cached.get("is_fresh"):
                evaluate_real_test_folder(analyzer=StatVisionHTTPHandler.analyzer, force=False)
        except Exception:
            pass

    # Pré-aquece padrões treinados e atualiza a acurácia real em segundo plano sem atrasar a abertura
    threading.Thread(target=_background_warmup_and_eval, daemon=True).start()
    print(f"=====================================================================")
    print(f" StatMath rodando em: http://127.0.0.1:{port}")
    print(f" Pressione Ctrl+C para encerrar.")
    print(f"=====================================================================")
    server.serve_forever()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv[1:]) >= 1 and sys.argv[1].isdigit() else 8501
    run_standalone_server(port=port_arg)
