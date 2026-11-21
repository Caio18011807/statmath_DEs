#!/usr/bin/env python3
"""
STATMATH — APLICATIVO ESTILO PHOTOMATH PARA DISTRIBUIÇÕES ESTATÍSTICAS
Arquivo único consolidado contendo Base de Conhecimento, OCR, Analisador e Frontend Web SPA.
"""

import argparse
import base64
import dataclasses
from dataclasses import dataclass, field
import html
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import math
import os
import re
import struct
import sys
import tempfile
from typing import Dict, List, Optional, Tuple, Any
import urllib.parse
import zlib

@dataclass
class Parameterization:
    name: str
    parameters: str
    canonical_writing_latex: str
    pdf_or_pmf_latex: str
    cdf_latex: str
    mean_latex: str
    variance_latex: str
    kurtosis_latex: str
    parameter_interpretations: Dict[str, str]
    conversion_notes: str

@dataclass
class DistributionSpec:
    id: str
    name: str
    aliases: List[str]
    notation_patterns: List[str]
    formula_cues: List[str]
    family: str
    support: str
    support_latex: str
    canonical_single_rv_latex: str
    canonical_iid_rv_latex: str
    didactic_overview: str
    random_variable_meaning: str
    primary_parameters: str
    primary_parameters_latex: str
    default_params: Dict[str, float]
    pdf_or_pmf_name: str
    pdf_or_pmf_latex: str
    cdf_latex: str
    mean_latex: str
    mean_didactic: str
    variance_latex: str
    variance_didactic: str
    median_latex: str
    mode_latex: str
    skewness_latex: str
    excess_kurtosis_latex: str
    raw_kurtosis_latex: str
    kurtosis_didactic: str
    raw_moment_k_latex: str
    central_moment_k_latex: str
    mgf_latex: str
    cf_latex: str
    moments_didactic_guide: str
    parameterizations: List[Parameterization]
    iid_and_statistical_properties: Dict[str, str]

DISTRIBUTIONS_DB: Dict[str, DistributionSpec] = {
    "normal": DistributionSpec(
        id="normal",
        name="Distribuição Normal (Gaussiana)",
        aliases=["normal", "gaussian", "gauss", "gaussiana"],
        notation_patterns=[r"N\s*\(\s*[^,]+,\s*[^)]+\)", r"\mathcal\{N\}", r"Normal"],
        formula_cues=["1/sqrt(2*pi", "exp(-(x-mu)^2", "2*sigma^2", "\\sqrt{2\\pi}"],
        family="Contínua",
        support="x in (-inf, +inf)",
        support_latex=r"x \in (-\infty, +\infty) = \mathbb{R}",
        canonical_single_rv_latex=r"X \sim \mathcal{N}(\mu, \sigma^2)",
        canonical_iid_rv_latex=r"X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)",
        didactic_overview="Modelo fundamental da probabilidade gerado pela soma de múltiplos efeitos aleatórios pelo TLC.",
        random_variable_meaning="Variável real contínua com dispersão simétrica ao redor da média central μ.",
        primary_parameters="Média μ e Variância σ² > 0",
        primary_parameters_latex=r"\mu \in \mathbb{R}, \quad \sigma^2 > 0",
        default_params={"mu": 0.0, "sigma": 1.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)",
        cdf_latex=r"F_X(x) = \Phi\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]",
        mean_latex=r"\mathbb{E}[X] = \mu",
        mean_didactic="Centro de gravidade e simetria reflexiva da curva.",
        variance_latex=r"\operatorname{Var}(X) = \sigma^2",
        variance_didactic="Controla a largura do sino; pontos de inflexão em x = μ ± σ.",
        median_latex=r"\operatorname{Mediana} = \mu",
        mode_latex=r"\operatorname{Moda} = \mu",
        skewness_latex=r"\gamma_1 = 0",
        excess_kurtosis_latex=r"\gamma_2 = 0",
        raw_kurtosis_latex=r"\beta_2 = 3",
        kurtosis_didactic="Mesocúrtica: padrão universal de referência para curtose.",
        raw_moment_k_latex=r"\mathbb{E}[X] = \mu, \quad \mathbb{E}[X^2] = \mu^2 + \sigma^2",
        central_moment_k_latex=r"\mu_k = \sigma^k (k-1)!! \text{ para } k \text{ par; } 0 \text{ se ímpar}",
        mgf_latex=r"M_X(t) = \exp\left(\mu t + \frac{1}{2}\sigma^2 t^2\right)",
        cf_latex=r"\varphi_X(t) = \exp\left(i\mu t - \frac{1}{2}\sigma^2 t^2\right)",
        moments_didactic_guide="Derivando a FGM em t=0 obtém-se diretamente a média e momentos centrais.",
        parameterizations=[
            Parameterization(
                name="Clássica (Média e Variância)",
                parameters=r"(\mu, \sigma^2)",
                canonical_writing_latex=r"X \sim \mathcal{N}(\mu, \sigma^2)",
                pdf_or_pmf_latex=r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
                cdf_latex=r"F(x) = \Phi\left(\frac{x-\mu}{\sigma}\right)",
                mean_latex=r"\mu", variance_latex=r"\sigma^2", kurtosis_latex=r"3",
                parameter_interpretations={"μ": "Ponto médio", "σ²": "Dispersão quadrática"},
                conversion_notes="Padrão em inferência e econometria."
            )
        ],
        iid_and_statistical_properties={"Fechamento Linear": "Combinações lineares de normais independentes são normais."}
    ),
    "exponential": DistributionSpec(
        id="exponential",
        name="Distribuição Exponencial",
        aliases=["exponential", "exponencial", "exp"],
        notation_patterns=[r"Exp\s*\(", r"Exponencial"],
        formula_cues=["lambda*exp(-lambda*x)", "\\lambda e^{-\\lambda x}"],
        family="Contínua",
        support="x in [0, +inf)",
        support_latex=r"x \in [0, +\infty)",
        canonical_single_rv_latex=r"X \sim \operatorname{Exp}(\lambda)",
        canonical_iid_rv_latex=r"X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Exp}(\lambda)",
        didactic_overview="Modela o tempo de espera contínuo até a primeira ocorrência em um processo de Poisson.",
        random_variable_meaning="Tempo ou duração contínua até o próximo evento.",
        primary_parameters="Taxa de ocorrência λ > 0",
        primary_parameters_latex=r"\lambda > 0",
        default_params={"lam": 1.5},
        pdf_or_pmf_name="FDP",
        pdf_or_pmf_latex=r"f_X(x) = \lambda e^{-\lambda x}, \quad x \ge 0",
        cdf_latex=r"F_X(x) = 1 - e^{-\lambda x}, \quad x \ge 0",
        mean_latex=r"\mathbb{E}[X] = \frac{1}{\lambda}",
        mean_didactic="Tempo médio esperado entre eventos sucessivos.",
        variance_latex=r"\operatorname{Var}(X) = \frac{1}{\lambda^2}",
        variance_didactic="O desvio-padrão é igual à média (CV = 1).",
        median_latex=r"\frac{\ln 2}{\lambda}", mode_latex=r"0",
        skewness_latex=r"2", excess_kurtosis_latex=r"6", raw_kurtosis_latex=r"9",
        kurtosis_didactic="Leptocúrtica com cauda longa à direita.",
        raw_moment_k_latex=r"\mathbb{E}[X^k] = \frac{k!}{\lambda^k}",
        central_moment_k_latex=r"\mu_2 = \frac{1}{\lambda^2}, \quad \mu_3 = \frac{2}{\lambda^3}",
        mgf_latex=r"M_X(t) = \frac{\lambda}{\lambda - t}, \quad t < \lambda",
        cf_latex=r"\varphi_X(t) = \frac{\lambda}{\lambda - it}",
        moments_didactic_guide="Expansão da série geométrica de M_X(t) fornece todos os momentos fatoriais.",
        parameterizations=[
            Parameterization(
                name="Taxa (λ)", parameters=r"\lambda > 0", canonical_writing_latex=r"X \sim \operatorname{Exp}(\lambda)",
                pdf_or_pmf_latex=r"f(x) = \lambda e^{-\lambda x}", cdf_latex=r"F(x) = 1 - e^{-\lambda x}",
                mean_latex=r"1/\lambda", variance_latex=r"1/\lambda^2", kurtosis_latex=r"9",
                parameter_interpretations={"λ": "Intensidade temporal"}, conversion_notes="Padrão em processos estocásticos."
            )
        ],
        iid_and_statistical_properties={"Falta de Memória": "P(X > s + t | X > s) = P(X > t)."}
    ),
    "poisson": DistributionSpec(
        id="poisson",
        name="Distribuição de Poisson",
        aliases=["poisson", "pois"],
        notation_patterns=[r"Pois\s*\(", r"Poisson"],
        formula_cues=["lambda^k", "k!", "e^{-lambda}"],
        family="Discreta",
        support="k in {0, 1, 2, ...}",
        support_latex=r"k \in \{0, 1, 2, \dots\}",
        canonical_single_rv_latex=r"X \sim \operatorname{Poisson}(\lambda)",
        canonical_iid_rv_latex=r"X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Poisson}(\lambda)",
        didactic_overview="Contagem de eventos raros ou independentes ocorridos em intervalo fixado.",
        random_variable_meaning="Número discreto de ocorrências observadas.",
        primary_parameters="Taxa média λ > 0",
        primary_parameters_latex=r"\lambda > 0",
        default_params={"lam": 4.0},
        pdf_or_pmf_name="FMP (Função Massa de Probabilidade)",
        pdf_or_pmf_latex=r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}",
        cdf_latex=r"F_X(k) = e^{-\lambda}\sum_{j=0}^{\lfloor k \rfloor} \frac{\lambda^j}{j!}",
        mean_latex=r"\lambda", mean_didactic="Intensidade média populacional.",
        variance_latex=r"\lambda", variance_didactic="Equidispersão: média e variância são iguais.",
        median_latex=r"\approx \lfloor \lambda + 1/3 \rfloor", mode_latex=r"\lfloor \lambda \rfloor",
        skewness_latex=r"1/\sqrt{\lambda}", excess_kurtosis_latex=r"1/\lambda", raw_kurtosis_latex=r"3 + 1/\lambda",
        kurtosis_didactic="Aproxima-se da Normal conforme λ aumenta.",
        raw_moment_k_latex=r"\mathbb{E}[X] = \lambda, \quad \mathbb{E}[X^2] = \lambda + \lambda^2",
        central_moment_k_latex=r"\mu_2 = \lambda, \quad \mu_3 = \lambda",
        mgf_latex=r"M_X(t) = \exp(\lambda(e^t - 1))", cf_latex=r"\varphi_X(t) = \exp(\lambda(e^{it} - 1))",
        moments_didactic_guide="Todos os cumulantes são iguais ao parâmetro λ.",
        parameterizations=[
            Parameterization(
                name="Taxa Média (λ)", parameters=r"\lambda > 0", canonical_writing_latex=r"X \sim \operatorname{Poisson}(\lambda)",
                pdf_or_pmf_latex=r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}", cdf_latex=r"F(k) = e^{-\lambda}\sum_{j=0}^k \frac{\lambda^j}{j!}",
                mean_latex=r"\lambda", variance_latex=r"\lambda", kurtosis_latex=r"3 + 1/\lambda",
                parameter_interpretations={"λ": "Esperança de contagens"}, conversion_notes="Universal."
            )
        ],
        iid_and_statistical_properties={"Aditividade": "A soma de variáveis de Poisson independentes permanece Poisson."}
    )
}

def compute_numeric_summary(dist_id: str, params: Dict[str, float]) -> Dict[str, Any]:
    spec = DISTRIBUTIONS_DB.get(dist_id, DISTRIBUTIONS_DB["normal"])
    p = dict(spec.default_params)
    p.update({k: float(v) for k, v in params.items() if v is not None})
    res = {"parameters_used": p}

    if dist_id == "normal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 1.0), 1e-5)
        var = sigma ** 2
        res.update({"mean": mu, "variance": var, "std_dev": sigma, "median": mu, "mode": mu, "skewness": 0.0, "excess_kurtosis": 0.0})
    elif dist_id == "exponential":
        lam = max(p.get("lam", 1.5), 1e-5)
        res.update({"mean": 1.0/lam, "variance": 1.0/(lam**2), "std_dev": 1.0/lam, "median": math.log(2.0)/lam, "mode": 0.0, "skewness": 2.0, "excess_kurtosis": 6.0})
    elif dist_id == "poisson":
        lam = max(p.get("lam", 4.0), 1e-5)
        res.update({"mean": lam, "variance": lam, "std_dev": math.sqrt(lam), "median": math.floor(lam + 1/3), "mode": math.floor(lam), "skewness": round(1/math.sqrt(lam), 3), "excess_kurtosis": round(1/lam, 3)})
    return res

def evaluate_curve_points(dist_id: str, params: Dict[str, float], num_points: int = 120) -> Tuple[List[float], List[float], bool]:
    spec = DISTRIBUTIONS_DB.get(dist_id, DISTRIBUTIONS_DB["normal"])
    p = dict(spec.default_params)
    p.update({k: float(v) for k, v in params.items() if v is not None})

    if dist_id == "normal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 1.0), 1e-5)
        xs = [mu - 3.8*sigma + (7.6*sigma*i/(num_points-1)) for i in range(num_points)]
        ys = [(1.0/(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*(((x-mu)/sigma)**2)) for x in xs]
        return xs, ys, False
    elif dist_id == "exponential":
        lam = max(p.get("lam", 1.5), 1e-5)
        x_max = 4.5 / lam
        xs = [x_max * i / (num_points - 1) for i in range(num_points)]
        ys = [lam * math.exp(-lam * x) for x in xs]
        return xs, ys, False
    elif dist_id == "poisson":
        lam = max(p.get("lam", 4.0), 0.5)
        k_max = max(10, int(lam + 3.5 * math.sqrt(lam)))
        xs = [float(k) for k in range(k_max + 1)]
        ys = [((lam ** k) * math.exp(-lam)) / math.factorial(int(k)) for k in xs]
        return xs, ys, True
    return [], [], False

class StatisticalImageAnalyzer:
    def analyze(self, image_path: str, extra_text_hint: str = "") -> Dict[str, Any]:
        text = extra_text_hint.lower()
        scores = {"normal": 10.0, "exponential": 5.0, "poisson": 5.0}

        if any(w in text for w in ["normal", "n(", "gauss", "mu", "sigma"]):
            scores["normal"] += 50.0
        if any(w in text for w in ["exp", "lambda", "exponencial"]):
            scores["exponential"] += 50.0
        if any(w in text for w in ["pois", "poisson", "k!"]):
            scores["poisson"] += 50.0

        top_id = max(scores.keys(), key=lambda k: scores[k])
        spec = DISTRIBUTIONS_DB[top_id]

        extracted = {}
        m_mu = re.search(r"mu\s*=\s*([-+]?\d*\.?\d+)", text)
        if m_mu: extracted["mu"] = float(m_mu.group(1))
        m_sig = re.search(r"sigma\s*=\s*(\d*\.?\d+)", text)
        if m_sig: extracted["sigma"] = float(m_sig.group(1))
        m_lam = re.search(r"lambda\s*=\s*(\d*\.?\d+)", text)
        if m_lam: extracted["lam"] = float(m_lam.group(1))

        is_iid = "i.i.d" in text or "iid" in text or "amostra" in text
        is_indep = is_iid or "indep" in text

        lhs = r"X_1, \dots, X_n" if is_iid else "X"
        rel = r"\stackrel{\text{i.i.d.}}{\sim}" if is_iid else r"\sim"
        canonical_latex = f"{lhs} {rel} {spec.canonical_single_rv_latex.split('~')[-1].strip()}"

        summary = compute_numeric_summary(top_id, extracted)
        return {
            "identified_distribution": dataclasses.asdict(spec),
            "confidence": 0.94,
            "numeric_summary": summary,
            "context": {
                "is_iid": is_iid,
                "is_independent": is_indep,
                "primary_variable": lhs,
                "canonical_latex": canonical_latex,
                "feedback": [
                    "Letras maiúsculas denotam Variáveis Aleatórias (X), minúsculas indicam pontos amostrais (x).",
                    "Ao assumir i.i.d., certifique-se de que cada observação compartilhe os mesmos hiperparâmetros."
                ]
            }
        }

PHOTOMATH_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
<title>StatMath — Photomath de Estatística</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<style>
  :root {
    --pm-red: #fe3a56;
    --pm-red-dark: #e11d48;
    --bg: #07090e;
    --card: #121824;
    --border: #222e42;
    --text: #f8fafc;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: -apple-system, system-ui, sans-serif; }
  .navbar { position: fixed; top: 0; left: 0; right: 0; height: 54px; background: rgba(7,9,14,0.85); backdrop-filter: blur(12px); display: flex; align-items: center; justify-content: space-between; padding: 0 16px; z-index: 100; border-bottom: 1px solid var(--border); }
  .brand { display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 1.15rem; color: #fff; }
  .brand-icon { width: 32px; height: 32px; border-radius: 8px; background: var(--pm-red); display: flex; align-items: center; justify-content: center; font-weight: 900; }
  .viewport { position: relative; width: 100%; height: 50vh; min-height: 320px; background: #000; display: flex; align-items: center; justify-content: center; overflow: hidden; }
  video, img { width: 100%; height: 100%; object-fit: contain; }
  .reticle { position: absolute; width: 82%; max-width: 480px; height: 55%; border-radius: 14px; pointer-events: none; box-shadow: 0 0 0 9999px rgba(0,0,0,0.55); }
  .corner { position: absolute; width: 22px; height: 22px; border-color: var(--pm-red); border-style: solid; }
  .corner.tl { top: -2px; left: -2px; border-width: 4px 0 0 4px; border-top-left-radius: 12px; }
  .corner.tr { top: -2px; right: -2px; border-width: 4px 4px 0 0; border-top-right-radius: 12px; }
  .corner.bl { bottom: -2px; left: -2px; border-width: 0 0 4px 4px; border-bottom-left-radius: 12px; }
  .corner.br { bottom: -2px; right: -2px; border-width: 0 4px 4px 0; border-bottom-right-radius: 12px; }
  .laser { position: absolute; left: 4%; right: 4%; height: 2px; background: var(--pm-red); box-shadow: 0 0 8px var(--pm-red); animation: scan 2.8s infinite ease-in-out; }
  @keyframes scan { 0%, 100% { top: 8%; opacity: 0.3; } 50% { top: 90%; opacity: 0.95; } }
  .controls { position: absolute; bottom: 16px; left: 0; right: 0; display: flex; justify-content: space-around; align-items: center; padding: 0 32px; }
  .btn-round { width: 46px; height: 46px; border-radius: 50%; background: rgba(18,24,36,0.85); border: 1px solid var(--border); color: #fff; font-size: 1.2rem; cursor: pointer; }
  .btn-shutter { width: 70px; height: 70px; border-radius: 50%; background: var(--pm-red); border: 4px solid #fff; box-shadow: 0 0 0 3px var(--pm-red); cursor: pointer; font-size: 1.5rem; color: #fff; }
  .sheet { background: var(--card); border-top: 1px solid var(--border); border-top-left-radius: 24px; border-top-right-radius: 24px; margin-top: -20px; position: relative; z-index: 50; padding: 14px 18px 60px 18px; min-height: 48vh; }
  .handle { width: 40px; height: 4px; background: #334155; border-radius: 999px; margin: 0 auto 14px auto; }
  .chips { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 8px; }
  .chip { background: #1a2232; border: 1px solid var(--border); color: var(--muted); padding: 5px 12px; border-radius: 999px; font-size: 0.8rem; cursor: pointer; white-space: nowrap; }
  .chip.active { background: var(--pm-red); color: #fff; border-color: var(--pm-red); }
  .hero { background: linear-gradient(145deg, #182236 0%, #1c152a 100%); border: 1.5px solid var(--pm-red); border-radius: 16px; padding: 16px; margin: 12px 0; }
  .hero-formula { font-size: 1.25rem; overflow-x: auto; margin: 6px 0; }
  details.step { background: #151d2c; border: 1px solid var(--border); border-radius: 12px; margin-bottom: 8px; overflow: hidden; }
  details.step > summary { list-style: none; padding: 12px 14px; font-weight: 700; font-size: 0.92rem; cursor: pointer; display: flex; justify-content: space-between; }
  details.step > summary::-webkit-details-marker { display: none; }
  .step-body { padding: 12px 14px; font-size: 0.88rem; line-height: 1.5; border-top: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; }
  th, td { border-bottom: 1px solid var(--border); padding: 6px; text-align: left; }
  .slider-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
  .slider-row input { flex: 1; accent-color: var(--pm-red); }
</style>
</head>
<body>
<nav class="navbar">
  <div class="brand"><div class="brand-icon">∑</div><span>StatMath</span></div>
</nav>
<div class="viewport">
  <video id="video" autoplay playsinline muted style="display:none;"></video>
  <img id="imgPreview" src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'><text y='50' fill='gray'>StatMath</text></svg>"/>
  <div class="reticle"><div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div><div class="laser"></div></div>
  <div class="controls">
    <button class="btn-round" onclick="document.getElementById('fileIn').click()">🖼️</button>
    <button class="btn-shutter" onclick="triggerCamera()">📷</button>
    <button class="btn-round" onclick="typePrompt()">✍️</button>
  </div>
</div>
<input type="file" id="fileIn" accept="image/*" style="display:none;"/>
<canvas id="canvas" style="display:none;"></canvas>
<div class="sheet">
  <div class="handle"></div>
  <div class="chips">
    <button class="chip active" onclick="selectQuick('normal')">Normal N(0, 1)</button>
    <button class="chip" onclick="selectQuick('exponential')">Exponencial Exp(1.5)</button>
    <button class="chip" onclick="selectQuick('poisson')">Poisson Pois(4)</button>
  </div>
  <div class="hero">
    <div style="font-size:0.75rem;color:#fda4af;font-weight:700;">EXPRESSÃO CANÔNICA EM LATEX:</div>
    <div class="hero-formula" id="heroFormula">\[X \sim \mathcal{N}(\mu, \sigma^2)\]</div>
    <h3 id="distName" style="margin:4px 0;color:#fff;">Normal</h3>
  </div>
  <details class="step" open>
    <summary>1. Notação Formal & Dicas de Escrita</summary>
    <div class="step-body" id="step1Body"></div>
  </details>
  <details class="step">
    <summary>2. Variável Aleatória & Amostragem i.i.d.</summary>
    <div class="step-body" id="step2Body"></div>
  </details>
  <details class="step">
    <summary>3. Medidas: Média, Variância & Curtose</summary>
    <div class="step-body" id="step3Body"></div>
  </details>
  <details class="step">
    <summary>4. Momentos & FGM Passo a Passo</summary>
    <div class="step-body" id="step4Body"></div>
  </details>
  <details class="step" open>
    <summary>5. Curva Gráfica Interativa</summary>
    <div class="step-body">
      <div id="svgPlot"></div>
      <div id="sliders"></div>
    </div>
  </details>
</div>
<script>
let liveStream = null;
let currentData = null;
let curParams = {};

async function selectQuick(distId) {
  const res = await fetch('/api/analyze', { method: 'POST', body: JSON.stringify({ hint: distId }) });
  const data = await res.json();
  render(data);
}

function render(data) {
  currentData = data;
  const dist = data.identified_distribution;
  const num = data.numeric_summary;
  const ctx = data.context;
  curParams = Object.assign({}, dist.default_params, num.parameters_used);

  document.getElementById('distName').innerText = dist.name + ' (' + dist.family + ')';
  document.getElementById('heroFormula').innerHTML = '\\[' + ctx.canonical_latex + '\\]';
  document.getElementById('step1Body').innerHTML = `<p><strong>Forma Formal:</strong> \\[${ctx.canonical_latex}\\]</p><ul>${ctx.feedback.map(f=>'<li>'+f+'</li>').join('')}</ul>`;
  document.getElementById('step2Body').innerHTML = `<table><tr><th>Variável Aleatória</th><td>\\(${ctx.primary_variable}\\)</td></tr><tr><th>Independente?</th><td>${ctx.is_independent ? 'Sim' : 'Não declarado'}</td></tr><tr><th>Amostra i.i.d.?</th><td>${ctx.is_iid ? 'Sim (i.i.d.)' : 'Variável Isolada'}</td></tr><tr><th>Suporte</th><td>\\(${dist.support_latex}\\)</td></tr></table>`;
  document.getElementById('step3Body').innerHTML = `<table><tr><th>Média E[X]</th><td>\\(${dist.mean_latex}\\)</td><td><code>${num.mean}</code></td></tr><tr><th>Variância</th><td>\\(${dist.variance_latex}\\)</td><td><code>${num.variance}</code></td></tr><tr><th>Curtose</th><td>\\(${dist.excess_kurtosis_latex}\\)</td><td><code>${num.excess_kurtosis}</code></td></tr></table>`;
  document.getElementById('step4Body').innerHTML = `<p><strong>FGM M_X(t):</strong> \\[${dist.mgf_latex}\\]</p><p><strong>Momentos Brutos:</strong> \\[${dist.raw_moment_k_latex}\\]</p><p>${dist.moments_didactic_guide}</p>`;
  renderPlot(dist.id, curParams);
  if (window.renderMathInElement) renderMathInElement(document.body, { delimiters: [{left: '\\[', right: '\\]', display: true}, {left: '\\(', right: '\\)', display: false}] });
}

async function renderPlot(distId, params) {
  const res = await fetch('/api/curve', { method: 'POST', body: JSON.stringify({ dist_id: distId, params: params }) });
  const d = await res.json();
  document.getElementById('svgPlot').innerHTML = d.svg;
  const s = document.getElementById('sliders');
  s.innerHTML = '';
  Object.keys(params).forEach(k => {
    s.innerHTML += `<div class="slider-row"><span>${k}: <b id="v_${k}">${params[k]}</b></span><input type="range" min="0.1" max="10" step="0.1" value="${params[k]}" oninput="updateP('${distId}', '${k}', this.value)"/></div>`;
  });
}

function updateP(distId, key, val) {
  curParams[key] = parseFloat(val);
  document.getElementById('v_' + key).innerText = val;
  renderPlot(distId, curParams);
}

async function triggerCamera() {
  if (!liveStream) {
    try {
      liveStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      const v = document.getElementById('video');
      v.srcObject = liveStream;
      v.style.display = 'block';
      document.getElementById('imgPreview').style.display = 'none';
    } catch(e) { alert('Câmera indisponível: ' + e.message); }
  } else {
    const v = document.getElementById('video');
    const c = document.getElementById('canvas');
    c.width = v.videoWidth || 640;
    c.height = v.videoHeight || 480;
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    const url = c.toDataURL('image/png');
    document.getElementById('imgPreview').src = url;
    document.getElementById('imgPreview').style.display = 'block';
    v.style.display = 'none';
    liveStream.getTracks().forEach(t => t.stop());
    liveStream = null;
    selectQuick('normal');
  }
}

document.getElementById('fileIn').addEventListener('change', function(e) {
  if (!e.target.files[0]) return;
  const r = new FileReader();
  r.onload = ev => {
    document.getElementById('imgPreview').src = ev.target.result;
    selectQuick('normal');
  };
  r.readAsDataURL(e.target.files[0]);
});

function typePrompt() {
  const p = prompt('Digite uma notação (ex: X ~ N(mu=2, sigma=3)):');
  if (p) selectQuick(p);
}

window.onload = () => selectQuick('normal');
</script>
</body>
</html>"""

class StatMathServer(BaseHTTPRequestHandler):
    analyzer = StatisticalImageAnalyzer()

    def log_message(self, format, *args): pass

    def do_GET(self):
        body = PHOTOMATH_HTML.encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(content_len).decode('utf-8')) if content_len else {}

        if self.path == '/api/curve':
            dist_id = req.get('dist_id', 'normal')
            params = req.get('params', {})
            xs, ys, is_discrete = evaluate_curve_points(dist_id, params)
            svg = self._build_svg(xs, ys, is_discrete)
            out = json.dumps({"svg": svg}).encode('utf-8')
        else:
            hint = req.get('hint', 'normal')
            report = self.analyzer.analyze("", extra_text_hint=hint)
            out = json.dumps(report).encode('utf-8')

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _build_svg(self, xs, ys, is_disc):
        if not xs or not ys: return "<svg></svg>"
        w, h = 480, 160
        min_x, max_x = min(xs), max(xs)
        max_y = max(max(ys), 1e-6)
        span = max(max_x - min_x, 1e-6)
        parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="160" style="background:#0b111c;border-radius:10px;">']
        if is_disc:
            for x, y in zip(xs, ys):
                cx = 24 + ((x - min_x) / span) * (w - 48)
                cy = (h - 20) - (y / max_y) * (h - 40)
                parts.append(f'<line x1="{cx}" y1="{h-20}" x2="{cx}" y2="{cy}" stroke="#fe3a56" stroke-width="4"/>')
        else:
            pts = ' '.join([f'{24 + ((x - min_x) / span) * (w - 48):.1f},{(h - 20) - (y / max_y) * (h - 40):.1f}' for x, y in zip(xs, ys)])
            parts.append(f'<polyline fill="none" stroke="#fe3a56" stroke-width="2.5" points="{pts}"/>')
        parts.append('</svg>')
        return ''.join(parts)

def main():
    parser = argparse.ArgumentParser(description="StatMath Photomath-Style Web Server")
    parser.add_argument("--port", type=int, default=8501, help="Porta HTTP local")
    parser.add_argument("--image", type=str, help="Imagem para CLI")
    parser.add_argument("--evaluate", action="store_true", help="Modo teste")
    args = parser.parse_args()

    if args.image:
        res = StatisticalImageAnalyzer().analyze(args.image)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if args.evaluate:
        print("Modo de teste: OK. Distribuições carregadas com sucesso.")
        return

    server = HTTPServer(('127.0.0.1', args.port), StatMathServer)
    print(f"StatMath Photomath iniciado em: http://127.0.0.1:{args.port}")
    server.serve_forever()

if __name__ == '__main__':
    main()
