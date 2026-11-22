#!/usr/bin/env python3
"""
====================================================================================================
 STATMATH (STATVISION OCR) — ARQUIVO ÚNICO CONSOLIDADO PARA GOOGLE AI STUDIO
====================================================================================================
Este arquivo único reúne TODOS os módulos Python do projeto StatVision OCR / StatMath
(Tutor Inteligente estilo Photomath para reconhecimento de distribuições estatísticas escritas à mão):

  1. Base de Conhecimento Simbólica e Educacional (`DISTRIBUTIONS_DB` com 10 famílias de distribuições,
     todas as parametrizações, FDP/FMP, FDA Acumulada, Média, Variância, Mediana, Moda, Curtose,
     Fórmula Geral de Momentos, FGM, FC e propriedades de Independência e Amostragem i.i.d.)
  2. Motor Numérico e Avaliador de Momentos e Curvas (`compute_numeric_summary`, `evaluate_curve_points`)
  3. Motor de OCR Offline para Escrita Manuscrita e Parser Estatístico (`OfflineOCREngine`)
  4. Analisador Geométrico de Curvas e Propriedades da Imagem (`CurveShapeAnalyzer`)
  5. Gerador de Cadernos Manuscritos de Exemplo (`generate_all_demo_samples`)
  6. Pipeline Unificado de Análise Educacional (`StatisticalImageAnalyzer`)
  7. Sistema de Treinamento Contínuo e Verificação de Acertos/Erros (`train_and_evaluate`)
  8. Aplicativo Front-End Estilo Photomath com Scanner de Câmera, Permissão de Câmera,
     Card Principal em LaTeX e Passos Retráteis/Minimizáveis (`app.py` + `cli.py`)

Como executar:
  - Iniciar Aplicativo Web Estilo Photomath (http://127.0.0.1:8501):
      python3 stat_vision_completo_ai_studio.py
  - Avaliar acertos/erros nas pastas de treinamento (`training_data/<distribuicao>/`):
      python3 stat_vision_completo_ai_studio.py --evaluate
  - Treinar automaticamente com novas fotos nas pastas:
      python3 stat_vision_completo_ai_studio.py --learn
  - Analisar uma imagem via linha de comando (CLI):
      python3 stat_vision_completo_ai_studio.py --image caminho/para/foto.png
====================================================================================================
"""

from __future__ import annotations
import argparse
import base64
import dataclasses
from dataclasses import dataclass, field
import datetime
import difflib
import hashlib
import html
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Dict, List, Optional, Tuple, Callable, Any
import urllib.parse
import zlib


# ================================================================================================
# SEÇÃO 1. BASE DE CONHECIMENTO SIMBÓLICA E EDUCACIONAL (knowledge_base.py)
# ================================================================================================

"""
Base de Conhecimento Simbólica e Educacional de Distribuições Estatísticas (Português Brasileiro).

Focada no ensino universitário e intuitivo de Probabilidade e Estatística a partir
de imagens de notações escritas à mão (ex.: X_1, ..., X_n ~ i.i.d. N(μ, σ²)).

Para cada distribuição, armazena:
- Como deveria ser escrito (notação formal tipográfica em LaTeX para variável única e amostra i.i.d.)
- Nome, família e suporte da variável aleatória
- Explicação didática intuitiva ("O que esta distribuição modela na prática?")
- Todas as parametrizações com:
  * Função Densidade de Probabilidade (FDP / FMP) naquela parametrização
  * Função de Distribuição Acumulada (FDA / CDF) naquela parametrização
  * Média, Variância e Curtose naquela parametrização
  * Interpretação física, geométrica e estatística de CADA parâmetro daquela parametrização
- Fórmulas de momentos (Momentos Brutos E[X^k], Momentos Centrais E[(X-μ)^k], Função Geradora de Momentos M_X(t) com passo a passo de derivação)
- Média, Variância, Mediana, Moda, Assimetria, Excesso de Curtose e Curtose Bruta com interpretação didática
- Propriedades de Independência, Amostragem i.i.d. e papel da Variável Aleatória
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


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
    family: str                   # "Contínua" ou "Discreta"
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
    entropy_latex: str
    fisher_info_latex: str
    parameterizations: List[Parameterization]
    iid_and_statistical_properties: Dict[str, str]
    shape_signature: Dict[str, Any]


DISTRIBUTIONS_DB: Dict[str, DistributionSpec] = {
    "normal": DistributionSpec(
        id="normal",
        name="Distribuição Normal (ou Gaussiana)",
        aliases=["normal", "gaussian", "gauss", "gaussiana", "standard normal", "normal padrao"],
        notation_patterns=[
            r"\bN\s*\(\s*[^,]+,\s*[^)]+\)",
            r"\mathcal\{N\}",
            r"\bNormal\s*\(",
            r"\bGauss",
            r"\bZ\s*~\s*N\s*\(\s*0\s*,\s*1\s*\)",
        ],
        formula_cues=["1/sqrt(2*pi", "exp(-(x-mu)^2", "2*sigma^2", "e^{-(x-", "\\sqrt{2\\pi}"],
        family="Contínua",
        support="x in (-infinito, +infinito)",
        support_latex=r"x \in (-\infty, +\infty) = \mathbb{R}",
        canonical_single_rv_latex=r"X \sim \mathcal{N}(\mu, \sigma^2)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)",
        didactic_overview=(
            "A Distribuição Normal (ou Gaussiana) é a mais importante de toda a probabilidade e estatística. "
            "Pelo Teorema Central do Limite (TCL), sempre que uma grandeza resulta da soma de muitos pequenos efeitos "
            "aleatórios independentes (como erros de medição, altura humana, notas em exames ou ruído térmico), "
            "seu comportamento converge para a curva em forma de sino (bell curve) da Normal."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua X representa uma medição real (X ∈ ℝ) cujo valor oscila simetricamente "
            "ao redor de um centro típico μ, com probabilidades que caem exponencialmente rápido conforme nos afastamos desse centro."
        ),
        primary_parameters="Média (locação) μ ∈ ℝ, Variância σ² > 0 (ou desvio-padrão de escala σ > 0)",
        primary_parameters_latex=r"\mu \in \mathbb{R} \text{ (locação/média)}, \quad \sigma^2 > 0 \text{ (variância)} \text{ ou } \sigma > 0 \text{ (escala)}",
        default_params={"mu": 0.0, "sigma": 1.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right), \quad x \in \mathbb{R}",
        cdf_latex=r"F_X(x; \mu, \sigma^2) = \mathbb{P}(X \le x) = \Phi\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]",
        mean_latex=r"\mathbb{E}[X] = \mu",
        mean_didactic="É o centro de gravidade, o ponto de simetria perfeito e também o pico máximo (moda e mediana) da curva em sino.",
        variance_latex=r"\operatorname{Var}(X) = \sigma^2",
        variance_didactic=(
            "Mede o espalhamento quadrático ao redor de μ. Geometricamente, os pontos de inflexão da curva Normal "
            "(onde a curvatura muda de côncava para convexa) ocorrem exatamente em x = μ - σ e x = μ + σ. "
            "Pela regra empírica: ~68,27% dos dados estão em [μ - σ, μ + σ] e ~95,45% estão em [μ - 2σ, μ + 2σ]."
        ),
        median_latex=r"\operatorname{Mediana}(X) = \mu",
        mode_latex=r"\operatorname{Moda}(X) = \mu",
        skewness_latex=r"\gamma_1 = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^3\right] = 0 \quad (\text{Perfeitamente simétrica})",
        excess_kurtosis_latex=r"\gamma_2 = \frac{\mu_4}{\sigma^4} - 3 = 0 \quad (\text{Mesocúrtica — padrão de referência})",
        raw_kurtosis_latex=r"\beta_2 = \frac{\mathbb{E}[(X-\mu)^4]}{\sigma^4} = 3",
        kurtosis_didactic=(
            "A curtose bruta da Normal vale exatamente 3 (para qualquer μ e σ!). Por isso, na Estatística definimos o "
            "'Excesso de Curtose' como γ₂ = β₂ - 3: a Normal é o marco zero (mesocúrtica). Distribuições com γ₂ > 0 têm "
            "caudas mais pesadas que a Normal (leptocúrticas), e com γ₂ < 0 têm caudas mais leves/achatadas (platicúrticas)."
        ),
        raw_moment_k_latex=(
            r"\mathbb{E}[X^k] = \sigma^k (-i)^k H_k\left(\frac{i\mu}{\sigma}\right) = "
            r"\sum_{j=0}^{\lfloor k/2 \rfloor} \binom{k}{2j} (2j-1)!! \, \sigma^{2j} \mu^{k-2j} "
            r"\quad \Big(\mathbb{E}[X]=\mu, \; \mathbb{E}[X^2]=\mu^2+\sigma^2, \; "
            r"\mathbb{E}[X^3]=\mu^3+3\mu\sigma^2, \; \mathbb{E}[X^4]=\mu^4+6\mu^2\sigma^2+3\sigma^4\Big)"
        ),
        central_moment_k_latex=(
            r"\mu_k = \mathbb{E}[(X - \mu)^k] = \begin{cases} 0 & \text{se } k \text{ é ímpar} \\ "
            r"\sigma^k (k-1)!! = \sigma^k \frac{k!}{2^{k/2}(k/2)!} & \text{se } k \text{ é par} \end{cases}"
        ),
        mgf_latex=r"M_X(t) = \mathbb{E}[e^{tX}] = \exp\left(\mu t + \frac{1}{2}\sigma^2 t^2\right), \quad t \in \mathbb{R}",
        cf_latex=r"\varphi_X(t) = \mathbb{E}[e^{itX}] = \exp\left(i\mu t - \frac{1}{2}\sigma^2 t^2\right)",
        moments_didactic_guide=(
            "Como obter qualquer momento pela Função Geradora de Momentos (FGM)? "
            "Basta derivar M_X(t) = exp(μt + ½σ²t²) em relação a t e avaliar em t = 0: "
            "1ª derivada M'_X(0) = μ (Média); "
            "2ª derivada M''_X(0) = μ² + σ² = E[X²], logo Var(X) = E[X²] - (E[X])² = σ². "
            "Além disso, na Função Geradora de Cumulantes K_X(t) = ln M_X(t) = μt + ½σ²t², todos os cumulantes de ordem k ≥ 3 são ZERO!"
        ),
        entropy_latex=r"h(X) = \frac{1}{2}\ln(2\pi e \sigma^2)",
        fisher_info_latex=r"\mathcal{I}(\mu, \sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Clássica por Média e Variância (Padrão Acadêmico)",
                parameters=r"(\mu, \sigma^2) \in \mathbb{R} \times (0, \infty)",
                canonical_writing_latex=r"X \sim \mathcal{N}(\mu, \sigma^2)",
                pdf_or_pmf_latex=r"f_X(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
                cdf_latex=r"F_X(x; \mu, \sigma^2) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{x-\mu}{\sqrt{2\sigma^2}}\right)\right]",
                mean_latex=r"\mathbb{E}[X] = \mu",
                variance_latex=r"\operatorname{Var}(X) = \sigma^2",
                kurtosis_latex=r"\beta_2 = 3 \quad (\text{Excesso } \gamma_2 = 0)",
                parameter_interpretations={
                    "μ (Média / Parâmetro de Locação)": (
                        "Desloca toda a curva horizontalmente para a esquerda ou direita sem alterar sua forma. "
                        "Tem a mesma unidade de medida da variável X."
                    ),
                    "σ² (Variância / Dispersão Quadrática)": (
                        "Controla o quão espalhada é a distribuição (em unidades de X ao quadrado). "
                        "Quando somamos variáveis Normais independentes, as variâncias σ² se somam diretamente!"
                    ),
                },
                conversion_notes="É a convenção mais usada em livros de Probabilidade, Inferência Estatística e Econometria.",
            ),
            Parameterization(
                name="2. Parametrização por Média e Desvio-Padrão / Escala (Computacional)",
                parameters=r"(\mu, \sigma) \in \mathbb{R} \times (0, \infty)",
                canonical_writing_latex=r"X \sim \mathcal{N}(\mu, \sigma)",
                pdf_or_pmf_latex=r"f_X(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)",
                cdf_latex=r"F_X(x; \mu, \sigma) = \Phi\left(\frac{x-\mu}{\sigma}\right)",
                mean_latex=r"\mathbb{E}[X] = \mu",
                variance_latex=r"\operatorname{Var}(X) = \sigma^2",
                kurtosis_latex=r"\beta_2 = 3 \quad (\text{Excesso } \gamma_2 = 0)",
                parameter_interpretations={
                    "μ (Média / Locação)": "Centro de simetria e valor esperado da variável aleatória X.",
                    "σ (Desvio-Padrão / Escala)": (
                        "Mede o espalhamento na MESMA unidade física de X. Representa a distância horizontal "
                        "entre o pico μ e os pontos de inflexão da curva em sino."
                    ),
                },
                conversion_notes="Padrão em pacotes computacionais como Python SciPy (scipy.stats.norm(loc=μ, scale=σ)) e linguagem R (dnorm(x, mean=μ, sd=σ)).",
            ),
            Parameterization(
                name="3. Parametrização por Média e Precisão (Estatística Bayesiana)",
                parameters=r"(\mu, \tau) \text{ onde } \tau = \frac{1}{\sigma^2} > 0",
                canonical_writing_latex=r"X \sim \mathcal{N}_{\text{prec}}(\mu, \tau)",
                pdf_or_pmf_latex=r"f_X(x; \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left(-\frac{\tau}{2}(x-\mu)^2\right)",
                cdf_latex=r"F_X(x; \mu, \tau) = \Phi\left(\sqrt{\tau}(x-\mu)\right)",
                mean_latex=r"\mathbb{E}[X] = \mu",
                variance_latex=r"\operatorname{Var}(X) = \frac{1}{\tau}",
                kurtosis_latex=r"\beta_2 = 3 \quad (\text{Excesso } \gamma_2 = 0)",
                parameter_interpretations={
                    "μ (Média)": "Centro de locação da distribuição.",
                    "τ = 1/σ² (Precisão / Informação)": (
                        "Mede o grau de certeza ou concentração dos dados ao redor de μ. "
                        "Quanto MAIOR a precisão τ, mais estreito e alto é o pico da curva; quando τ → 0, a incerteza é máxima."
                    ),
                },
                conversion_notes="Muito usada em Inferência Bayesiana (PyMC, Stan, JAGS), pois a distribuição a priori conjugada para a precisão τ é simplesmente uma distribuição Gama.",
            ),
            Parameterization(
                name="4. Parametrização Canônica da Família Exponencial Natural",
                parameters=r"(\eta_1, \eta_2) = \left(\frac{\mu}{\sigma^2}, \, -\frac{1}{2\sigma^2}\right) \text{ com } \eta_2 < 0",
                canonical_writing_latex=r"X \sim \operatorname{EF-Normal}(\eta_1, \eta_2)",
                pdf_or_pmf_latex=r"f_X(x; \eta_1, \eta_2) = \sqrt{\frac{-\eta_2}{\pi}} \exp\left(\eta_1 x + \eta_2 x^2 + \frac{\eta_1^2}{4\eta_2}\right)",
                cdf_latex=r"F_X(x; \eta_1, \eta_2) = \Phi\left(\sqrt{-2\eta_2}\left(x + \frac{\eta_1}{2\eta_2}\right)\right)",
                mean_latex=r"\mathbb{E}[X] = -\frac{\eta_1}{2\eta_2}",
                variance_latex=r"\operatorname{Var}(X) = -\frac{1}{2\eta_2}",
                kurtosis_latex=r"\beta_2 = 3 \quad (\text{Excesso } \gamma_2 = 0)",
                parameter_interpretations={
                    "η₁ = μ/σ² (Primeiro Parâmetro Natural)": "Peso linear acoplado à estatística suficiente T₁(x) = x (razão sinal-ruído escalada).",
                    "η₂ = -1/(2σ²) (Segundo Parâmetro Natural)": "Peso quadrático negativo acoplado à estatística suficiente T₂(x) = x², garantindo que a integral da densidade convirja.",
                },
                conversion_notes="Fundamental em Modelos Lineares Generalizados (MLG/GLM) e MachineProbabilistic Graphical Models.",
            ),
        ],
        iid_and_statistical_properties={
            "Fechamento Aditivo sob Independência": r"Se $X_1, \dots, X_n$ são independentes com $X_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$, qualquer combinação linear permanece Normal: $\sum_{i=1}^n a_i X_i \sim \mathcal{N}\left(\sum_{i=1}^n a_i \mu_i, \, \sum_{i=1}^n a_i^2 \sigma_i^2\right)$.",
            "Amostra i.i.d. e Teorema de Cochran / Lukacs": r"Se $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$, então a média amostral $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$ e a variância amostral escalada $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ são **estocasticamente independentes** ($\bar{X} \perp\!\!\!\perp S^2$). A Normal é a ÚNICA distribuição em que média e variância amostrais são independentes!",
            "Estatística Suficiente e Completa": r"Para uma amostra i.i.d., o par $T(X_1, \dots, X_n) = \left(\sum_{i=1}^n X_i, \sum_{i=1}^n X_i^2\right)$ resume 100% da informação amostral sobre $(\mu, \sigma^2)$.",
            "Incorrelação Equivalente à Independência": r"Se $(X, Y)$ possuem distribuição Normal Bivariada conjunta, então $\operatorname{Cov}(X, Y) = 0 \iff X \perp\!\!\!\perp Y$ (propriedade especial da família Gaussiana).",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": True,
            "unimodal": True,
            "skewness_target": 0.0,
            "excess_kurtosis_target": 0.0,
            "support_type": "real_line",
        },
    ),

    "exponential": DistributionSpec(
        id="exponential",
        name="Distribuição Exponencial",
        aliases=["exponential", "exponencial", "exp", "negative exponential"],
        notation_patterns=[
            r"Exp\s*\(",
            r"Exponencial\s*\(",
            r"Exponential\s*\(",
            r"\lambda\s*e\^\{-\lambda",
        ],
        formula_cues=["lambda*exp(-lambda*x)", "\\lambda e^{-\\lambda x}", "1/beta", "e^{-x/\\beta}"],
        family="Contínua",
        support="x in [0, +infinito)",
        support_latex=r"x \in [0, +\infty)",
        canonical_single_rv_latex=r"X \sim \operatorname{Exp}(\lambda)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Exp}(\lambda)",
        didactic_overview=(
            "A Distribuição Exponencial modela o **tempo de espera contínuo** até a ocorrência do próximo evento "
            "em um Processo de Poisson homogêneo (como o tempo até um componente eletrônico falhar, o tempo até o "
            "próximo cliente chegar a uma fila, ou o tempo até o decaimento de um átomo radioativo)."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua X ≥ 0 representa o tempo (ou distância) decorrido até que um evento ocorra."
        ),
        primary_parameters="Taxa de ocorrência λ > 0 (ou escala/tempo médio β = 1/λ > 0)",
        primary_parameters_latex=r"\lambda > 0 \text{ (taxa)} \quad \text{ou} \quad \beta = \frac{1}{\lambda} > 0 \text{ (escala/média)}",
        default_params={"lam": 1.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; \lambda) = \lambda e^{-\lambda x} \, \mathbb{I}(x \ge 0)",
        cdf_latex=r"F_X(x; \lambda) = \begin{cases} 1 - e^{-\lambda x} & \text{se } x \ge 0 \\ 0 & \text{se } x < 0 \end{cases}",
        mean_latex=r"\mathbb{E}[X] = \frac{1}{\lambda} = \beta",
        mean_didactic="Se eventos acontecem a uma taxa de λ por hora, o tempo médio de espera entre eventos é exatamente 1/λ horas.",
        variance_latex=r"\operatorname{Var}(X) = \frac{1}{\lambda^2} = \beta^2",
        variance_didactic="Na Exponencial, o desvio-padrão √Var(X) = 1/λ é sempre igual à própria média E[X] = 1/λ (Coeficiente de Variação CV = 1).",
        median_latex=r"\operatorname{Mediana}(X) = \frac{\ln 2}{\lambda} \approx \frac{0.6931}{\lambda} < \mathbb{E}[X]",
        mode_latex=r"\operatorname{Moda}(X) = 0",
        skewness_latex=r"\gamma_1 = 2 \quad (\text{Forte assimetria positiva à direita})",
        excess_kurtosis_latex=r"\gamma_2 = 6 \quad (\text{Leptocúrtica — cauda direita alongada})",
        raw_kurtosis_latex=r"\beta_2 = \frac{\mathbb{E}[(X-\mu)^4]}{\sigma^4} = 9",
        kurtosis_didactic=(
            "A curtose da Exponencial é constante (β₂ = 9, excesso γ₂ = 6) e NÃO depende de λ! "
            "Isso ocorre porque λ é apenas um fator de escala temporal que estica ou comprime o eixo X sem mudar o formato geométrico da curva."
        ),
        raw_moment_k_latex=r"\mathbb{E}[X^k] = \frac{k!}{\lambda^k} = k! \, \beta^k = \frac{\Gamma(k+1)}{\lambda^k} \quad (k > -1)",
        central_moment_k_latex=r"\mu_k = \mathbb{E}\left[\left(X - \frac{1}{\lambda}\right)^k\right] = \frac{!k}{\lambda^k} = \frac{k!}{\lambda^k} \sum_{j=0}^k \frac{(-1)^j}{j!} \quad (\text{onde } !k \text{ é o subfatorial})",
        mgf_latex=r"M_X(t) = \mathbb{E}[e^{tX}] = \frac{\lambda}{\lambda - t} = (1 - \beta t)^{-1}, \quad \text{para } t < \lambda",
        cf_latex=r"\varphi_X(t) = \mathbb{E}[e^{itX}] = \frac{\lambda}{\lambda - it}",
        moments_didactic_guide=(
            "Expandindo a FGM em série geométrica: M_X(t) = (1 - t/λ)⁻¹ = ∑_{k=0}^∞ (t/λ)^k = ∑_{k=0}^∞ (k! / λ^k) · (t^k / k!). "
            "Comparando com a série de Taylor ∑ E[X^k] (t^k / k!), obtemos imediatamente a fórmula geral de momentos: E[X^k] = k! / λ^k."
        ),
        entropy_latex=r"h(X) = 1 - \ln(\lambda) = 1 + \ln(\beta)",
        fisher_info_latex=r"\mathcal{I}(\lambda) = \frac{1}{\lambda^2}, \quad \mathcal{I}(\beta) = \frac{1}{\beta^2}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização por Taxa λ (Padrão em Processos Estocásticos)",
                parameters=r"\lambda > 0 \text{ (eventos por unidade de tempo)}",
                canonical_writing_latex=r"X \sim \operatorname{Exp}(\lambda)",
                pdf_or_pmf_latex=r"f_X(x; \lambda) = \lambda e^{-\lambda x}, \quad x \ge 0",
                cdf_latex=r"F_X(x; \lambda) = 1 - e^{-\lambda x}, \quad x \ge 0",
                mean_latex=r"\mathbb{E}[X] = \frac{1}{\lambda}",
                variance_latex=r"\operatorname{Var}(X) = \frac{1}{\lambda^2}",
                kurtosis_latex=r"\beta_2 = 9 \quad (\text{Excesso } \gamma_2 = 6)",
                parameter_interpretations={
                    "λ (Taxa de Falha / Intensidade de Chegada)": (
                        "Representa a velocidade média com que os eventos acontecem por unidade de tempo. "
                        "Quanto MAIOR for λ, mais rápido o evento ocorre e menor é o tempo de espera médio (1/λ)."
                    ),
                },
                conversion_notes="Conecta diretamente a Exponencial ao Processo de Poisson de taxa λ.",
            ),
            Parameterization(
                name="2. Parametrização por Escala / Tempo Médio β (Engenharia de Confiabilidade e SciPy)",
                parameters=r"\beta > 0 \text{ onde } \beta = \frac{1}{\lambda}",
                canonical_writing_latex=r"X \sim \operatorname{Exp}(\text{escala}=\beta)",
                pdf_or_pmf_latex=r"f_X(x; \beta) = \frac{1}{\beta} e^{-x/\beta}, \quad x \ge 0",
                cdf_latex=r"F_X(x; \beta) = 1 - e^{-x/\beta}, \quad x \ge 0",
                mean_latex=r"\mathbb{E}[X] = \beta",
                variance_latex=r"\operatorname{Var}(X) = \beta^2",
                kurtosis_latex=r"\beta_2 = 9 \quad (\text{Excesso } \gamma_2 = 6)",
                parameter_interpretations={
                    "β = 1/λ (MTTF / Tempo Médio Entre Falhas)": (
                        "É diretamente o tempo médio esperado de vida ou espera na mesma unidade de tempo de X (ex.: horas, dias). "
                        "Quanto MAIOR β, mais longa é a vida útil média do sistema."
                    ),
                },
                conversion_notes="Usada no Python SciPy (scipy.stats.expon(scale=β)) e em Análise de Sobrevivência.",
            ),
            Parameterization(
                name="3. Parametrização Deslocada de 2 Parâmetros (Garantia / Limiar Mínimo θ)",
                parameters=r"(\lambda, \theta) \text{ com } \lambda > 0, \, x \ge \theta",
                canonical_writing_latex=r"X \sim \operatorname{Exp}(\lambda, \theta)",
                pdf_or_pmf_latex=r"f_X(x; \lambda, \theta) = \lambda e^{-\lambda(x - \theta)}, \quad x \ge \theta",
                cdf_latex=r"F_X(x; \lambda, \theta) = 1 - e^{-\lambda(x - \theta)}, \quad x \ge \theta",
                mean_latex=r"\mathbb{E}[X] = \theta + \frac{1}{\lambda}",
                variance_latex=r"\operatorname{Var}(X) = \frac{1}{\lambda^2}",
                kurtosis_latex=r"\beta_2 = 9 \quad (\text{Excesso } \gamma_2 = 6)",
                parameter_interpretations={
                    "θ (Limiar / Tempo de Garantia Mínimo)": "Instante mínimo a partir do qual o evento pode começar a ocorrer (X ≥ θ).",
                    "λ (Taxa após o limiar)": "Taxa constante de ocorrência depois que o tempo ultrapassa θ.",
                },
                conversion_notes="Muito útil quando existe um tempo morto ou período mínimo de incubação/garantia θ.",
            ),
        ],
        iid_and_statistical_properties={
            "Propriedade de Perda de Memória (Memoryless)": r"$\mathbb{P}(X > s + t \mid X > s) = \mathbb{P}(X > t)$ para quaisquer $s, t \ge 0$. A Exponencial é a ÚNICA distribuição contínua sem memória: saber que o componente já durou $s$ horas não altera a distribuição do tempo restante!",
            "Soma de Amostra i.i.d. (Conexão com Gama / Erlang)": r"Se $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Exp}(\lambda)$, então o tempo total até o $n$-ésimo evento é $S_n = \sum_{i=1}^n X_i \sim \operatorname{Gama}(n, \lambda) = \operatorname{Erlang}(n, \lambda)$.",
            "Mínimo de Exponenciais Independentes": r"Se $X_i \sim \operatorname{Exp}(\lambda_i)$ são independentes, o tempo até o primeiro evento acontecer é $\min(X_1, \dots, X_n) \sim \operatorname{Exp}\left(\sum_{i=1}^n \lambda_i\right)$.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": False,
            "unimodal": True,
            "monotone_decreasing": True,
            "skewness_target": 2.0,
            "excess_kurtosis_target": 6.0,
            "support_type": "non_negative",
        },
    ),

    "uniform": DistributionSpec(
        id="uniform",
        name="Distribuição Uniforme Contínua (Retangular)",
        aliases=["uniform", "uniforme", "rectangular", "retangular", "unif"],
        notation_patterns=[
            r"\bU\s*\(\s*[^,]+,\s*[^)]+\)",
            r"\bUnif\s*\(",
            r"\bUniforme?\s*\(",
            r"\mathcal\{U\}",
        ],
        formula_cues=["1/(b-a)", "\\frac{1}{b-a}", "(b-a)^2/12", "12"],
        family="Contínua",
        support="x in [a, b]",
        support_latex=r"x \in [a, b] \quad (-\infty < a < b < +\infty)",
        canonical_single_rv_latex=r"X \sim \operatorname{Uniforme}(a, b) \quad \text{ou} \quad X \sim \mathcal{U}(a, b)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}(a, b)",
        didactic_overview=(
            "A Distribuição Uniforme Contínua modela a incerteza completa dentro de um intervalo fechado [a, b]: "
            "qualquer subintervalo de mesmo comprimento dentro de [a, b] tem exatamente a mesma probabilidade de conter X. "
            "É a base para a geração de números pseudoaleatórios em computadores (quando a=0 e b=1)."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua X representa um valor sorteado de maneira perfeitamente equiprovável "
            "entre um limite mínimo a e um limite máximo b."
        ),
        primary_parameters="Limite inferior a ∈ ℝ, Limite superior b > a",
        primary_parameters_latex=r"a \in \mathbb{R} \text{ (mínimo)}, \quad b > a \text{ (máximo)}",
        default_params={"a": 0.0, "b": 1.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; a, b) = \begin{cases} \frac{1}{b - a} & \text{se } a \le x \le b \\ 0 & \text{caso contrário} \end{cases}",
        cdf_latex=r"F_X(x; a, b) = \begin{cases} 0 & \text{se } x < a \\ \frac{x - a}{b - a} & \text{se } a \le x \le b \\ 1 & \text{se } x > b \end{cases}",
        mean_latex=r"\mathbb{E}[X] = \frac{a + b}{2}",
        mean_didactic="É exatamente o ponto médio geométrico do intervalo [a, b].",
        variance_latex=r"\operatorname{Var}(X) = \frac{(b - a)^2}{12}",
        variance_didactic="Depende apenas da largura total L = (b - a) do intervalo. O fator 12 no denominador surge da integração de (x - μ)² no intervalo.",
        median_latex=r"\operatorname{Mediana}(X) = \frac{a + b}{2}",
        mode_latex=r"\operatorname{Moda}(X) = \text{Qualquer valor em } [a, b] \text{ (platô constante)}",
        skewness_latex=r"\gamma_1 = 0 \quad (\text{Simétrica no intervalo } [a, b])",
        excess_kurtosis_latex=r"\gamma_2 = -\frac{6}{5} = -1.2 \quad (\text{Platicúrtica — topo reto e sem caudas})",
        raw_kurtosis_latex=r"\beta_2 = \frac{9}{5} = 1.8",
        kurtosis_didactic=(
            "Como a distribuição Uniforme é uma caixa retangular perfeita sem caudas além de [a, b], "
            "sua curtose bruta é 1,8 (menor que 3 da Normal), resultando em um excesso de curtose negativo γ₂ = -1,2 (platicúrtica)."
        ),
        raw_moment_k_latex=r"\mathbb{E}[X^k] = \frac{b^{k+1} - a^{k+1}}{(k + 1)(b - a)} = \frac{1}{k+1} \sum_{j=0}^k a^j b^{k-j}",
        central_moment_k_latex=r"\mu_k = \mathbb{E}[(X - \mu)^k] = \begin{cases} 0 & \text{se } k \text{ é ímpar} \\ \frac{(b - a)^k}{2^k (k + 1)} & \text{se } k \text{ é par} \end{cases}",
        mgf_latex=r"M_X(t) = \begin{cases} \frac{e^{tb} - e^{ta}}{t(b - a)} & \text{se } t \neq 0 \\ 1 & \text{se } t = 0 \end{cases}",
        cf_latex=r"\varphi_X(t) = \frac{e^{itb} - e^{ita}}{it(b - a)}",
        moments_didactic_guide=(
            "Para calcular o k-ésimo momento bruto E[X^k], integramos x^k · 1/(b-a) de a até b: "
            "a primitiva de x^k é x^(k+1)/(k+1), gerando diretamente (b^(k+1) - a^(k+1)) / ((k+1)(b - a))."
        ),
        entropy_latex=r"h(X) = \ln(b - a)",
        fisher_info_latex=r"\text{Família não-regular (o suporte } [a,b] \text{ depende dos parâmetros).}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização pelos Extremos do Intervalo [a, b] (Padrão)",
                parameters=r"(a, b) \text{ com } a < b",
                canonical_writing_latex=r"X \sim \mathcal{U}(a, b)",
                pdf_or_pmf_latex=r"f_X(x; a, b) = \frac{1}{b - a} \, \mathbb{I}_{[a,b]}(x)",
                cdf_latex=r"F_X(x; a, b) = \frac{x - a}{b - a} \text{ para } x \in [a, b]",
                mean_latex=r"\frac{a + b}{2}",
                variance_latex=r"\frac{(b - a)^2}{12}",
                kurtosis_latex=r"\beta_2 = 1.8 \quad (\gamma_2 = -1.2)",
                parameter_interpretations={
                    "a (Limite Inferior / Mínimo)": "Menor valor possível que a variável aleatória X pode assumir.",
                    "b (Limite Superior / Máximo)": "Maior valor possível que a variável aleatória X pode assumir. A altura da densidade é 1/(b-a).",
                },
                conversion_notes="Padrão em toda a literatura matemática e no R (dunif(x, min=a, max=b)).",
            ),
            Parameterization(
                name="2. Parametrização de Locação e Escala / Largura (SciPy)",
                parameters=r"(a, L) \text{ onde } L = b - a > 0 \text{ é a largura}",
                canonical_writing_latex=r"X \sim \mathcal{U}(\text{loc}=a, \text{scale}=L)",
                pdf_or_pmf_latex=r"f_X(x; a, L) = \frac{1}{L} \, \mathbb{I}_{[a, \, a+L]}(x)",
                cdf_latex=r"F_X(x; a, L) = \frac{x - a}{L} \text{ para } x \in [a, a+L]",
                mean_latex=r"a + \frac{L}{2}",
                variance_latex=r"\frac{L^2}{12}",
                kurtosis_latex=r"\beta_2 = 1.8 \quad (\gamma_2 = -1.2)",
                parameter_interpretations={
                    "a (Ponto Inicial / Locação)": "Início do intervalo de suporte.",
                    "L = b - a (Amplitude / Escala)": "Comprimento total do intervalo onde X pode cair.",
                },
                conversion_notes="Atenção no Python SciPy: scipy.stats.uniform(loc=a, scale=L) representa o intervalo [a, a+L]!",
            ),
            Parameterization(
                name="3. Parametrização por Ponto Médio e Meia-Largura (Metrologia e Erros)",
                parameters=r"(\mu, h) \text{ onde } a = \mu - h, \, b = \mu + h",
                canonical_writing_latex=r"X \sim \mathcal{U}(\mu - h, \, \mu + h)",
                pdf_or_pmf_latex=r"f_X(x; \mu, h) = \frac{1}{2h} \, \mathbb{I}_{[\mu-h, \, \mu+h]}(x)",
                cdf_latex=r"F_X(x; \mu, h) = \frac{x - \mu + h}{2h} \text{ para } x \in [\mu-h, \mu+h]",
                mean_latex=r"\mu",
                variance_latex=r"\frac{h^2}{3}",
                kurtosis_latex=r"\beta_2 = 1.8 \quad (\gamma_2 = -1.2)",
                parameter_interpretations={
                    "μ (Valor Nominal Central)": "Valor central da medição.",
                    "h (Tolerância / Erro Máximo ±h)": "Margem máxima de erro simétrico para mais ou para menos (ex.: erro de arredondamento ou quantização).",
                },
                conversion_notes="Muito usada em Engenharia de Controle, Processamento Digital de Sinais e Metrologia (Incerteza Tipo B).",
            ),
        ],
        iid_and_statistical_properties={
            "Transformação Integral de Probabilidade (Gerador Universal)": r"Se $U \sim \mathcal{U}(0, 1)$ e $F$ é qualquer FDA contínua invertível, então $X = F^{-1}(U)$ possui exatamente distribuição $F$!",
            "Estatísticas de Ordem i.i.d. (Conexão com a Beta)": r"Para uma amostra $U_1, \dots, U_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}(0, 1)$, a $k$-ésima estatística de ordem segue $U_{(k)} \sim \operatorname{Beta}(k, n - k + 1)$.",
            "Estatística Suficiente": r"Para $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}(a, b)$, o par formado pelo mínimo e máximo amostral $T = (\min_i X_i, \max_i X_i)$ é conjuntamente suficiente.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": True,
            "unimodal": False,
            "flat_top": True,
            "skewness_target": 0.0,
            "excess_kurtosis_target": -1.2,
            "support_type": "bounded",
        },
    ),

    "gamma": DistributionSpec(
        id="gamma",
        name="Distribuição Gama (incluindo Erlang)",
        aliases=["gamma", "gama", "erlang"],
        notation_patterns=[
            r"Gamma\s*\(",
            r"Gama\s*\(",
            r"\Gamma\s*\(\s*[^,]+,\s*[^)]+\)",
            r"Erlang\s*\(",
        ],
        formula_cues=["x^(alpha-1)", "x^{k-1}", "\\Gamma(\\alpha)", "\\Gamma(k)", "beta^alpha"],
        family="Contínua",
        support="x in (0, +infinito)",
        support_latex=r"x \in (0, +\infty)",
        canonical_single_rv_latex=r"X \sim \operatorname{Gama}(\alpha, \beta)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Gama}(\alpha, \beta)",
        didactic_overview=(
            "A Distribuição Gama generaliza a Exponencial: enquanto a Exponencial mede o tempo até o 1º evento, "
            "a Gama(α, β) mede o **tempo total de espera até que ocorram α eventos** independentes a uma taxa β. "
            "É amplamente usada para modelar tempos de vida, volume de chuvas, custos de sinistros de seguros e inferência Bayesiana."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua positiva X > 0 representa o tempo acumulado para α ocorrências "
            "ou uma grandeza contínua estritamente positiva e assimétrica à direita."
        ),
        primary_parameters="Forma α (ou k) > 0, Taxa β > 0 (ou Escala θ = 1/β > 0)",
        primary_parameters_latex=r"\alpha > 0 \text{ (forma)}, \quad \beta > 0 \text{ (taxa)} \text{ ou } \theta = \frac{1}{\beta} > 0 \text{ (escala)}",
        default_params={"alpha": 3.0, "beta": 1.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0",
        cdf_latex=r"F_X(x; \alpha, \beta) = \frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)} = P(\alpha, \beta x) \quad (\text{função gama incompleta regularizada})",
        mean_latex=r"\mathbb{E}[X] = \frac{\alpha}{\beta} = k\theta",
        mean_didactic="Como cada um dos α sub-eventos demora em média 1/β, a soma de α etapas tem média α/β.",
        variance_latex=r"\operatorname{Var}(X) = \frac{\alpha}{\beta^2} = k\theta^2",
        variance_didactic="Pela independência das α etapas Exponenciais, somamos α vezes a variância 1/β², obtendo α/β².",
        median_latex=r"\text{Sem forma fechada simples; aproximada por } \frac{1}{\beta}\left(\alpha - \frac{1}{3} + \frac{0.02}{\alpha}\right) \text{ para } \alpha \ge 1",
        mode_latex=r"\operatorname{Moda}(X) = \begin{cases} \frac{\alpha - 1}{\beta} = (k-1)\theta & \text{se } \alpha \ge 1 \\ 0 & \text{se } 0 < \alpha < 1 \end{cases}",
        skewness_latex=r"\gamma_1 = \frac{2}{\sqrt{\alpha}} \quad (\text{Diminui conforme } \alpha \text{ cresce})",
        excess_kurtosis_latex=r"\gamma_2 = \frac{6}{\alpha}",
        raw_kurtosis_latex=r"\beta_2 = 3 + \frac{6}{\alpha}",
        kurtosis_didactic=(
            "Repare que o excesso de curtose γ₂ = 6/α depende EXCLUSIVAMENTE do parâmetro de forma α! "
            "Quando α = 1 (Exponencial), γ₂ = 6. Quando α → ∞, γ₂ → 0 e a Gama converge para uma Normal pelo Teorema Central do Limite."
        ),
        raw_moment_k_latex=r"\mathbb{E}[X^m] = \frac{\Gamma(\alpha + m)}{\beta^m \Gamma(\alpha)} = \frac{\alpha(\alpha+1)\cdots(\alpha+m-1)}{\beta^m}",
        central_moment_k_latex=r"\mu_2 = \frac{\alpha}{\beta^2}, \quad \mu_3 = \frac{2\alpha}{\beta^3}, \quad \mu_4 = \frac{3\alpha(\alpha+2)}{\beta^4}",
        mgf_latex=r"M_X(t) = \left(1 - \frac{t}{\beta}\right)^{-\alpha} = (1 - \theta t)^{-k}, \quad \text{para } t < \beta",
        cf_latex=r"\varphi_X(t) = \left(1 - \frac{it}{\beta}\right)^{-\alpha}",
        moments_didactic_guide=(
            "Para calcular E[X^m] direto pela integral: ∫₀^∞ x^m (β^α / Γ(α)) x^(α-1) e^(-βx) dx. "
            "Juntando as potências de x em x^((α+m)-1), reconhecemos o núcleo de uma Gama(α+m, β), resultando em Γ(α+m) / (β^m Γ(α))."
        ),
        entropy_latex=r"h(X) = \alpha - \ln\beta + \ln\Gamma(\alpha) + (1 - \alpha)\psi(\alpha)",
        fisher_info_latex=r"\mathcal{I}(\alpha, \beta) = \begin{pmatrix} \psi_1(\alpha) & -1/\beta \\ -1/\beta & \alpha/\beta^2 \end{pmatrix}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização por Forma α e Taxa β (Estatística Matemática e Bayesiana)",
                parameters=r"(\alpha, \beta) > 0",
                canonical_writing_latex=r"X \sim \operatorname{Gama}(\alpha, \beta)",
                pdf_or_pmf_latex=r"f_X(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0",
                cdf_latex=r"F_X(x; \alpha, \beta) = \frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)}",
                mean_latex=r"\frac{\alpha}{\beta}",
                variance_latex=r"\frac{\alpha}{\beta^2}",
                kurtosis_latex=r"\beta_2 = 3 + \frac{6}{\alpha} \quad \left(\gamma_2 = \frac{6}{\alpha}\right)",
                parameter_interpretations={
                    "α (Parâmetro de Forma / Número de Eventos)": (
                        "Governa a silhueta da curva (assimetria e curtose). Se α inteiro, é o número de eventos Exponenciais acumulados (distribuição de Erlang)."
                    ),
                    "β (Parâmetro de Taxa / Rate)": (
                        "Velocidade de ocorrência por unidade de X. Quanto maior β, mais concentrada próxima de zero fica a distribuição."
                    ),
                },
                conversion_notes="É a parametrização mais comum em Inferência Bayesiana (priori conjugada para a taxa λ da Poisson e para a precisão τ da Normal).",
            ),
            Parameterization(
                name="2. Parametrização por Forma k e Escala θ (SciPy e Econometria)",
                parameters=r"(k, \theta) > 0 \text{ onde } k = \alpha \text{ e } \theta = \frac{1}{\beta}",
                canonical_writing_latex=r"X \sim \operatorname{Gama}(k, \theta)",
                pdf_or_pmf_latex=r"f_X(x; k, \theta) = \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-x/\theta}, \quad x > 0",
                cdf_latex=r"F_X(x; k, \theta) = \frac{\gamma(k, x/\theta)}{\Gamma(k)}",
                mean_latex=r"k\theta",
                variance_latex=r"k\theta^2",
                kurtosis_latex=r"\beta_2 = 3 + \frac{6}{k} \quad \left(\gamma_2 = \frac{6}{k}\right)",
                parameter_interpretations={
                    "k (Forma)": "Quantidade efetiva de etapas ou grau de simetria da curva.",
                    "θ = 1/β (Escala)": "Duração ou tamanho médio de cada uma das k etapas individuais, na mesma unidade de X.",
                },
                conversion_notes="Padrão em Python SciPy (scipy.stats.gamma(a=k, scale=θ)).",
            ),
            Parameterization(
                name="3. Parametrização por Média μ e Forma α (Modelos Lineares Generalizados — GLM)",
                parameters=r"(\mu, \alpha) > 0 \text{ onde } \mu = \frac{\alpha}{\beta}",
                canonical_writing_latex=r"X \sim \operatorname{Gama}_{\text{GLM}}(\mu, \alpha)",
                pdf_or_pmf_latex=r"f_X(x; \mu, \alpha) = \frac{1}{\Gamma(\alpha)}\left(\frac{\alpha}{\mu}\right)^\alpha x^{\alpha-1} e^{-\alpha x / \mu}, \quad x > 0",
                cdf_latex=r"F_X(x; \mu, \alpha) = \frac{\gamma(\alpha, \alpha x / \mu)}{\Gamma(\alpha)}",
                mean_latex=r"\mu",
                variance_latex=r"\frac{\mu^2}{\alpha}",
                kurtosis_latex=r"\beta_2 = 3 + \frac{6}{\alpha}",
                parameter_interpretations={
                    "μ (Valor Esperado / Média)": "Média direta da resposta positiva modelada na regressão.",
                    "1/α = φ (Parâmetro de Dispersão)": "Quadrado do Coeficiente de Variação (CV² = Var(X)/μ² = 1/α).",
                },
                conversion_notes="Usada em Regressão Gama (GLM), onde a variância cresce quadraticamente com a média.",
            ),
        ],
        iid_and_statistical_properties={
            "Fechamento Aditivo (Mesma Taxa β)": r"Se $X_i \sim \operatorname{Gama}(\alpha_i, \beta)$ são independentes com a MESMA taxa $\beta$, então $\sum_{i=1}^n X_i \sim \operatorname{Gama}\left(\sum_{i=1}^n \alpha_i, \, \beta\right)$.",
            "Amostra i.i.d. de Exponenciais": r"Se $Y_1, \dots, Y_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Exp}(\beta)$, então $\sum_{i=1}^n Y_i \sim \operatorname{Gama}(n, \beta)$.",
            "Casos Especiais Famosos": r"$\operatorname{Gama}(1, \lambda) = \operatorname{Exp}(\lambda)$ e $\operatorname{Gama}(\nu/2, 1/2) = \chi^2_\nu$ (Qui-Quadrado com $\nu$ graus de liberdade).",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": False,
            "unimodal": True,
            "skewness_target": 1.15,
            "excess_kurtosis_target": 2.0,
            "support_type": "non_negative",
        },
    ),

    "beta": DistributionSpec(
        id="beta",
        name="Distribuição Beta",
        aliases=["beta", "distribuicao beta"],
        notation_patterns=[
            r"Beta\s*\(\s*[^,]+,\s*[^)]+\)",
            r"\operatorname\{Beta\}",
        ],
        formula_cues=["x^(alpha-1)*(1-x)^(beta-1)", "x^{\\alpha-1}(1-x)^{\\beta-1}", "B(\\alpha,\\beta)"],
        family="Contínua",
        support="x in [0, 1]",
        support_latex=r"x \in [0, 1]",
        canonical_single_rv_latex=r"X \sim \operatorname{Beta}(\alpha, \beta)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Beta}(\alpha, \beta)",
        didactic_overview=(
            "A Distribuição Beta é a distribuição natural para modelar **proporções, taxas, porcentagens ou probabilidades** "
            "restritas ao intervalo [0, 1]. Na Estatística Bayesiana, é a distribuição a priori conjugada para a "
            "probabilidade de sucesso p de ensaios de Bernoulli/Binomial."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua X ∈ [0, 1] representa uma fração, proporção ou probabilidade desconhecida."
        ),
        primary_parameters="Parâmetros de forma α > 0 e β > 0",
        primary_parameters_latex=r"\alpha > 0 \text{ (forma 1 / pseudo-sucessos)}, \quad \beta > 0 \text{ (forma 2 / pseudo-fracassos)}",
        default_params={"alpha": 2.0, "beta": 5.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; \alpha, \beta) = \frac{x^{\alpha - 1}(1 - x)^{\beta - 1}}{\mathrm{B}(\alpha, \beta)}, \quad x \in [0, 1], \quad \text{onde } \mathrm{B}(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}",
        cdf_latex=r"F_X(x; \alpha, \beta) = I_x(\alpha, \beta) = \frac{\mathrm{B}(x; \alpha, \beta)}{\mathrm{B}(\alpha, \beta)} \quad (\text{função beta incompleta regularizada})",
        mean_latex=r"\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}",
        mean_didactic="É a proporção direta do peso α sobre a soma total de pesos (α + β).",
        variance_latex=r"\operatorname{Var}(X) = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}",
        variance_didactic="Quanto maior a soma (α + β), menor a variância e mais estreita fica a distribuição em torno da média α/(α+β).",
        median_latex=r"I_{1/2}^{[-1]}(\alpha, \beta) \approx \frac{\alpha - \frac{1}{3}}{\alpha + \beta - \frac{2}{3}} \quad (\text{para } \alpha, \beta > 1)",
        mode_latex=r"\operatorname{Moda}(X) = \frac{\alpha - 1}{\alpha + \beta - 2} \quad (\text{para } \alpha, \beta > 1)",
        skewness_latex=r"\gamma_1 = \frac{2(\beta - \alpha)\sqrt{\alpha + \beta + 1}}{(\alpha + \beta + 2)\sqrt{\alpha\beta}}",
        excess_kurtosis_latex=r"\gamma_2 = \frac{6\left[(\alpha - \beta)^2(\alpha + \beta + 1) - \alpha\beta(\alpha + \beta + 2)\right]}{\alpha\beta(\alpha + \beta + 2)(\alpha + \beta + 3)}",
        raw_kurtosis_latex=r"\beta_2 = 3 + \gamma_2",
        kurtosis_didactic="A Beta é extremamente camaleônica: pode ter forma de U (α,β < 1), reta uniforme (α=β=1), ou sino (α,β > 1).",
        raw_moment_k_latex=r"\mathbb{E}[X^k] = \frac{\mathrm{B}(\alpha + k, \beta)}{\mathrm{B}(\alpha, \beta)} = \prod_{r=0}^{k-1} \frac{\alpha + r}{\alpha + \beta + r}",
        central_moment_k_latex=r"\mu_2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}",
        mgf_latex=r"M_X(t) = {}_1F_1(\alpha; \alpha + \beta; t) = 1 + \sum_{k=1}^{\infty} \left(\prod_{r=0}^{k-1} \frac{\alpha + r}{\alpha + \beta + r}\right) \frac{t^k}{k!}",
        cf_latex=r"\varphi_X(t) = {}_1F_1(\alpha; \alpha + \beta; it)",
        moments_didactic_guide=(
            "Para achar E[X^k], integramos x^(α+k-1) (1-x)^(β-1) / B(α,β) em [0,1]. "
            "O numerador é exatamente a função Beta B(α+k, β)! Usando Γ(z+1) = zΓ(z), simplificamos para o produtório (α+r)/(α+β+r)."
        ),
        entropy_latex=r"h(X) = \ln\mathrm{B}(\alpha, \beta) - (\alpha - 1)\psi(\alpha) - (\beta - 1)\psi(\beta) + (\alpha + \beta - 2)\psi(\alpha + \beta)",
        fisher_info_latex=r"\mathcal{I}(\alpha, \beta) = \begin{pmatrix} \psi_1(\alpha) - \psi_1(\alpha+\beta) & -\psi_1(\alpha+\beta) \\ -\psi_1(\alpha+\beta) & \psi_1(\beta) - \psi_1(\alpha+\beta) \end{pmatrix}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Padrão por Pesos de Forma (α, β) / Pseudo-Contagens",
                parameters=r"(\alpha, \beta) > 0",
                canonical_writing_latex=r"X \sim \operatorname{Beta}(\alpha, \beta)",
                pdf_or_pmf_latex=r"f_X(x; \alpha, \beta) = \frac{1}{\mathrm{B}(\alpha,\beta)} x^{\alpha-1}(1-x)^{\beta-1}, \quad x \in [0, 1]",
                cdf_latex=r"F_X(x; \alpha, \beta) = I_x(\alpha, \beta)",
                mean_latex=r"\frac{\alpha}{\alpha+\beta}",
                variance_latex=r"\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}",
                kurtosis_latex=r"3 + \gamma_2(\alpha, \beta)",
                parameter_interpretations={
                    "α (Forma Esquerda / Pseudo-Sucessos)": "Puxa a massa de probabilidade em direção a 1. Em inferência Bayesiana, representa o número prévio de sucessos.",
                    "β (Forma Direita / Pseudo-Fracassos)": "Puxa a massa de probabilidade em direção a 0. Representa o número prévio de fracassos.",
                },
                conversion_notes="Se α = β, a distribuição é perfeitamente simétrica em torno de 0,5. Se α = 1 e β = 1, torna-se Uniforme(0, 1).",
            ),
            Parameterization(
                name="2. Parametrização por Média μ e Precisão/Tamanho Amostral ν (Regressão Beta)",
                parameters=r"(\mu, \nu) \text{ onde } \mu = \frac{\alpha}{\alpha+\beta} \in (0,1) \text{ e } \nu = \alpha + \beta > 0",
                canonical_writing_latex=r"X \sim \operatorname{Beta}_{\text{reg}}(\mu, \nu)",
                pdf_or_pmf_latex=r"f_X(x; \mu, \nu) = \frac{\Gamma(\nu)}{\Gamma(\mu\nu)\Gamma((1-\mu)\nu)} x^{\mu\nu-1}(1-x)^{(1-\mu)\nu-1}",
                cdf_latex=r"F_X(x; \mu, \nu) = I_x(\mu\nu, (1-\mu)\nu)",
                mean_latex=r"\mu",
                variance_latex=r"\frac{\mu(1-\mu)}{\nu + 1}",
                kurtosis_latex=r"3 + \gamma_2(\mu\nu, (1-\mu)\nu)",
                parameter_interpretations={
                    "μ (Proporção Média Esperada)": "Centro médio da proporção em (0, 1).",
                    "ν = α + β (Concentração / Tamanho de Amostra Equivalente)": "Quanto maior ν, menor a variância μ(1-μ)/(ν+1) ao redor de μ.",
                },
                conversion_notes="Proposta por Ferrari & Cribari-Neto para modelos de Regressão Beta onde a resposta é uma taxa ou proporção.",
            ),
        ],
        iid_and_statistical_properties={
            "Conjugação Bayesiana com Binomial/Bernoulli": r"Se a priori $p \sim \operatorname{Beta}(\alpha, \beta)$ e observamos $k$ sucessos em $n$ ensaios i.i.d. $\operatorname{Bernoulli}(p)$, a posteriori é $p \mid k \sim \operatorname{Beta}(\alpha + k, \, \beta + n - k)$.",
            "Relação com Gamas Independentes": r"Se $U \sim \operatorname{Gama}(\alpha, \theta)$ e $V \sim \operatorname{Gama}(\beta, \theta)$ são independentes, então $\frac{U}{U + V} \sim \operatorname{Beta}(\alpha, \beta)$.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": False,
            "unimodal": True,
            "skewness_target": 0.6,
            "excess_kurtosis_target": -0.1,
            "support_type": "unit_interval",
        },
    ),

    "chi_square": DistributionSpec(
        id="chi_square",
        name="Distribuição Qui-Quadrado (χ²)",
        aliases=["chi-square", "chi square", "qui-quadrado", "qui quadrado", "chi2", "chisq"],
        notation_patterns=[
            r"\chi\^2",
            r"Chi-?[Ss]quare",
            r"Qui-?[Qq]uadrado",
            r"chisq",
        ],
        formula_cues=["2^(k/2)", "x^(k/2-1)", "e^{-x/2}", "\\chi^2"],
        family="Contínua",
        support="x in [0, +infinito)",
        support_latex=r"x \in [0, +\infty)",
        canonical_single_rv_latex=r"X \sim \chi^2_k \quad \text{ou} \quad X \sim \chi^2(k)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \chi^2_k",
        didactic_overview=(
            "A Distribuição Qui-Quadrado com k graus de liberdade nasce da **soma dos quadrados de k variáveis aleatórias "
            "Normais Padrão independentes**: X = Z₁² + Z₂² + ... + Zₖ². Por isso, ela é a base para testes de aderência "
            "de Pearson, tabelas de contingência e intervalos de confiança para a variância amostral S²."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua X ≥ 0 representa uma soma de desvios quadráticos padronizados (energia ou variância amostral escalada)."
        ),
        primary_parameters="Graus de liberdade k (ou ν) ∈ {1, 2, 3, ...}",
        primary_parameters_latex=r"k \in \mathbb{N}^+ \text{ (graus de liberdade)}",
        default_params={"k": 4.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; k) = \frac{1}{2^{k/2}\Gamma(k/2)} x^{\frac{k}{2} - 1} e^{-x/2}, \quad x > 0",
        cdf_latex=r"F_X(x; k) = \frac{\gamma(k/2, x/2)}{\Gamma(k/2)} = P\left(\frac{k}{2}, \frac{x}{2}\right)",
        mean_latex=r"\mathbb{E}[X] = k",
        mean_didactic="Como cada Normal Padrão Z_i tem E[Z_i²] = Var(Z_i) = 1, a soma de k quadrados tem média exatamente igual a k!",
        variance_latex=r"\operatorname{Var}(X) = 2k",
        variance_didactic="Como Var(Z_i²) = E[Z_i⁴] - (E[Z_i²])² = 3 - 1² = 2, a soma de k termos independentes tem variância 2k.",
        median_latex=r"\operatorname{Mediana}(X) \approx k\left(1 - \frac{2}{9k}\right)^3 \quad (\text{Aproximação de Wilson–Hilferty})",
        mode_latex=r"\operatorname{Moda}(X) = \max(k - 2, 0)",
        skewness_latex=r"\gamma_1 = \sqrt{\frac{8}{k}}",
        excess_kurtosis_latex=r"\gamma_2 = \frac{12}{k}",
        raw_kurtosis_latex=r"\beta_2 = 3 + \frac{12}{k}",
        kurtosis_didactic="Para k pequeno, a curva é fortemente leptocúrtica (γ₂ = 12/k); quando k cresce, γ₂ → 0 e ela se aproxima de uma Normal(k, 2k).",
        raw_moment_k_latex=r"\mathbb{E}[X^m] = 2^m \frac{\Gamma\left(\frac{k}{2} + m\right)}{\Gamma\left(\frac{k}{2}\right)} = k(k+2)(k+4)\cdots(k+2m-2)",
        central_moment_k_latex=r"\mu_2 = 2k, \quad \mu_3 = 8k, \quad \mu_4 = 12k(k + 4)",
        mgf_latex=r"M_X(t) = (1 - 2t)^{-k/2}, \quad \text{para } t < \frac{1}{2}",
        cf_latex=r"\varphi_X(t) = (1 - 2it)^{-k/2}",
        moments_didactic_guide=(
            "A Qui-Quadrado χ²_k é exatamente um caso particular da distribuição Gama com forma α = k/2 e taxa β = 1/2 (escala θ = 2)."
        ),
        entropy_latex=r"h(X) = \frac{k}{2} + \ln\left(2\Gamma\left(\frac{k}{2}\right)\right) + \left(1 - \frac{k}{2}\right)\psi\left(\frac{k}{2}\right)",
        fisher_info_latex=r"\mathcal{I}(k) = \frac{1}{4}\psi_1\left(\frac{k}{2}\right)",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Central Padrão por Graus de Liberdade k",
                parameters=r"k \in \mathbb{N}^+",
                canonical_writing_latex=r"X \sim \chi^2_k",
                pdf_or_pmf_latex=r"f_X(x; k) = \frac{x^{k/2-1} e^{-x/2}}{2^{k/2}\Gamma(k/2)}, \quad x > 0",
                cdf_latex=r"F_X(x; k) = \frac{\gamma(k/2, x/2)}{\Gamma(k/2)}",
                mean_latex=r"k",
                variance_latex=r"2k",
                kurtosis_latex=r"3 + \frac{12}{k} \quad \left(\gamma_2 = \frac{12}{k}\right)",
                parameter_interpretations={
                    "k (Graus de Liberdade)": "Número de parcelas Normais Padrão independentes elevadas ao quadrado que compõem a soma.",
                },
                conversion_notes="Equivale a Gama(forma = k/2, taxa = 1/2).",
            ),
            Parameterization(
                name="2. Qui-Quadrado Não-Central (Poder de Testes Estatísticos)",
                parameters=r"(k, \lambda) \text{ onde } \lambda = \sum_{i=1}^k \mu_i^2 \ge 0 \text{ é o parâmetro de não-centralidade}",
                canonical_writing_latex=r"X \sim \chi'^2_k(\lambda)",
                pdf_or_pmf_latex=r"f_X(x; k, \lambda) = \frac{1}{2} e^{-(x+\lambda)/2} \left(\frac{x}{\lambda}\right)^{k/4 - 1/2} I_{k/2 - 1}(\sqrt{\lambda x})",
                cdf_latex=r"F_X(x; k, \lambda) = \sum_{j=0}^\infty e^{-\lambda/2} \frac{(\lambda/2)^j}{j!} F_{\chi^2_{k+2j}}(x)",
                mean_latex=r"k + \lambda",
                variance_latex=r"2(k + 2\lambda)",
                kurtosis_latex=r"3 + \frac{12(k + 4\lambda)}{(k + 2\lambda)^2}",
                parameter_interpretations={
                    "k (Graus de Liberdade)": "Dimensão do espaço de variáveis Normais somadas.",
                    "λ (Parâmetro de Não-Centralidade)": "Soma dos quadrados das médias quando a hipótese nula H₀ é falsa (mede o afastamento de H₀).",
                },
                conversion_notes="Surge quando somamos quadrados de Normais com média diferente de zero: Y_i ~ N(μ_i, 1).",
            ),
        ],
        iid_and_statistical_properties={
            "Construção a partir de Normais Padrão i.i.d.": r"Se $Z_1, \dots, Z_k \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$, então $\sum_{i=1}^k Z_i^2 \sim \chi^2_k$.",
            "Aditividade sob Independência": r"Se $X_1 \sim \chi^2_{k_1}$ e $X_2 \sim \chi^2_{k_2}$ são independentes, então $X_1 + X_2 \sim \chi^2_{k_1 + k_2}$.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": False,
            "unimodal": True,
            "skewness_target": 1.41,
            "excess_kurtosis_target": 3.0,
            "support_type": "non_negative",
        },
    ),

    "student_t": DistributionSpec(
        id="student_t",
        name="Distribuição t de Student",
        aliases=["student t", "t-distribution", "t de student", "student"],
        notation_patterns=[
            r"t_\{\\nu\}",
            r"Student",
            r"t-distribution",
            r"\operatorname\{t\}\s*\(",
        ],
        formula_cues=["(1 + t^2/nu)", "(1+x^2/\\nu)^{-(\\nu+1)/2}", "\\Gamma((\\nu+1)/2)"],
        family="Contínua",
        support="x in (-infinito, +infinito)",
        support_latex=r"x \in (-\infty, +\infty)",
        canonical_single_rv_latex=r"T \sim t_\nu \quad \text{ou} \quad T \sim \operatorname{Student}(\nu)",
        canonical_iid_rv_latex=r"T_1, T_2, \dots, T_n \stackrel{\text{i.i.d.}}{\sim} t_\nu",
        didactic_overview=(
            "Criada por William Sealy Gosset ('Student'), a distribuição t surge ao padronizarmos a média amostral de "
            "uma amostra Normal quando a **variância populacional σ² é desconhecida** e substituída pela variância "
            "amostral S². Possui formato de sino simétrico como a Normal, mas com **caudas mais pesadas** (maior propensão a valores extremos)."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua T ∈ ℝ representa uma razão entre uma Normal padrão e a raiz de uma Qui-Quadrado independente, "
            "ou dados reais sujeitos a outliers."
        ),
        primary_parameters="Graus de liberdade ν > 0",
        primary_parameters_latex=r"\nu > 0 \text{ (graus de liberdade)}",
        default_params={"nu": 5.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_T(t; \nu) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu + 1}{2}}, \quad t \in \mathbb{R}",
        cdf_latex=r"F_T(t; \nu) = \frac{1}{2} + t \Gamma\left(\frac{\nu+1}{2}\right) \frac{{}_2F_1\left(\frac{1}{2}, \frac{\nu+1}{2}; \frac{3}{2}; -\frac{t^2}{\nu}\right)}{\sqrt{\pi\nu}\,\Gamma\left(\frac{\nu}{2}\right)}",
        mean_latex=r"\mathbb{E}[T] = 0 \quad (\text{somente para } \nu > 1; \text{ indefinida se } \nu \le 1)",
        mean_didactic="Sempre centrada em 0 quando ν > 1. Para ν = 1 (Distribuição de Cauchy), as caudas são tão pesadas que a integral da média não converge!",
        variance_latex=r"\operatorname{Var}(T) = \frac{\nu}{\nu - 2} \quad (\text{para } \nu > 2; \; +\infty \text{ se } 1 < \nu \le 2)",
        variance_didactic="É sempre maior que 1 (pois ν/(ν-2) > 1), refletindo a incerteza extra de estimar σ por S. Quando ν → ∞, Var(T) → 1.",
        median_latex=r"\operatorname{Mediana}(T) = 0 \quad (\text{existe para todo } \nu > 0)",
        mode_latex=r"\operatorname{Moda}(T) = 0",
        skewness_latex=r"\gamma_1 = 0 \quad (\text{para } \nu > 3)",
        excess_kurtosis_latex=r"\gamma_2 = \frac{6}{\nu - 4} \quad (\text{para } \nu > 4; \; +\infty \text{ se } 2 < \nu \le 4)",
        raw_kurtosis_latex=r"\beta_2 = 3 + \frac{6}{\nu - 4} = \frac{3(\nu - 2)}{\nu - 4} \quad (\text{para } \nu > 4)",
        kurtosis_didactic="Como γ₂ = 6/(ν-4) > 0, a t de Student é sempre leptocúrtica (caudas pesadas). Quanto menor ν, mais pesadas são as caudas.",
        raw_moment_k_latex=r"\mathbb{E}[T^k] = \begin{cases} 0 & \text{se } k \text{ é ímpar e } k < \nu \\ \nu^{k/2} \prod_{i=1}^{k/2} \frac{2i - 1}{\nu - 2i} & \text{se } k \text{ é par e } k < \nu \\ \text{Não existe} & \text{se } k \ge \nu \end{cases}",
        central_moment_k_latex=r"\mu_k = \mathbb{E}[T^k] \quad (\text{pois } \mu = 0)",
        mgf_latex=r"M_T(t) = \text{Não existe para } t \neq 0 \text{ (caudas decaem polinomialmente, não exponencialmente!)}",
        cf_latex=r"\varphi_T(t) = \frac{K_{\nu/2}(\sqrt{\nu}|t|) \, (\sqrt{\nu}|t|)^{\nu/2}}{\Gamma(\nu/2) 2^{\nu/2 - 1}}",
        moments_didactic_guide=(
            "Fato fundamental para provas: na t de Student com ν graus de liberdade, SÓ EXISTEM momentos até a ordem k < ν! "
            "Como os momentos de ordem k ≥ ν divergem, a Função Geradora de Momentos M_T(t) = E[e^{tT}] não existe para t ≠ 0."
        ),
        entropy_latex=r"h(T) = \frac{\nu + 1}{2}\left[\psi\left(\frac{\nu+1}{2}\right) - \psi\left(\frac{\nu}{2}\right)\right] + \ln\left(\sqrt{\nu}\,\mathrm{B}\left(\frac{\nu}{2}, \frac{1}{2}\right)\right)",
        fisher_info_latex=r"\mathcal{I}(\mu) = \frac{\nu + 1}{\sigma^2(\nu + 3)}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Padrão (1 Parâmetro: Graus de Liberdade ν)",
                parameters=r"\nu > 0",
                canonical_writing_latex=r"T \sim t_\nu",
                pdf_or_pmf_latex=r"f_T(t; \nu) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\Gamma(\nu/2)}\left(1+\frac{t^2}{\nu}\right)^{-(\nu+1)/2}",
                cdf_latex=r"F_T(t; \nu) = 1 - \frac{1}{2} I_{\frac{\nu}{\nu+t^2}}\left(\frac{\nu}{2}, \frac{1}{2}\right) \quad (t \ge 0)",
                mean_latex=r"0 \quad (\nu > 1)",
                variance_latex=r"\frac{\nu}{\nu - 2} \quad (\nu > 2)",
                kurtosis_latex=r"3 + \frac{6}{\nu - 4} \quad (\nu > 4)",
                parameter_interpretations={
                    "ν (Graus de Liberdade / Peso das Caudas)": (
                        "Controla o peso das caudas. Em testes t de uma amostra de tamanho n, ν = n - 1. "
                        "Quando ν = 1, é a distribuição de Cauchy; quando ν > 30, já é quase idêntica à Normal(0, 1)."
                    ),
                },
                conversion_notes="Usada diretamente no Teste t de Student e intervalos de confiança com variância desconhecida.",
            ),
            Parameterization(
                name="2. Parametrização de Locação e Escala de 3 Parâmetros (Finanças e Bayesiana Robusta)",
                parameters=r"(\nu, \mu, \sigma) \text{ onde } X = \mu + \sigma T",
                canonical_writing_latex=r"X \sim t_\nu(\mu, \sigma^2)",
                pdf_or_pmf_latex=r"f_X(x; \nu, \mu, \sigma) = \frac{\Gamma((\nu+1)/2)}{\sigma\sqrt{\nu\pi}\Gamma(\nu/2)}\left[1+\frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right]^{-(\nu+1)/2}",
                cdf_latex=r"F_X(x; \nu, \mu, \sigma) = F_T\left(\frac{x - \mu}{\sigma}; \nu\right)",
                mean_latex=r"\mu \quad (\nu > 1)",
                variance_latex=r"\sigma^2 \frac{\nu}{\nu - 2} \quad (\nu > 2)",
                kurtosis_latex=r"3 + \frac{6}{\nu - 4} \quad (\nu > 4)",
                parameter_interpretations={
                    "μ (Locação / Centro)": "Média e centro de simetria dos dados.",
                    "σ (Escala)": "Fator de escala (atenção: a variância total é σ² · ν/(ν-2)).",
                    "ν (Graus de Liberdade / Robustez)": "Controla a resistência a valores discrepantes (outliers).",
                },
                conversion_notes="Muito usada para modelar retornos financeiros diários (que têm caudas pesadas).",
            ),
        ],
        iid_and_statistical_properties={
            "Definição por Razão Independente": r"Se $Z \sim \mathcal{N}(0, 1)$ e $V \sim \chi^2_\nu$ são independentes ($Z \perp\!\!\!\perp V$), então $T = \frac{Z}{\sqrt{V/\nu}} \sim t_\nu$.",
            "Pivô em Amostra Normal i.i.d.": r"Para $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$, a estatística $T = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": True,
            "unimodal": True,
            "skewness_target": 0.0,
            "excess_kurtosis_target": 6.0,
            "support_type": "real_line",
        },
    ),

    "lognormal": DistributionSpec(
        id="lognormal",
        name="Distribuição Log-Normal (ou de Galton)",
        aliases=["lognormal", "log-normal", "galton"],
        notation_patterns=[
            r"LogNormal\s*\(",
            r"\operatorname\{Lognormal\}",
            r"\ln\s*X\s*~\s*N",
        ],
        formula_cues=["1/(x*sigma*sqrt(2*pi))", "(\\ln x - \\mu)^2", "(ln(x)-mu)^2"],
        family="Contínua",
        support="x in (0, +infinito)",
        support_latex=r"x \in (0, +\infty)",
        canonical_single_rv_latex=r"X \sim \operatorname{LogNormal}(\mu, \sigma^2) \iff \ln(X) \sim \mathcal{N}(\mu, \sigma^2)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{LogNormal}(\mu, \sigma^2)",
        didactic_overview=(
            "Uma variável positiva X tem distribuição Log-Normal quando o seu **logaritmo natural Y = ln(X) segue uma "
            "distribuição Normal(μ, σ²)**. Enquanto a Normal surge da SOMA de efeitos independentes, a Log-Normal surge "
            "do **PRODUTO multiplicativo** de muitos fatores positivos independentes (como preços de ações no modelo Black-Scholes, "
            "distribuição de renda salarial e tamanho de partículas)."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua estritamente positiva X > 0 cresce de forma multiplicativa."
        ),
        primary_parameters="Média logarítmica μ ∈ ℝ e Variância logarítmica σ² > 0 (parâmetros de ln X)",
        primary_parameters_latex=r"\mu \in \mathbb{R}, \quad \sigma^2 > 0 \quad (\text{média e variância de } \ln X)",
        default_params={"mu": 0.0, "sigma": 0.6},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; \mu, \sigma^2) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0",
        cdf_latex=r"F_X(x; \mu, \sigma^2) = \Phi\left(\frac{\ln x - \mu}{\sigma}\right), \quad x > 0",
        mean_latex=r"\mathbb{E}[X] = \exp\left(\mu + \frac{\sigma^2}{2}\right)",
        mean_didactic="Atenção: a média de X NÃO é e^μ, mas sim e^(μ + σ²/2), pois a exponencial puxa valores grandes na cauda direita!",
        variance_latex=r"\operatorname{Var}(X) = \left(e^{\sigma^2} - 1\right) e^{2\mu + \sigma^2}",
        variance_didactic="Cresce exponencialmente com σ².",
        median_latex=r"\operatorname{Mediana}(X) = e^\mu \quad (\text{Média Geométrica})",
        mode_latex=r"\operatorname{Moda}(X) = e^{\mu - \sigma^2}",
        skewness_latex=r"\gamma_1 = \left(e^{\sigma^2} + 2\right)\sqrt{e^{\sigma^2} - 1}",
        excess_kurtosis_latex=r"\gamma_2 = e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 6",
        raw_kurtosis_latex=r"\beta_2 = e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 3",
        kurtosis_didactic="Possui alta curtose positiva (leptocúrtica), crescendo muito rápido conforme σ aumenta.",
        raw_moment_k_latex=r"\mathbb{E}[X^k] = \mathbb{E}[e^{k \ln X}] = M_{\ln X}(k) = \exp\left(k\mu + \frac{1}{2}k^2\sigma^2\right)",
        central_moment_k_latex=r"\mu_2 = (e^{\sigma^2}-1)e^{2\mu+\sigma^2}",
        mgf_latex=r"M_X(t) = \text{Não existe (diverge) para } t > 0, \text{ embora todos os momentos } \mathbb{E}[X^k] \text{ sejam finitos!}",
        cf_latex=r"\varphi_X(t) = \text{Sem forma fechada elementar}",
        moments_didactic_guide=(
            "Truque genial para calcular E[X^k] na Log-Normal: como X = e^Y onde Y ~ N(μ, σ²), temos "
            "E[X^k] = E[e^{kY}], que é exatamente a Função Geradora de Momentos da Normal avaliada em t = k: exp(kμ + ½k²σ²)!"
        ),
        entropy_latex=r"h(X) = \mu + \frac{1}{2}\ln(2\pi e \sigma^2)",
        fisher_info_latex=r"\mathcal{I}(\mu, \sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização na Escala Logarítmica (μ, σ²)",
                parameters=r"(\mu, \sigma^2) \text{ onde } \ln X \sim \mathcal{N}(\mu, \sigma^2)",
                canonical_writing_latex=r"X \sim \operatorname{LogNormal}(\mu, \sigma^2)",
                pdf_or_pmf_latex=r"f_X(x; \mu, \sigma^2) = \frac{1}{x\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)",
                cdf_latex=r"F_X(x; \mu, \sigma^2) = \Phi\left(\frac{\ln x - \mu}{\sigma}\right)",
                mean_latex=r"e^{\mu + \sigma^2/2}",
                variance_latex=r"(e^{\sigma^2}-1)e^{2\mu+\sigma^2}",
                kurtosis_latex=r"e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 3",
                parameter_interpretations={
                    "μ (Média do Logaritmo ln X)": "Não é a média de X, mas o logaritmo da mediana: e^μ é exatamente a Mediana de X!",
                    "σ² (Variância do Logaritmo ln X)": "Controla a assimetria e o peso da cauda direita na escala original.",
                },
                conversion_notes="Padrão em probabilidade e no R (dlnorm(x, meanlog=μ, sdlog=σ)).",
            ),
            Parameterization(
                name="2. Parametrização por Forma s e Escala/Mediana m (SciPy)",
                parameters=r"(s, m) \text{ onde } s = \sigma \text{ e } m = e^\mu",
                canonical_writing_latex=r"X \sim \operatorname{LogNormal}(s=\sigma, \text{scale}=e^\mu)",
                pdf_or_pmf_latex=r"f_X(x; s, m) = \frac{1}{s x \sqrt{2\pi}} \exp\left(-\frac{\ln^2(x/m)}{2s^2}\right)",
                cdf_latex=r"F_X(x; s, m) = \Phi\left(\frac{\ln(x/m)}{s}\right)",
                mean_latex=r"m \, e^{s^2/2}",
                variance_latex=r"m^2 (e^{s^2}-1)e^{s^2}",
                kurtosis_latex=r"e^{4s^2} + 2e^{3s^2} + 3e^{2s^2} - 3",
                parameter_interpretations={
                    "m = e^μ (Mediana / Escala Geométrica)": "Valor mediano direto na escala de X.",
                    "s = σ (Desvio-Padrão Geométrico Logarítmico)": "Parâmetro de forma que governa o espalhamento multiplicativo.",
                },
                conversion_notes="Usada em scipy.stats.lognorm(s=σ, scale=exp(μ)).",
            ),
        ],
        iid_and_statistical_properties={
            "Fechamento Multiplicativo sob Independência": r"Se $X_1, \dots, X_n$ são independentes com $X_i \sim \operatorname{LogNormal}(\mu_i, \sigma_i^2)$, o produto segue $\prod_{i=1}^n X_i \sim \operatorname{LogNormal}\left(\sum_{i=1}^n \mu_i, \, \sum_{i=1}^n \sigma_i^2\right)$.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": False,
            "unimodal": True,
            "skewness_target": 2.2,
            "excess_kurtosis_target": 8.0,
            "support_type": "non_negative",
        },
    ),

    "weibull": DistributionSpec(
        id="weibull",
        name="Distribuição de Weibull",
        aliases=["weibull"],
        notation_patterns=[
            r"Weibull\s*\(",
            r"\operatorname\{Weibull\}",
        ],
        formula_cues=["(k/lambda)*(x/lambda)^(k-1)", "e^{-(x/\\lambda)^k}"],
        family="Contínua",
        support="x in [0, +infinito)",
        support_latex=r"x \in [0, +\infty)",
        canonical_single_rv_latex=r"X \sim \operatorname{Weibull}(\lambda, k)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Weibull}(\lambda, k)",
        didactic_overview=(
            "A Distribuição de Weibull é o pilar da **Engenharia de Confiabilidade e Análise de Sobrevivência**. "
            "Diferente da Exponencial (que assume taxa de falha constante), a Weibull permite modelar sistemas que "
            "sofrem envelhecimento/desgaste (taxa de falha crescente, k > 1) ou mortalidade infantil (taxa decrescente, k < 1)."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua X ≥ 0 representa o tempo até a falha de um material, peça mecânica ou velocidade de ventos."
        ),
        primary_parameters="Escala λ > 0 e Forma k > 0",
        primary_parameters_latex=r"\lambda > 0 \text{ (escala)}, \quad k > 0 \text{ (forma)}",
        default_params={"k": 2.0, "lam": 1.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; \lambda, k) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, \quad x \ge 0",
        cdf_latex=r"F_X(x; \lambda, k) = 1 - e^{-(x/\lambda)^k}, \quad x \ge 0",
        mean_latex=r"\mathbb{E}[X] = \lambda \Gamma\left(1 + \frac{1}{k}\right)",
        mean_didactic="Proporcional ao parâmetro de escala λ modulado pela função Gama(1 + 1/k).",
        variance_latex=r"\operatorname{Var}(X) = \lambda^2 \left[\Gamma\left(1 + \frac{2}{k}\right) - \left(\Gamma\left(1 + \frac{1}{k}\right)\right)^2\right]",
        variance_didactic="Cresce com λ² e depende da forma k.",
        median_latex=r"\operatorname{Mediana}(X) = \lambda (\ln 2)^{1/k}",
        mode_latex=r"\operatorname{Moda}(X) = \lambda \left(\frac{k - 1}{k}\right)^{1/k} \quad (\text{para } k > 1; \; 0 \text{ se } k \le 1)",
        skewness_latex=r"\gamma_1 = \frac{\Gamma(1 + 3/k)\lambda^3 - 3\mu\sigma^2 - \mu^3}{\sigma^3}",
        excess_kurtosis_latex=r"\gamma_2 = \frac{\lambda^4\Gamma(1+4/k) - 4\gamma_1\sigma^3\mu - 6\mu^2\sigma^2 - \mu^4}{\sigma^4} - 3",
        raw_kurtosis_latex=r"\beta_2 = \gamma_2 + 3",
        kurtosis_didactic="Depende apenas do parâmetro de forma k.",
        raw_moment_k_latex=r"\mathbb{E}[X^m] = \lambda^m \Gamma\left(1 + \frac{m}{k}\right)",
        central_moment_k_latex=r"\mu_2 = \lambda^2 \left[\Gamma(1+2/k) - \Gamma(1+1/k)^2\right]",
        mgf_latex=r"M_X(t) = \sum_{n=0}^\infty \frac{t^n \lambda^n}{n!} \Gamma\left(1 + \frac{n}{k}\right) \quad (k \ge 1)",
        cf_latex=r"\varphi_X(t) = \sum_{n=0}^\infty \frac{(it)^n \lambda^n}{n!} \Gamma\left(1 + \frac{n}{k}\right)",
        moments_didactic_guide=(
            "Fazendo a substituição u = (x/λ)^k na integral de E[X^m], temos x = λ u^(1/k) e obtemos imediatamente λ^m Γ(1 + m/k)."
        ),
        entropy_latex=r"h(X) = \gamma_{\text{Euler}}\left(1 - \frac{1}{k}\right) + \ln\left(\frac{\lambda}{k}\right) + 1",
        fisher_info_latex=r"\mathcal{I}(k, \lambda) = \begin{pmatrix} \frac{1}{k^2}\left[\frac{\pi^2}{6} + (1-\gamma)^2\right] & -\frac{1-\gamma}{\lambda} \\ -\frac{1-\gamma}{\lambda} & \frac{k^2}{\lambda^2} \end{pmatrix}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Padrão de Escala λ e Forma k (Engenharia e Confiabilidade)",
                parameters=r"(\lambda, k) > 0",
                canonical_writing_latex=r"X \sim \operatorname{Weibull}(\lambda, k)",
                pdf_or_pmf_latex=r"f_X(x; \lambda, k) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}",
                cdf_latex=r"F_X(x; \lambda, k) = 1 - e^{-(x/\lambda)^k}",
                mean_latex=r"\lambda\Gamma(1 + 1/k)",
                variance_latex=r"\lambda^2[\Gamma(1+2/k) - \Gamma(1+1/k)^2]",
                kurtosis_latex=r"\beta_2(k)",
                parameter_interpretations={
                    "k (Módulo de Weibull / Forma)": (
                        "Se k < 1: taxa de falha decrescente (falhas precoces). "
                        "Se k = 1: taxa de falha constante (vira a distribuição Exponencial!). "
                        "Se k > 1: taxa de falha crescente (desgaste/envelhecimento)."
                    ),
                    "λ (Vida Característica / Escala)": (
                        "É o tempo exato em que 63,21% (1 - e⁻¹) das unidades já falharam, qualquer que seja o valor de k!"
                    ),
                },
                conversion_notes="Padrão em SciPy (weibull_min) e R (dweibull).",
            ),
            Parameterization(
                name="2. Parametrização de Risco Proporcional (b, k)",
                parameters=r"(b, k) \text{ onde } b = \lambda^{-k}",
                canonical_writing_latex=r"X \sim \operatorname{Weibull}_{\text{risco}}(b, k)",
                pdf_or_pmf_latex=r"f_X(x; b, k) = b k x^{k-1} e^{-b x^k}",
                cdf_latex=r"F_X(x; b, k) = 1 - e^{-b x^k}",
                mean_latex=r"b^{-1/k}\Gamma(1 + 1/k)",
                variance_latex=r"b^{-2/k}[\Gamma(1+2/k) - \Gamma(1+1/k)^2]",
                kurtosis_latex=r"\beta_2(k)",
                parameter_interpretations={
                    "b = λ^(-k) (Multiplicador de Taxa de Risco)": "Escala diretamente a função de risco (hazard function) h(x) = b k x^(k-1).",
                    "k (Expoente Temporal de Risco)": "Potência com que o risco cresce no tempo.",
                },
                conversion_notes="Usada em modelos de regressão de sobrevivência de riscos proporcionais de Cox.",
            ),
        ],
        iid_and_statistical_properties={
            "Fechamento para o Mínimo de Amostra i.i.d. (Elo Mais Fraco)": r"Se $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Weibull}(\lambda, k)$, então $\min(X_1, \dots, X_n) \sim \operatorname{Weibull}(\lambda n^{-1/k}, k)$.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": False,
            "unimodal": True,
            "skewness_target": 0.63,
            "excess_kurtosis_target": 0.25,
            "support_type": "non_negative",
        },
    ),

    "poisson": DistributionSpec(
        id="poisson",
        name="Distribuição de Poisson",
        aliases=["poisson", "pois", "eventos raros"],
        notation_patterns=[
            r"Pois\s*\(",
            r"Poisson\s*\(",
            r"\frac\{\\lambda\^k\s*e\^\{-\\lambda\}\}\{k!\}",
        ],
        formula_cues=["lambda^k", "e^{-lambda}/k!", "\\frac{\\lambda^k e^{-\\lambda}}{k!}", "k!"],
        family="Discreta",
        support="k in {0, 1, 2, 3, ...}",
        support_latex=r"k \in \mathbb{N}_0 = \{0, 1, 2, 3, \dots\}",
        canonical_single_rv_latex=r"X \sim \operatorname{Poisson}(\lambda) \quad \text{ou} \quad X \sim \operatorname{Pois}(\lambda)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Poisson}(\lambda)",
        didactic_overview=(
            "A Distribuição de Poisson modela a **contagem discreta de eventos** que ocorrem de forma independente "
            "dentro de um intervalo fixo de tempo, área ou volume (ex.: número de chamadas recebidas por uma central "
            "em 1 hora, número de defeitos por metro de tecido ou número de gols em uma partida)."
        ),
        random_variable_meaning=(
            "A variável aleatória discreta X ∈ {0, 1, 2, ...} conta quantas vezes o evento ocorreu no intervalo observado."
        ),
        primary_parameters="Taxa média de ocorrências λ > 0 no intervalo",
        primary_parameters_latex=r"\lambda > 0 \text{ (número médio de ocorrências no intervalo)}",
        default_params={"lam": 4.0},
        pdf_or_pmf_name="FMP (Função Massa de Probabilidade — Discreta)",
        pdf_or_pmf_latex=r"P(X = k; \lambda) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k \in \{0, 1, 2, \dots\}",
        cdf_latex=r"F_X(k; \lambda) = \mathbb{P}(X \le k) = e^{-\lambda} \sum_{j=0}^{\lfloor k \rfloor} \frac{\lambda^j}{j!} = \frac{\Gamma(\lfloor k + 1 \rfloor, \lambda)}{\lfloor k \rfloor!}",
        mean_latex=r"\mathbb{E}[X] = \lambda",
        mean_didactic="O próprio parâmetro λ é a contagem média esperada de ocorrências.",
        variance_latex=r"\operatorname{Var}(X) = \lambda \quad (\text{Propriedade de Equidispersão: Média = Variância})",
        variance_didactic=(
            "Na Poisson, a variância é EXATAMENTE igual à média (Var(X) = E[X] = λ). "
            "Na prática, se os seus dados de contagem têm variância muito maior que a média (superdispersão), usa-se a Binomial Negativa!"
        ),
        median_latex=r"\operatorname{Mediana}(X) \approx \left\lfloor \lambda + \frac{1}{3} - \frac{0.02}{\lambda} \right\rfloor",
        mode_latex=r"\operatorname{Moda}(X) = \lfloor \lambda \rfloor \quad (\text{se } \lambda \in \mathbb{Z}^+, \text{ bimodal em } \lambda-1 \text{ e } \lambda)",
        skewness_latex=r"\gamma_1 = \frac{1}{\sqrt{\lambda}}",
        excess_kurtosis_latex=r"\gamma_2 = \frac{1}{\lambda}",
        raw_kurtosis_latex=r"\beta_2 = 3 + \frac{1}{\lambda}",
        kurtosis_didactic="Leptocúrtica com excesso γ₂ = 1/λ. Conforme a taxa média λ cresce, γ₂ → 0 e a Poisson se aproxima de uma Normal(λ, λ).",
        raw_moment_k_latex=r"\mathbb{E}[X^m] = \sum_{j=0}^m \left\{ \begin{matrix} m \\ j \end{matrix} \right\} \lambda^j \quad \Big(\mathbb{E}[X]=\lambda, \; \mathbb{E}[X^2]=\lambda+\lambda^2, \; \mathbb{E}[X^3]=\lambda+3\lambda^2+\lambda^3\Big)",
        central_moment_k_latex=r"\mu_2 = \lambda, \quad \mu_3 = \lambda, \quad \mu_4 = \lambda(1 + 3\lambda) \quad (\text{Todos os cumulantes valem } \kappa_m = \lambda)",
        mgf_latex=r"M_X(t) = \mathbb{E}[e^{tX}] = \exp\left(\lambda(e^t - 1)\right), \quad t \in \mathbb{R}",
        cf_latex=r"\varphi_X(t) = \exp\left(\lambda(e^{it} - 1)\right)",
        moments_didactic_guide=(
            "Na Poisson, os momentos fatoriais são facílimos: E[X(X-1)...(X-m+1)] = λ^m. "
            "Além disso, tomando o logaritmo da FGM, obtemos K_X(t) = ln M_X(t) = λ(e^t - 1) = ∑_{m=1}^∞ λ (t^m / m!), "
            "provando que TODOS os cumulantes da Poisson são iguais a λ!"
        ),
        entropy_latex=r"H(X) = \lambda[1 - \ln(\lambda)] + e^{-\lambda}\sum_{k=0}^\infty \frac{\lambda^k \ln(k!)}{k!}",
        fisher_info_latex=r"\mathcal{I}(\lambda) = \frac{1}{\lambda}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Padrão por Taxa Média no Intervalo (λ)",
                parameters=r"\lambda > 0",
                canonical_writing_latex=r"X \sim \operatorname{Poisson}(\lambda)",
                pdf_or_pmf_latex=r"P(X = k; \lambda) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k = 0, 1, 2, \dots",
                cdf_latex=r"F_X(k; \lambda) = \sum_{j=0}^{\lfloor k \rfloor} \frac{e^{-\lambda}\lambda^j}{j!}",
                mean_latex=r"\lambda",
                variance_latex=r"\lambda",
                kurtosis_latex=r"3 + \frac{1}{\lambda} \quad \left(\gamma_2 = \frac{1}{\lambda}\right)",
                parameter_interpretations={
                    "λ (Intensidade / Número Médio de Eventos)": (
                        "Quantidade média esperada de ocorrências na janela observada. Também é igual à variância da contagem."
                    ),
                },
                conversion_notes="Padrão em probabilidade, SciPy (scipy.stats.poisson(mu=λ)) e R (dpois).",
            ),
            Parameterization(
                name="2. Parametrização por Taxa Unitária ρ e Exposição/Tempo t (Processo de Poisson)",
                parameters=r"(\rho, t) > 0 \text{ onde } \lambda = \rho t",
                canonical_writing_latex=r"N(t) \sim \operatorname{Poisson}(\rho t)",
                pdf_or_pmf_latex=r"P(N(t) = k) = \frac{e^{-\rho t}(\rho t)^k}{k!}, \quad k = 0, 1, 2, \dots",
                cdf_latex=r"F_{N(t)}(k) = \sum_{j=0}^{\lfloor k \rfloor} \frac{e^{-\rho t}(\rho t)^j}{j!}",
                mean_latex=r"\rho t",
                variance_latex=r"\rho t",
                kurtosis_latex=r"3 + \frac{1}{\rho t}",
                parameter_interpretations={
                    "ρ (Taxa por Unidade de Tempo/Área)": "Ritmo intrínseco de chegada (ex.: 3 clientes por minuto).",
                    "t (Tamanho da Janela de Exposição)": "Duração do período ou tamanho da região observada (ex.: 10 minutos → média ρt = 30).",
                },
                conversion_notes="Conecta diretamente a contagem discreta N(t) ~ Poisson(ρt) ao tempo contínuo entre chegadas T ~ Exp(ρ).",
            ),
            Parameterization(
                name="3. Parametrização Canônica Log-Linear (Regressão de Poisson GLM)",
                parameters=r"\eta = \ln(\lambda) \in \mathbb{R}",
                canonical_writing_latex=r"X \sim \operatorname{Poisson}(e^\eta)",
                pdf_or_pmf_latex=r"P(X = k; \eta) = \frac{1}{k!} \exp\left(\eta k - e^\eta\right)",
                cdf_latex=r"F_X(k; \eta) = \sum_{j=0}^{\lfloor k \rfloor} \frac{1}{j!}\exp(\eta j - e^\eta)",
                mean_latex=r"e^\eta",
                variance_latex=r"e^\eta",
                kurtosis_latex=r"3 + e^{-\eta}",
                parameter_interpretations={
                    "η = ln(λ) (Preditor Linear Logarítmico)": "Permite que η varie livremente em (-∞, +∞) garantindo que a taxa média λ = e^η seja sempre positiva.",
                },
                conversion_notes="Explica por que a função de ligação canônica da Regressão de Poisson é o logaritmo.",
            ),
        ],
        iid_and_statistical_properties={
            "Fechamento Aditivo sob Independência (Teorema de Raikov)": r"Se $X_1, \dots, X_n$ são independentes com $X_i \sim \operatorname{Poisson}(\lambda_i)$, então $\sum_{i=1}^n X_i \sim \operatorname{Poisson}\left(\sum_{i=1}^n \lambda_i\right)$.",
            "Amostra i.i.d. e Estimador de Máxima Verossimilhança": r"Para $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Poisson}(\lambda)$, a soma $\sum_{i=1}^n X_i \sim \operatorname{Poisson}(n\lambda)$ é suficiente e completa, e $\hat{\lambda}_{\text{EMV}} = \bar{X}$ atinge o limite inferior de Cramér-Rao $\lambda/n$.",
        },
        shape_signature={
            "is_discrete": True,
            "symmetric": False,
            "unimodal": True,
            "skewness_target": 0.5,
            "excess_kurtosis_target": 0.25,
            "support_type": "non_negative_integers",
        },
    ),

    "binomial": DistributionSpec(
        id="binomial",
        name="Distribuição Binomial (e Bernoulli)",
        aliases=["binomial", "bernoulli", "binom"],
        notation_patterns=[
            r"Bin\s*\(",
            r"Binom\s*\(",
            r"Binomial\s*\(",
            r"Bernoulli\s*\(",
            r"Bern\s*\(",
            r"\binom\{n\}\{k\}",
        ],
        formula_cues=["(n choose k)", "\\binom{n}{k}", "p^k*(1-p)^(n-k)", "np(1-p)"],
        family="Discreta",
        support="k in {0, 1, 2, ..., n}",
        support_latex=r"k \in \{0, 1, 2, \dots, n\}",
        canonical_single_rv_latex=r"X \sim \operatorname{Binomial}(n, p) \quad \text{ou} \quad X \sim \operatorname{Bin}(n, p)",
        canonical_iid_rv_latex=r"Y_1, Y_2, \dots, Y_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Bernoulli}(p) \implies X = \sum_{i=1}^n Y_i \sim \operatorname{Binomial}(n, p)",
        didactic_overview=(
            "A Distribuição Binomial modela o **número total de sucessos k em n tentativas independentes e identicamente "
            "distribuídas (i.i.d.) de Bernoulli**, onde cada tentativa tem probabilidade constante p de sucesso e q = 1 - p de fracasso."
        ),
        random_variable_meaning=(
            "A variável aleatória discreta X ∈ {0, 1, ..., n} conta quantos sucessos ocorreram nas n repetições independentes do experimento."
        ),
        primary_parameters="Número de tentativas n ∈ ℕ⁺ e Probabilidade de sucesso p ∈ [0, 1] (com q = 1 - p)",
        primary_parameters_latex=r"n \in \mathbb{N}^+ \text{ (tentativas)}, \quad p \in [0, 1] \text{ (probabilidade de sucesso em cada tentativa)}",
        default_params={"n": 10.0, "p": 0.5},
        pdf_or_pmf_name="FMP (Função Massa de Probabilidade — Discreta)",
        pdf_or_pmf_latex=r"P(X = k; n, p) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k \in \{0, 1, \dots, n\}",
        cdf_latex=r"F_X(k; n, p) = \sum_{j=0}^{\lfloor k \rfloor} \binom{n}{j} p^j (1 - p)^{n - j} = I_{1 - p}(n - \lfloor k \rfloor, \, \lfloor k \rfloor + 1)",
        mean_latex=r"\mathbb{E}[X] = np",
        mean_didactic="Como X é a soma de n variáveis Bernoulli(p) com média p, pela linearidade da esperança E[X] = n · p.",
        variance_latex=r"\operatorname{Var}(X) = np(1 - p) = npq",
        variance_didactic=(
            "Como as n tentativas são independentes e cada Bernoulli tem variância p(1-p), somamos n vezes obtendo np(1-p). "
            "A variância é máxima quando p = 0,5 (incerteza máxima) e zero quando p = 0 ou p = 1."
        ),
        median_latex=r"\operatorname{Mediana}(X) \in \{\lfloor np \rfloor, \lceil np \rceil\}",
        mode_latex=r"\operatorname{Moda}(X) = \lfloor (n + 1)p \rfloor",
        skewness_latex=r"\gamma_1 = \frac{1 - 2p}{\sqrt{np(1 - p)}} \quad (\text{Simétrica quando } p = 0.5)",
        excess_kurtosis_latex=r"\gamma_2 = \frac{1 - 6p(1 - p)}{np(1 - p)}",
        raw_kurtosis_latex=r"\beta_2 = 3 + \frac{1 - 6p(1 - p)}{np(1 - p)}",
        kurtosis_didactic="Para p = 0,5, o excesso de curtose é negativo γ₂ = -2/n (platicúrtica). Quando n → ∞, γ₂ → 0 (Teorema de De Moivre–Laplace).",
        raw_moment_k_latex=r"\mathbb{E}[X^m] = \sum_{j=0}^m \left\{ \begin{matrix} m \\ j \end{matrix} \right\} \frac{n!}{(n-j)!} p^j \quad \Big(\mathbb{E}[X]=np, \; \mathbb{E}[X^2]=np(1-p)+n^2p^2\Big)",
        central_moment_k_latex=r"\mu_2 = npq, \quad \mu_3 = npq(q - p), \quad \mu_4 = npq\left[1 + 3(n - 2)pq\right]",
        mgf_latex=r"M_X(t) = \mathbb{E}[e^{tX}] = \left(1 - p + p e^t\right)^n = \left(q + p e^t\right)^n, \quad t \in \mathbb{R}",
        cf_latex=r"\varphi_X(t) = \left(1 - p + p e^{it}\right)^n",
        moments_didactic_guide=(
            "Como cada tentativa Y_i ~ Bernoulli(p) tem FGM M_Y(t) = (1 - p) + p e^t, e X = Y₁ + ... + Y_n é a soma de n variáveis "
            "i.i.d., a FGM da soma é simplesmente o produto das FGMs: M_X(t) = [M_Y(t)]^n = (1 - p + p e^t)^n!"
        ),
        entropy_latex=r"H(X) = \frac{1}{2}\ln\left(2\pi e \, np(1-p)\right) + \mathcal{O}\left(\frac{1}{n}\right)",
        fisher_info_latex=r"\mathcal{I}(p) = \frac{n}{p(1 - p)}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Clássica por Número de Ensaios n e Probabilidade p",
                parameters=r"(n, p) \in \mathbb{N}^+ \times [0, 1]",
                canonical_writing_latex=r"X \sim \operatorname{Binomial}(n, p)",
                pdf_or_pmf_latex=r"P(X = k; n, p) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, \dots, n",
                cdf_latex=r"F_X(k; n, p) = \sum_{j=0}^{\lfloor k \rfloor} \binom{n}{j} p^j (1-p)^{n-j}",
                mean_latex=r"np",
                variance_latex=r"np(1-p)",
                kurtosis_latex=r"3 + \frac{1 - 6p(1-p)}{np(1-p)}",
                parameter_interpretations={
                    "n (Número Total de Repetições / Ensaios)": "Quantidade fixa de vezes que o experimento binário é repetido sob condições idênticas e independentes.",
                    "p (Probabilidade Individual de Sucesso)": "Chance de sucesso em cada tentativa isolada (0 ≤ p ≤ 1). O complementar q = 1 - p é a chance de fracasso.",
                },
                conversion_notes="Padrão universal em Probabilidade, SciPy (scipy.stats.binom(n, p)) e R (dbinom).",
            ),
            Parameterization(
                name="2. Caso Especial de Ensaio Único: Distribuição de Bernoulli (n = 1)",
                parameters=r"p \in [0, 1] \text{ com } n = 1",
                canonical_writing_latex=r"Y \sim \operatorname{Bernoulli}(p)",
                pdf_or_pmf_latex=r"P(Y = y; p) = p^y (1-p)^{1-y}, \quad y \in \{0, 1\}",
                cdf_latex=r"F_Y(y; p) = \begin{cases} 0 & y < 0 \\ 1 - p & 0 \le y < 1 \\ 1 & y \ge 1 \end{cases}",
                mean_latex=r"p",
                variance_latex=r"p(1-p)",
                kurtosis_latex=r"3 + \frac{1 - 6p(1-p)}{p(1-p)}",
                parameter_interpretations={
                    "p (Probabilidade de Sucesso Y = 1)": "Probabilidade de o indicador binário assumir o valor 1.",
                },
                conversion_notes="É o tijolo fundamental de onde nasce a Binomial através da soma i.i.d.",
            ),
            Parameterization(
                name="3. Parametrização Log-Chances / Logit (Regressão Logística GLM)",
                parameters=r"\eta = \ln\left(\frac{p}{1-p}\right) \in \mathbb{R}",
                canonical_writing_latex=r"X \sim \operatorname{Binomial}\left(n, \, \sigma(\eta)\right)",
                pdf_or_pmf_latex=r"P(X = k; n, \eta) = \binom{n}{k} \exp\left(\eta k - n\ln(1 + e^\eta)\right)",
                cdf_latex=r"F_X(k; n, \eta) = \sum_{j=0}^{\lfloor k \rfloor} \binom{n}{j} \frac{e^{\eta j}}{(1+e^\eta)^n}",
                mean_latex=r"n \frac{e^\eta}{1 + e^\eta}",
                variance_latex=r"n \frac{e^\eta}{(1 + e^\eta)^2}",
                kurtosis_latex=r"3 + \frac{1 - 6\sigma(\eta)(1-\sigma(\eta))}{n\sigma(\eta)(1-\sigma(\eta))}",
                parameter_interpretations={
                    "η = ln(p/(1-p)) (Log-Odds / Logito)": "Logaritmo da razão de chances (odds). Mapeia probabilidades de (0,1) para toda a reta real (-∞, +∞) via função sigmoide σ(η) = 1/(1+e^{-η}).",
                },
                conversion_notes="Forma canônica da Família Exponencial usada em Regressão Logística e Redes Neurais de Classificação.",
            ),
        ],
        iid_and_statistical_properties={
            "Construção via Soma i.i.d. de Bernoullis": r"Se $Y_1, \dots, Y_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Bernoulli}(p)$, então $X = \sum_{i=1}^n Y_i \sim \operatorname{Binomial}(n, p)$.",
            "Fechamento Aditivo (Mesma probabilidade p)": r"Se $X_1 \sim \operatorname{Binomial}(n_1, p)$ e $X_2 \sim \operatorname{Binomial}(n_2, p)$ são independentes com o MESMO $p$, então $X_1 + X_2 \sim \operatorname{Binomial}(n_1 + n_2, p)$.",
        },
        shape_signature={
            "is_discrete": True,
            "symmetric": True,
            "unimodal": True,
            "skewness_target": 0.0,
            "excess_kurtosis_target": -0.2,
            "support_type": "bounded_integers",
        },
    ),

    "geometric": DistributionSpec(
        id="geometric",
        name="Distribuição Geométrica",
        aliases=["geometric", "geometrica", "geom"],
        notation_patterns=[
            r"Geom\s*\(",
            r"Geometrica?\s*\(",
            r"\(1-p\)\^\{k-1\}p",
        ],
        formula_cues=["(1-p)^(k-1)*p", "(1-p)^k*p", "q^(k-1)*p"],
        family="Discreta",
        support="k in {1, 2, 3, ...} (tentativas) ou {0, 1, 2, ...} (fracassos)",
        support_latex=r"k \in \{1, 2, 3, \dots\} \text{ (tentativas)} \quad \text{ou} \quad y \in \{0, 1, 2, \dots\} \text{ (fracassos)}",
        canonical_single_rv_latex=r"X \sim \operatorname{Geom}(p)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Geom}(p)",
        didactic_overview=(
            "A Distribuição Geométrica modela o **número de tentativas independentes de Bernoulli necessárias até "
            "obter o PRIMEIRO sucesso** (ou o número de fracassos antes do primeiro sucesso). É o equivalente discreto "
            "da distribuição Exponencial e a única distribuição discreta com a propriedade de perda de memória."
        ),
        random_variable_meaning=(
            "A variável aleatória discreta X conta em qual tentativa ocorre o primeiro sucesso (X ∈ {1, 2, ...}) "
            "ou quantos fracassos ocorreram antes dele (Y = X - 1 ∈ {0, 1, ...})."
        ),
        primary_parameters="Probabilidade de sucesso p ∈ (0, 1] em cada tentativa (com q = 1 - p)",
        primary_parameters_latex=r"p \in (0, 1] \quad (\text{com } q = 1 - p)",
        default_params={"p": 0.35},
        pdf_or_pmf_name="FMP (Função Massa de Probabilidade — Discreta)",
        pdf_or_pmf_latex=r"P(X = k; p) = (1 - p)^{k - 1} p, \quad k \in \{1, 2, 3, \dots\}",
        cdf_latex=r"F_X(k; p) = 1 - (1 - p)^{\lfloor k \rfloor}, \quad k \ge 1",
        mean_latex=r"\mathbb{E}[X] = \frac{1}{p} \quad \left(\text{ou } \frac{1 - p}{p} \text{ na parametrização de fracassos}\right)",
        mean_didactic="Se a chance de acertar é p = 0,20 (1 em 5), esperamos precisar em média de 1/p = 5 tentativas até o primeiro acerto.",
        variance_latex=r"\operatorname{Var}(X) = \frac{1 - p}{p^2}",
        variance_didactic="A variância é idêntica nas duas parametrizações (pois Y = X - 1 é apenas um deslocamento constante de 1 unidade!).",
        median_latex=r"\operatorname{Mediana}(X) = \left\lceil \frac{-1}{\log_2(1 - p)} \right\rceil",
        mode_latex=r"\operatorname{Moda}(X) = 1 \quad (\text{ou } 0 \text{ na contagem de fracassos})",
        skewness_latex=r"\gamma_1 = \frac{2 - p}{\sqrt{1 - p}}",
        excess_kurtosis_latex=r"\gamma_2 = 6 + \frac{p^2}{1 - p}",
        raw_kurtosis_latex=r"\beta_2 = 9 + \frac{p^2}{1 - p}",
        kurtosis_didactic="O excesso de curtose é sempre maior que 6 (tende a 6 quando p → 0, convergindo para a Exponencial).",
        raw_moment_k_latex=r"\mathbb{E}[X] = \frac{1}{p}, \quad \mathbb{E}[X^2] = \frac{2 - p}{p^2}, \quad \mathbb{E}[X^m] = p \, \operatorname{Li}_{-m}(1 - p)",
        central_moment_k_latex=r"\mu_2 = \frac{1-p}{p^2}, \quad \mu_3 = \frac{(1-p)(2-p)}{p^3}, \quad \mu_4 = \frac{(1-p)(9 - 9p + p^2)}{p^4}",
        mgf_latex=r"M_X(t) = \frac{p e^t}{1 - (1 - p)e^t}, \quad \text{para } t < -\ln(1 - p)",
        cf_latex=r"\varphi_X(t) = \frac{p e^{it}}{1 - (1 - p)e^{it}}",
        moments_didactic_guide=(
            "A FGM vem diretamente da soma de uma Progressão Geométrica infinita de razão r = (1-p)e^t: "
            "E[e^{tX}] = ∑_{k=1}^∞ p e^t [(1-p)e^t]^(k-1) = p e^t / (1 - (1-p)e^t)."
        ),
        entropy_latex=r"H(X) = \frac{-(1 - p)\ln(1 - p) - p\ln p}{p}",
        fisher_info_latex=r"\mathcal{I}(p) = \frac{1}{p^2(1 - p)}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização de Tentativas até o 1º Sucesso (Suporte {1, 2, 3, ...})",
                parameters=r"p \in (0, 1], \quad k \in \{1, 2, 3, \dots\}",
                canonical_writing_latex=r"X \sim \operatorname{Geom}_1(p)",
                pdf_or_pmf_latex=r"P(X = k; p) = (1 - p)^{k - 1} p, \quad k = 1, 2, 3, \dots",
                cdf_latex=r"F_X(k; p) = 1 - (1 - p)^{\lfloor k \rfloor}, \quad k \ge 1",
                mean_latex=r"\frac{1}{p}",
                variance_latex=r"\frac{1 - p}{p^2}",
                kurtosis_latex=r"9 + \frac{p^2}{1 - p} \quad \left(\gamma_2 = 6 + \frac{p^2}{1-p}\right)",
                parameter_interpretations={
                    "p (Probabilidade de Sucesso por Tentativa)": "Chance de sucesso em cada tentativa independente; k inclui a tentativa final bem-sucedida.",
                },
                conversion_notes="Padrão em livros de Probabilidade e no Python SciPy (scipy.stats.geom(p)).",
            ),
            Parameterization(
                name="2. Parametrização de Fracassos antes do 1º Sucesso (Suporte {0, 1, 2, ...})",
                parameters=r"p \in (0, 1], \quad y = k - 1 \in \{0, 1, 2, \dots\}",
                canonical_writing_latex=r"Y \sim \operatorname{Geom}_0(p)",
                pdf_or_pmf_latex=r"P(Y = y; p) = (1 - p)^y p, \quad y = 0, 1, 2, \dots",
                cdf_latex=r"F_Y(y; p) = 1 - (1 - p)^{\lfloor y \rfloor + 1}, \quad y \ge 0",
                mean_latex=r"\frac{1 - p}{p}",
                variance_latex=r"\frac{1 - p}{p^2}",
                kurtosis_latex=r"9 + \frac{p^2}{1 - p} \quad \left(\gamma_2 = 6 + \frac{p^2}{1-p}\right)",
                parameter_interpretations={
                    "p (Probabilidade de Sucesso)": "Chance de encerrar a sequência de fracassos. Aqui y = 0 significa sucesso logo na 1ª tentativa (zero fracassos).",
                },
                conversion_notes="Usada na linguagem R (dgeom(y, prob=p)) e como caso r=1 da Binomial Negativa.",
            ),
        ],
        iid_and_statistical_properties={
            "Perda de Memória Discreta": r"$\mathbb{P}(X > m + n \mid X > m) = \mathbb{P}(X > n) = (1 - p)^n$.",
            "Soma de Amostra i.i.d. (Conexão com Binomial Negativa / Pascal)": r"Se $X_1, \dots, X_r \stackrel{\text{i.i.d.}}{\sim} \operatorname{Geom}(p)$, então $\sum_{i=1}^r X_i \sim \operatorname{BinomialNegativa}(r, p)$.",
        },
        shape_signature={
            "is_discrete": True,
            "symmetric": False,
            "unimodal": True,
            "monotone_decreasing": True,
            "skewness_target": 2.0,
            "excess_kurtosis_target": 6.1,
            "support_type": "positive_integers",
        },
    ),

    "cauchy": DistributionSpec(
        id="cauchy",
        name="Distribuição de Cauchy (ou Cauchy–Lorentz)",
        aliases=["cauchy", "lorentz", "cauchy-lorentz", "breit-wigner", "distribuicao de cauchy"],
        notation_patterns=[
            r"\bCauchy\s*\(",
            r"\operatorname\{Cauchy\}",
            r"\bLorentz\s*\(",
        ],
        formula_cues=[
            "1/(pi*gamma*(1+((x-x0)/gamma)^2))",
            "1/(\\pi(1+x^2))",
            "\\arctan",
            "(x-x_0)^2 + \\gamma^2",
        ],
        family="Contínua",
        support="x in (-infinito, +infinito)",
        support_latex=r"x \in (-\infty, +\infty) = \mathbb{R}",
        canonical_single_rv_latex=r"X \sim \operatorname{Cauchy}(x_0, \gamma)",
        canonical_iid_rv_latex=r"X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Cauchy}(x_0, \gamma)",
        didactic_overview=(
            "A Distribuição de Cauchy (também chamada de Lorentz ou Breit–Wigner na Física) é o exemplo clássico de "
            "distribuição de **caudas extremamente pesadas** (heavy-tailed): embora seu gráfico lembre um sino simétrico "
            "centrado em x₀, suas caudas decaem apenas como 1/x², tão lentamente que **sua média E[X], sua variância Var(X) "
            "e todos os seus momentos algébricos NÃO EXISTEM** (as integrais divergem)! Ela surge como a razão entre duas "
            "Normais Padrão independentes (Z₁/Z₂) ou como a posição em que um raio de luz emitido em ângulo uniforme atinge uma parede."
        ),
        random_variable_meaning=(
            "A variável aleatória contínua X ∈ ℝ representa uma razão entre variáveis Gaussianas independentes, "
            "a coordenada horizontal de um feixe emitido em ângulo aleatório uniforme, ou linhas espectrais de ressonância física."
        ),
        primary_parameters="Locação (Mediana e Moda) x₀ ∈ ℝ e Escala (Meia-Largura à Meia-Altura / HWHM) γ > 0",
        primary_parameters_latex=r"x_0 \in \mathbb{R} \text{ (locação: mediana e moda)}, \quad \gamma > 0 \text{ (escala: meia-largura à meia-altura)}",
        default_params={"x0": 0.0, "gamma": 1.0},
        pdf_or_pmf_name="FDP (Função Densidade de Probabilidade)",
        pdf_or_pmf_latex=r"f_X(x; x_0, \gamma) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]} = \frac{\gamma}{\pi\left[(x - x_0)^2 + \gamma^2\right]}, \quad x \in \mathbb{R}",
        cdf_latex=r"F_X(x; x_0, \gamma) = \mathbb{P}(X \le x) = \frac{1}{2} + \frac{1}{\pi}\arctan\!\left(\frac{x - x_0}{\gamma}\right), \quad x \in \mathbb{R}",
        mean_latex=r"\mathbb{E}[X] = \text{Indefinida / Não existe} \quad \left(\int_{-\infty}^{+\infty} |x|\,f_X(x)\,dx = +\infty\right)",
        mean_didactic=(
            "Apesar de a curva ser perfeitamente simétrica em torno de x₀, a integral de |x|/(1+x²) é logaritmicamente divergente "
            "nas duas caudas (+∞ - ∞). Por isso, a média populacional NÃO existe e o centro da distribuição deve ser medido pela Mediana (x₀)!"
        ),
        variance_latex=r"\operatorname{Var}(X) = \text{Indefinida / Não existe} \quad (\text{Usa-se a Amplitude Interquartil } \operatorname{IQR} = 2\gamma)",
        variance_didactic=(
            "Como o segundo momento ∫ x² f(x) dx diverge para +∞, a variância e o desvio-padrão não existem. "
            "Para medir a dispersão da Cauchy, utiliza-se o parâmetro de escala γ (onde os quartis são Q₁ = x₀ - γ e Q₃ = x₀ + γ, logo IQR = Q₃ - Q₁ = 2γ)."
        ),
        median_latex=r"\operatorname{Mediana}(X) = x_0 \quad \left(\text{pois } F_X(x_0) = \frac{1}{2} + \frac{1}{\pi}\arctan(0) = 0.5\right)",
        mode_latex=r"\operatorname{Moda}(X) = x_0 \quad \left(\text{com altura máxima } f_X(x_0) = \frac{1}{\pi\gamma}\right)",
        skewness_latex=r"\gamma_1 = \text{Indefinida (3º momento não existe; porém a distribuição é geometricamente simétrica em torno de } x_0\text{)}",
        excess_kurtosis_latex=r"\gamma_2 = \text{Indefinida (4º momento diverge — caudas polinomiais ultra-pesadas } \propto x^{-2}\text{)}",
        raw_kurtosis_latex=r"\beta_2 = \text{Indefinida (diverge)}",
        kurtosis_didactic=(
            "Embora a curtose baseada no 4º momento não exista, em termos de quantis a Cauchy possui caudas muito mais "
            "pesadas que a Normal, gerando outliers extremos com alta frequência."
        ),
        raw_moment_k_latex=r"\mathbb{E}[X^k] = \text{Não existe para nenhuma ordem inteira } k \ge 1 \quad \left(\mathbb{E}[|X|^r] < \infty \iff -1 < r < 1\right)",
        central_moment_k_latex=r"\mu_k = \mathbb{E}[(X - x_0)^k] = \text{Não existe para } k \ge 1",
        mgf_latex=r"M_X(t) = \mathbb{E}[e^{tX}] = +\infty \text{ para todo } t \neq 0 \quad (\text{A FGM só existe em } t = 0, \text{ onde } M_X(0) = 1)",
        cf_latex=r"\varphi_X(t) = \mathbb{E}[e^{itX}] = \exp\left(i x_0 t - \gamma |t|\right), \quad t \in \mathbb{R}",
        moments_didactic_guide=(
            "Por que a Função Característica φ_X(t) = exp(i x₀ t - γ|t|) existe para todo t ∈ ℝ mas os momentos não existem? "
            "Porque o módulo |t| NÃO é diferenciável na origem t = 0! Como os momentos seriam dados pelas derivadas φ^(k)(0) "
            "na origem, a quina de |t| em t = 0 reflete exatamente a inexistência de todos os momentos E[X^k] para k ≥ 1."
        ),
        entropy_latex=r"h(X) = \ln(4\pi\gamma)",
        fisher_info_latex=r"\mathcal{I}(x_0, \gamma) = \begin{pmatrix} \frac{1}{2\gamma^2} & 0 \\ 0 & \frac{1}{2\gamma^2} \end{pmatrix}",
        parameterizations=[
            Parameterization(
                name="1. Parametrização Geral de Locação x₀ e Escala γ (Padrão Estatístico)",
                parameters=r"(x_0, \gamma) \in \mathbb{R} \times (0, \infty)",
                canonical_writing_latex=r"X \sim \operatorname{Cauchy}(x_0, \gamma)",
                pdf_or_pmf_latex=r"f_X(x; x_0, \gamma) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]} = \frac{\gamma}{\pi\left[(x - x_0)^2 + \gamma^2\right]}",
                cdf_latex=r"F_X(x; x_0, \gamma) = \frac{1}{2} + \frac{1}{\pi}\arctan\!\left(\frac{x - x_0}{\gamma}\right)",
                mean_latex=r"\text{Indefinida (Mediana } = x_0\text{)}",
                variance_latex=r"\text{Indefinida (IQR } = 2\gamma\text{)}",
                kurtosis_latex=r"\text{Indefinida}",
                parameter_interpretations={
                    "x₀ (Parâmetro de Locação / Mediana e Moda)": (
                        "É o ponto exato do pico máximo da curva e o centro de simetria (Mediana e Moda). "
                        "Diferente da Normal, x₀ NÃO é a média, pois a média não existe!"
                    ),
                    "γ (Parâmetro de Escala / HWHM)": (
                        "É a Meia-Largura à Meia-Altura (em x = x₀ ± γ, a densidade cai exatamente à metade do pico: f(x₀ ± γ) = ½ f(x₀)). "
                        "Além disso, 50% da probabilidade total está contida no intervalo [x₀ - γ, x₀ + γ]."
                    ),
                },
                conversion_notes="Padrão em Probabilidade, Python SciPy (scipy.stats.cauchy(loc=x₀, scale=γ)) e R (dcauchy(x, location=x₀, scale=γ)).",
            ),
            Parameterization(
                name="2. Distribuição de Cauchy Padrão (x₀ = 0, γ = 1 — Caso ν = 1 da t de Student)",
                parameters=r"x_0 = 0, \quad \gamma = 1",
                canonical_writing_latex=r"X \sim \operatorname{Cauchy}(0, 1) \iff X \sim t_1",
                pdf_or_pmf_latex=r"f_X(x) = \frac{1}{\pi(1 + x^2)}, \quad x \in \mathbb{R}",
                cdf_latex=r"F_X(x) = \frac{1}{2} + \frac{1}{\pi}\arctan(x), \quad F_X^{-1}(p) = \tan\!\left(\pi\left(p - \frac{1}{2}\right)\right)",
                mean_latex=r"\text{Indefinida (Mediana } = 0\text{)}",
                variance_latex=r"\text{Indefinida (IQR } = 2\text{)}",
                kurtosis_latex=r"\text{Indefinida}",
                parameter_interpretations={
                    "Sem parâmetros livres (x₀ = 0, γ = 1)": (
                        "Coincide exatamente com a distribuição t de Student com ν = 1 grau de liberdade e com a razão Z₁/Z₂ de duas Normais(0,1) independentes."
                    ),
                },
                conversion_notes="Se U ~ Uniforme(0, 1), podemos simular X ~ Cauchy(0, 1) diretamente por inversão exata: X = tan(π(U - 1/2)).",
            ),
            Parameterization(
                name="3. Parametrização Física de Lorentz / Breit–Wigner (Ressonância por FWHM Γ = 2γ)",
                parameters=r"(E_0, \Gamma) \text{ onde } E_0 = x_0 \text{ e } \Gamma = 2\gamma > 0 \text{ é a Largura Total à Meia-Altura (FWHM)}",
                canonical_writing_latex=r"X \sim \operatorname{Lorentz}(E_0, \Gamma)",
                pdf_or_pmf_latex=r"f_X(x; E_0, \Gamma) = \frac{1}{2\pi} \frac{\Gamma}{(x - E_0)^2 + (\Gamma/2)^2}",
                cdf_latex=r"F_X(x; E_0, \Gamma) = \frac{1}{2} + \frac{1}{\pi}\arctan\!\left(\frac{2(x - E_0)}{\Gamma}\right)",
                mean_latex=r"\text{Indefinida (Pico de Ressonância } = E_0\text{)}",
                variance_latex=r"\text{Indefinida (Largura Total FWHM } = \Gamma\text{)}",
                kurtosis_latex=r"\text{Indefinida}",
                parameter_interpretations={
                    "E₀ (Energia / Frequência Central de Ressonância)": "Posição central do pico espectral.",
                    "Γ = 2γ (Largura Total à Meia-Altura — FWHM)": "Largura total da linha espectral na metade da intensidade máxima (inversamente proporcional ao tempo de vida da partícula/estado excitado).",
                },
                conversion_notes="Padrão em Física Quântica, Espectroscopia Óptica e Física de Partículas.",
            ),
        ],
        iid_and_statistical_properties={
            "Falha da Lei dos Grandes Números (Média Amostral i.i.d.)": r"Fato surpreendente: se $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \operatorname{Cauchy}(x_0, \gamma)$, a média amostral $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ tem **EXATAMENTE a mesma distribuição de uma única observação**: $\bar{X}_n \sim \operatorname{Cauchy}(x_0, \gamma)$! Tirar a média de $1.000.000$ de observações Cauchy não reduz a incerteza em nada (deve-se usar a **mediana amostral** para estimar $x_0$).",
            "Fechamento Aditivo (Estabilidade α = 1)": r"Se $X_1 \sim \operatorname{Cauchy}(x_1, \gamma_1)$ e $X_2 \sim \operatorname{Cauchy}(x_2, \gamma_2)$ são independentes, então $X_1 + X_2 \sim \operatorname{Cauchy}(x_1 + x_2, \, \gamma_1 + \gamma_2)$.",
            "Razão de Normais Independentes": r"Se $Z_1, Z_2 \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$, então o quociente $Z_1 / Z_2 \sim \operatorname{Cauchy}(0, 1)$.",
        },
        shape_signature={
            "is_discrete": False,
            "symmetric": True,
            "unimodal": True,
            "skewness_target": 0.0,
            "excess_kurtosis_target": 10.0,
            "support_type": "real_line",
        },
    ),
}


def get_distribution_by_id(dist_id: str) -> Optional[DistributionSpec]:
    return DISTRIBUTIONS_DB.get(dist_id.lower())


def list_all_distributions() -> List[DistributionSpec]:
    return list(DISTRIBUTIONS_DB.values())


# ================================================================================================
# SEÇÃO 2. MOTOR NUMÉRICO E AVALIADOR DE MOMENTOS/CURVAS (symbolic_engine.py)
# ================================================================================================

"""
Numeric & Symbolic Evaluator for Statistical Distributions.

Computes exact numeric values for Mean, Variance, Standard Deviation,
Median, Mode, Skewness, Excess Kurtosis, Raw Kurtosis, First 4 Raw & Central Moments,
Differential/Discrete Entropy, and evaluates PDF/PMF and CDF curves given parameters.
"""

from typing import Dict, Any, Tuple, List
import math


def compute_numeric_summary(dist_id: str, params: Dict[str, float]) -> Dict[str, Any]:
    """
    Given a distribution ID and numeric parameters (falling back to defaults),
    computes exact numeric statistical quantities.
    """
    spec = DISTRIBUTIONS_DB.get(dist_id)
    if not spec:
        return {}

    p = dict(spec.default_params)
    p.update({k: float(v) for k, v in params.items() if v is not None})

    res: Dict[str, Any] = {"parameters_used": p}

    if dist_id == "normal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 1.0), 1e-6)
        var = sigma ** 2
        res.update({
            "mean": mu,
            "variance": var,
            "std_dev": sigma,
            "median": mu,
            "mode": mu,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "raw_kurtosis": 3.0,
            "raw_moments": {
                "E[X^1]": mu,
                "E[X^2]": mu**2 + var,
                "E[X^3]": mu**3 + 3 * mu * var,
                "E[X^4]": mu**4 + 6 * (mu**2) * var + 3 * (var**2),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": 0.0,
                "mu_4": 3 * (var**2),
            },
            "entropy_nats": 0.5 * math.log(2 * math.pi * math.e * var),
        })

    elif dist_id == "exponential":
        lam = max(p.get("lam", 1.0), 1e-6)
        mean = 1.0 / lam
        var = 1.0 / (lam ** 2)
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": mean,
            "median": math.log(2.0) / lam,
            "mode": 0.0,
            "skewness": 2.0,
            "excess_kurtosis": 6.0,
            "raw_kurtosis": 9.0,
            "raw_moments": {
                "E[X^1]": 1.0 / lam,
                "E[X^2]": 2.0 / (lam**2),
                "E[X^3]": 6.0 / (lam**3),
                "E[X^4]": 24.0 / (lam**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": 1.0 / (lam**2),
                "mu_3": 2.0 / (lam**3),
                "mu_4": 9.0 / (lam**4),
            },
            "entropy_nats": 1.0 - math.log(lam),
        })

    elif dist_id == "uniform":
        a = p.get("a", 0.0)
        b = p.get("b", 1.0)
        if b <= a:
            b = a + 1.0
        L = b - a
        mean = 0.5 * (a + b)
        var = (L ** 2) / 12.0
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": mean,
            "mode": f"Any x in [{a:.3g}, {b:.3g}]",
            "skewness": 0.0,
            "excess_kurtosis": -1.2,
            "raw_kurtosis": 1.8,
            "raw_moments": {
                "E[X^1]": mean,
                "E[X^2]": (b**3 - a**3) / (3.0 * L),
                "E[X^3]": (b**4 - a**4) / (4.0 * L),
                "E[X^4]": (b**5 - a**5) / (5.0 * L),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": 0.0,
                "mu_4": (L**4) / 80.0,
            },
            "entropy_nats": math.log(L),
        })

    elif dist_id == "gamma":
        alpha = max(p.get("alpha", 3.0), 1e-3)
        beta = max(p.get("beta", 1.0), 1e-6)
        mean = alpha / beta
        var = alpha / (beta ** 2)
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": (alpha - 1.0 / 3.0 + 0.02 / alpha) / beta if alpha >= 1 else mean * 0.7,
            "mode": (alpha - 1.0) / beta if alpha >= 1.0 else 0.0,
            "skewness": 2.0 / math.sqrt(alpha),
            "excess_kurtosis": 6.0 / alpha,
            "raw_kurtosis": 3.0 + 6.0 / alpha,
            "raw_moments": {
                "E[X^1]": alpha / beta,
                "E[X^2]": alpha * (alpha + 1) / (beta**2),
                "E[X^3]": alpha * (alpha + 1) * (alpha + 2) / (beta**3),
                "E[X^4]": alpha * (alpha + 1) * (alpha + 2) * (alpha + 3) / (beta**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": 2.0 * alpha / (beta**3),
                "mu_4": 3.0 * alpha * (alpha + 2.0) / (beta**4),
            },
            "entropy_nats": alpha - math.log(beta) + math.lgamma(alpha),
        })

    elif dist_id == "beta":
        alpha = max(p.get("alpha", 2.0), 1e-3)
        beta = max(p.get("beta", 5.0), 1e-3)
        s = alpha + beta
        mean = alpha / s
        var = (alpha * beta) / ((s ** 2) * (s + 1.0))
        skew = (2.0 * (beta - alpha) * math.sqrt(s + 1.0)) / ((s + 2.0) * math.sqrt(alpha * beta))
        ex_kurt = (6.0 * (((alpha - beta)**2) * (s + 1.0) - alpha * beta * (s + 2.0))) / (
            alpha * beta * (s + 2.0) * (s + 3.0)
        )
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": (alpha - 1.0 / 3.0) / (s - 2.0 / 3.0) if (alpha > 1 and beta > 1) else mean,
            "mode": (alpha - 1.0) / (s - 2.0) if (alpha > 1 and beta > 1) else 0.5,
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": 3.0 + ex_kurt,
            "raw_moments": {
                "E[X^1]": mean,
                "E[X^2]": mean * (alpha + 1) / (s + 1),
                "E[X^3]": mean * (alpha + 1) * (alpha + 2) / ((s + 1) * (s + 2)),
                "E[X^4]": mean * (alpha + 1) * (alpha + 2) * (alpha + 3) / ((s + 1) * (s + 2) * (s + 3)),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": skew * (var ** 1.5),
                "mu_4": (ex_kurt + 3.0) * (var ** 2),
            },
            "entropy_nats": math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(s),
        })

    elif dist_id == "chi_square":
        k = max(p.get("k", 4.0), 1.0)
        mean = k
        var = 2.0 * k
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": k * ((1.0 - 2.0 / (9.0 * k)) ** 3),
            "mode": max(k - 2.0, 0.0),
            "skewness": math.sqrt(8.0 / k),
            "excess_kurtosis": 12.0 / k,
            "raw_kurtosis": 3.0 + 12.0 / k,
            "raw_moments": {
                "E[X^1]": k,
                "E[X^2]": k * (k + 2),
                "E[X^3]": k * (k + 2) * (k + 4),
                "E[X^4]": k * (k + 2) * (k + 4) * (k + 6),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": 2.0 * k,
                "mu_3": 8.0 * k,
                "mu_4": 12.0 * k * (k + 4.0),
            },
            "entropy_nats": 0.5 * k + math.log(2.0) + math.lgamma(0.5 * k),
        })

    elif dist_id == "student_t":
        nu = max(p.get("nu", 5.0), 0.5)
        mean = 0.0 if nu > 1 else "Undefined (ν ≤ 1)"
        var = (nu / (nu - 2.0)) if nu > 2 else ("Infinity" if nu > 1 else "Undefined")
        skew = 0.0 if nu > 3 else "Undefined (ν ≤ 3)"
        ex_kurt = (6.0 / (nu - 4.0)) if nu > 4 else ("Infinity" if nu > 2 else "Undefined")
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var) if isinstance(var, float) else var,
            "median": 0.0,
            "mode": 0.0,
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": (3.0 + ex_kurt) if isinstance(ex_kurt, float) else ex_kurt,
            "raw_moments": {
                "E[X^1]": 0.0 if nu > 1 else "Undefined",
                "E[X^2]": var,
                "E[X^3]": 0.0 if nu > 3 else "Undefined",
                "E[X^4]": (3.0 * (nu**2) / ((nu - 2.0) * (nu - 4.0))) if nu > 4 else "Undefined",
            },
            "central_moments": {
                "mu_1": 0.0 if nu > 1 else "Undefined",
                "mu_2": var,
                "mu_3": 0.0 if nu > 3 else "Undefined",
                "mu_4": (3.0 * (nu**2) / ((nu - 2.0) * (nu - 4.0))) if nu > 4 else "Undefined",
            },
            "entropy_nats": math.lgamma(0.5 * nu) + 0.5 * math.log(nu * math.pi) - math.lgamma(0.5 * (nu + 1)),
        })

    elif dist_id == "lognormal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 0.6), 1e-6)
        s2 = sigma ** 2

        def _safe_exp(val: float) -> Any:
            if val > 700.0:
                return f"e^{{{val:.4g}}} (muito grande)"
            try:
                return math.exp(val)
            except OverflowError:
                return f"e^{{{val:.4g}}} (muito grande)"

        mean = _safe_exp(mu + 0.5 * s2)
        if s2 <= 300.0 and (2.0 * mu + 2.0 * s2) <= 700.0:
            var: Any = (math.exp(s2) - 1.0) * math.exp(2.0 * mu + s2)
            std_dev: Any = math.sqrt(var)
        else:
            var = "Muito grande (overflow numérico)"
            std_dev = "Muito grande (overflow numérico)"
        if s2 <= 300.0:
            skew: Any = (math.exp(s2) + 2.0) * math.sqrt(math.exp(s2) - 1.0)
        else:
            skew = "Muito grande (overflow numérico)"
        if 4.0 * s2 <= 700.0:
            ex_kurt: Any = math.exp(4 * s2) + 2 * math.exp(3 * s2) + 3 * math.exp(2 * s2) - 6.0
            raw_kurt: Any = ex_kurt + 3.0
        else:
            ex_kurt = "Muito grande (overflow numérico)"
            raw_kurt = "Muito grande (overflow numérico)"

        mu3: Any = skew * (var ** 1.5) if (isinstance(skew, float) and isinstance(var, float) and var < 1e100) else "Muito grande (overflow numérico)"
        mu4: Any = (ex_kurt + 3.0) * (var ** 2) if (isinstance(ex_kurt, float) and isinstance(var, float) and var < 1e100) else "Muito grande (overflow numérico)"

        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": std_dev,
            "median": _safe_exp(mu),
            "mode": _safe_exp(mu - s2),
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": raw_kurt,
            "raw_moments": {
                "E[X^1]": _safe_exp(mu + 0.5 * s2),
                "E[X^2]": _safe_exp(2 * mu + 2.0 * s2),
                "E[X^3]": _safe_exp(3 * mu + 4.5 * s2),
                "E[X^4]": _safe_exp(4 * mu + 8.0 * s2),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": mu3,
                "mu_4": mu4,
            },
            "entropy_nats": mu + 0.5 * math.log(2 * math.pi * math.e * s2),
        })

    elif dist_id == "weibull":
        k = max(p.get("k", 2.0), 0.1)
        lam = max(p.get("lam", 1.0), 1e-6)
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        g3 = math.gamma(1.0 + 3.0 / k)
        g4 = math.gamma(1.0 + 4.0 / k)
        mean = lam * g1
        var = (lam ** 2) * (g2 - g1 ** 2)
        std = math.sqrt(max(var, 1e-12))
        mu3 = (lam ** 3) * (g3 - 3 * g1 * g2 + 2 * (g1 ** 3))
        mu4 = (lam ** 4) * (g4 - 4 * g1 * g3 + 6 * (g1 ** 2) * g2 - 3 * (g1 ** 4))
        skew = mu3 / (std ** 3)
        raw_kurt = mu4 / (var ** 2)
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": std,
            "median": lam * (math.log(2.0) ** (1.0 / k)),
            "mode": lam * (((k - 1.0) / k) ** (1.0 / k)) if k > 1.0 else 0.0,
            "skewness": skew,
            "excess_kurtosis": raw_kurt - 3.0,
            "raw_kurtosis": raw_kurt,
            "raw_moments": {
                "E[X^1]": lam * g1,
                "E[X^2]": (lam**2) * g2,
                "E[X^3]": (lam**3) * g3,
                "E[X^4]": (lam**4) * g4,
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": mu3,
                "mu_4": mu4,
            },
            "entropy_nats": 0.5772156649 * (1.0 - 1.0 / k) + math.log(lam / k) + 1.0,
        })

    elif dist_id == "poisson":
        lam = max(p.get("lam", 4.0), 1e-6)
        res.update({
            "mean": lam,
            "variance": lam,
            "std_dev": math.sqrt(lam),
            "median": math.floor(lam + 1.0 / 3.0 - 0.02 / lam),
            "mode": math.floor(lam),
            "skewness": 1.0 / math.sqrt(lam),
            "excess_kurtosis": 1.0 / lam,
            "raw_kurtosis": 3.0 + 1.0 / lam,
            "raw_moments": {
                "E[X^1]": lam,
                "E[X^2]": lam + lam**2,
                "E[X^3]": lam + 3 * (lam**2) + lam**3,
                "E[X^4]": lam + 7 * (lam**2) + 6 * (lam**3) + lam**4,
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": lam,
                "mu_3": lam,
                "mu_4": lam * (1.0 + 3.0 * lam),
            },
            "entropy_nats": 0.5 * math.log(2 * math.pi * math.e * lam) - 1.0 / (12.0 * lam),
        })

    elif dist_id == "binomial":
        n = max(int(round(p.get("n", 10.0))), 1)
        prob = min(max(p.get("p", 0.5), 1e-6), 1.0 - 1e-6)
        q = 1.0 - prob
        mean = n * prob
        var = n * prob * q
        skew = (q - prob) / math.sqrt(var)
        ex_kurt = (1.0 - 6.0 * prob * q) / var
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": round(mean),
            "mode": math.floor((n + 1) * prob),
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": 3.0 + ex_kurt,
            "raw_moments": {
                "E[X^1]": mean,
                "E[X^2]": var + mean**2,
                "E[X^3]": n * prob * ((n - 1) * (n - 2) * (prob**2) + 3 * (n - 1) * prob + 1),
                "E[X^4]": (ex_kurt + 3.0) * (var**2) + 4 * mean * (q - prob) * var + 6 * (mean**2) * var + (mean**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": var * (q - prob),
                "mu_4": var * (1.0 + 3.0 * (n - 2) * prob * q),
            },
            "entropy_nats": 0.5 * math.log(2 * math.pi * math.e * var),
        })

    elif dist_id == "geometric":
        prob = min(max(p.get("p", 0.35), 1e-6), 1.0 - 1e-6)
        q = 1.0 - prob
        mean = 1.0 / prob
        var = q / (prob ** 2)
        skew = (2.0 - prob) / math.sqrt(q)
        ex_kurt = 6.0 + (prob ** 2) / q
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": math.ceil(-1.0 / math.log2(q)),
            "mode": 1,
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": 3.0 + ex_kurt,
            "raw_moments": {
                "E[X^1]": 1.0 / prob,
                "E[X^2]": (2.0 - prob) / (prob**2),
                "E[X^3]": (1.0 + 4.0 * q + q**2) / (prob**3),
                "E[X^4]": (1.0 + 11.0 * q + 11.0 * (q**2) + q**3) / (prob**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": q * (2.0 - prob) / (prob**3),
                "mu_4": q * (9.0 - 9.0 * prob + prob**2) / (prob**4),
            },
            "entropy_nats": (-q * math.log(q) - prob * math.log(prob)) / prob,
        })

    elif dist_id == "cauchy":
        x0 = p.get("x0", 0.0)
        gamma = max(p.get("gamma", 1.0), 1e-6)
        res.update({
            "mean": f"Indefinida (Mediana = {x0:g})",
            "variance": f"Indefinida (IQR = {2.0 * gamma:g})",
            "std_dev": f"Indefinido (γ = {gamma:g})",
            "median": x0,
            "mode": x0,
            "skewness": "Indefinida (Simétrica em torno de x₀)",
            "excess_kurtosis": "Indefinida (Caudas Polinomiais ∝ 1/x²)",
            "raw_kurtosis": "Indefinida (∞)",
            "raw_moments": {
                "E[X^1]": "Indefinido",
                "E[X^2]": "Indefinido (∞)",
                "E[X^3]": "Indefinido",
                "E[X^4]": "Indefinido (∞)",
            },
            "central_moments": {
                "mu_1": "Indefinido",
                "mu_2": "Indefinido (∞)",
                "mu_3": "Indefinido",
                "mu_4": "Indefinido (∞)",
            },
            "entropy_nats": math.log(4.0 * math.pi * gamma),
        })

    return res


def evaluate_curve_points(dist_id: str, params: Dict[str, float], num_points: int = 200) -> Tuple[List[float], List[float], bool]:
    """
    Returns (x_values, y_values, is_discrete) for plotting or comparing PDF/PMF curves.
    Pure Python math implementation (no external dependencies required).
    """
    spec = DISTRIBUTIONS_DB.get(dist_id)
    p = dict(spec.default_params) if spec else {}
    p.update({k: float(v) for k, v in params.items() if v is not None})

    if dist_id == "normal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 1.0), 1e-6)
        xs = [mu - 4 * sigma + (8 * sigma * i / (num_points - 1)) for i in range(num_points)]
        ys = [(1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (((x - mu) / sigma) ** 2)) for x in xs]
        return xs, ys, False

    elif dist_id == "exponential":
        lam = max(p.get("lam", 1.0), 1e-6)
        x_max = 5.0 / lam
        xs = [x_max * i / (num_points - 1) for i in range(num_points)]
        ys = [lam * math.exp(-lam * x) for x in xs]
        return xs, ys, False

    elif dist_id == "uniform":
        a = p.get("a", 0.0)
        b = p.get("b", 1.0)
        if b <= a:
            b = a + 1.0
        L = b - a
        margin = 0.25 * L
        x_min, x_max = a - margin, b + margin
        raw_xs = [x_min + (x_max - x_min) * i / (num_points - 1) for i in range(num_points)]
        eps = 1e-6 * L
        xs = sorted(set(raw_xs + [a - eps, a, b, b + eps]))
        ys = [(1.0 / L) if (a <= x <= b) else 0.0 for x in xs]
        return xs, ys, False

    elif dist_id == "gamma":
        alpha = max(p.get("alpha", 3.0), 0.1)
        beta = max(p.get("beta", 1.0), 1e-6)
        x_max = (alpha + 4.5 * math.sqrt(alpha)) / beta
        x_start = 0.0 if alpha >= 1.0 else max(1e-4, 0.003 * x_max)
        xs = [x_start + (x_max - x_start) * i / (num_points - 1) for i in range(num_points)]
        log_norm = alpha * math.log(beta) - math.lgamma(alpha)
        ys: List[float] = []
        for x in xs:
            if x <= 0.0:
                ys.append(beta if abs(alpha - 1.0) < 1e-9 else 0.0)
            else:
                ys.append(math.exp(log_norm + (alpha - 1.0) * math.log(x) - beta * x))
        return xs, ys, False

    elif dist_id == "beta":
        alpha = max(p.get("alpha", 2.0), 0.1)
        beta = max(p.get("beta", 5.0), 0.1)
        x_start = 0.0 if alpha >= 1.0 else 0.005
        x_end = 1.0 if beta >= 1.0 else 0.995
        xs = [x_start + (x_end - x_start) * i / (num_points - 1) for i in range(num_points)]
        log_b = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        ys = []
        for x in xs:
            if x <= 0.0:
                ys.append((1.0 / math.exp(log_b)) if abs(alpha - 1.0) < 1e-9 else 0.0)
            elif x >= 1.0:
                ys.append((1.0 / math.exp(log_b)) if abs(beta - 1.0) < 1e-9 else 0.0)
            else:
                ys.append(math.exp((alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x) - log_b))
        return xs, ys, False

    elif dist_id == "chi_square":
        k = max(p.get("k", 4.0), 1.0)
        x_max = k + 4.5 * math.sqrt(2.0 * k)
        x_start = 0.0 if k >= 2.0 else 0.015
        xs = [x_start + (x_max - x_start) * i / (num_points - 1) for i in range(num_points)]
        log_norm = -0.5 * k * math.log(2.0) - math.lgamma(0.5 * k)
        ys = []
        for x in xs:
            if x <= 0.0:
                ys.append(0.5 if abs(k - 2.0) < 1e-9 else 0.0)
            else:
                ys.append(math.exp(log_norm + (0.5 * k - 1.0) * math.log(x) - 0.5 * x))
        return xs, ys, False

    elif dist_id == "student_t":
        nu = max(p.get("nu", 5.0), 1.0)
        xs = [-5.0 + 10.0 * i / (num_points - 1) for i in range(num_points)]
        norm_const = math.exp(math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)) / math.sqrt(nu * math.pi)
        ys = [norm_const * ((1.0 + (x ** 2) / nu) ** (-0.5 * (nu + 1.0))) for x in xs]
        return xs, ys, False

    elif dist_id == "lognormal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 0.6), 1e-6)
        x_max = math.exp(min(mu + 2.8 * sigma, mu + 7.5, 20.0))
        xs = [max(1e-3, x_max * i / (num_points - 1)) for i in range(num_points)]
        ys = [
            (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (((math.log(x) - mu) / sigma) ** 2))
            for x in xs
        ]
        return xs, ys, False

    elif dist_id == "weibull":
        k = max(p.get("k", 2.0), 0.5)
        lam = max(p.get("lam", 1.0), 1e-6)
        x_max = lam * 3.0
        xs = [max(1e-4, x_max * i / (num_points - 1)) for i in range(num_points)]
        ys = [(k / lam) * ((x / lam) ** (k - 1.0)) * math.exp(-((x / lam) ** k)) for x in xs]
        return xs, ys, False

    elif dist_id == "poisson":
        lam = max(p.get("lam", 4.0), 1e-4)
        k_max = max(10, int(math.ceil(lam + 4.2 * math.sqrt(lam))))
        xs = [float(k) for k in range(k_max + 1)]
        ys = [
            math.exp(-lam) if k == 0 else math.exp(k * math.log(lam) - lam - math.lgamma(k + 1.0))
            for k in xs
        ]
        return xs, ys, True

    elif dist_id == "binomial":
        n = max(int(round(p.get("n", 10.0))), 1)
        prob = min(max(p.get("p", 0.5), 1e-6), 1.0 - 1e-6)
        xs = [float(k) for k in range(n + 1)]
        log_p = math.log(prob)
        log_q = math.log(1.0 - prob)
        log_n_fact = math.lgamma(n + 1.0)
        ys = [
            math.exp(log_n_fact - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0) + k * log_p + (n - k) * log_q)
            for k in xs
        ]
        return xs, ys, True

    elif dist_id == "geometric":
        prob = min(max(p.get("p", 0.35), 1e-6), 1.0 - 1e-6)
        k_max = min(35, max(10, int(math.ceil(5.5 / prob))))
        xs = [float(k) for k in range(1, k_max + 1)]
        ys = [((1.0 - prob) ** (k - 1)) * prob for k in xs]
        return xs, ys, True

    elif dist_id == "cauchy":
        x0 = p.get("x0", 0.0)
        gamma = max(p.get("gamma", 1.0), 1e-6)
        xs = [x0 - 5.5 * gamma + (11.0 * gamma * i / (num_points - 1)) for i in range(num_points)]
        ys = [1.0 / (math.pi * gamma * (1.0 + (((x - x0) / gamma) ** 2))) for x in xs]
        return xs, ys, False

    return [], [], False


# ================================================================================================
# SEÇÃO 3. MOTOR DE OCR OFFLINE PARA ESCRITA MANUSCRITA (ocr_engine.py)
# ================================================================================================

"""
Motor de OCR Offline para Escrita Manuscrita e Parser de Notação Estatística.

Especializado em:
1. Ler imagens de notações estatísticas escritas à mão (ex.: "X_1, ..., X_n ~ i.i.d. N(μ, σ²)")
   usando Apple Vision Framework (macOS nativo), EasyOCR ou Tesseract.
2. Normalizar ruídos típicos de OCR em escrita manuscrita (como '~' lido como '-', '*', '∼',
   'i.i.d.' lido como 'i.i.d', 'iid', '1.1.d.', parênteses mal fechados, letras gregas).
3. Identificar exatamente:
   - Qual é a Variável Aleatória (ex.: X, Y, Z, ou sequência indexada X_1, ..., X_n)
   - Se há Independência (X ⊥ Y) ou se são Independentes e Identicamente Distribuídas (i.i.d.)
   - Qual é a Distribuição e seus Parâmetros (simbólicos ou numéricos)
   - Como a expressão DEVERIA ser escrita formalmente (notação tipográfica LaTeX canônica + dicas didáticas de correção da escrita manuscrita).
"""

import base64
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class StatisticalContextDetails:
    is_iid: bool = False
    iid_evidence: List[str] = field(default_factory=list)
    is_independent: bool = False
    independence_evidence: List[str] = field(default_factory=list)
    identically_distributed: bool = False
    primary_random_variable: str = "X"
    sample_base_letter: str = "X"
    is_indexed_sample: bool = False
    random_variables_detected: List[str] = field(default_factory=list)
    conditioning_detected: List[str] = field(default_factory=list)
    sample_size_n: Optional[int] = None
    notation_statements: List[str] = field(default_factory=list)
    extracted_parameters: Dict[str, float] = field(default_factory=dict)
    symbolic_parameters_detected: List[str] = field(default_factory=list)
    parameterization_hint: Optional[str] = None
    canonical_how_to_write_latex: str = ""
    handwriting_feedback: List[str] = field(default_factory=list)
    notation_corrections: List[str] = field(default_factory=list)
    image_parameters_latex: str = ""
    image_pdf_latex: str = ""
    image_cdf_latex: str = ""
    image_cdf_note: str = ""


@dataclass
class OCRAnalysisResult:
    raw_text: str
    cleaned_handwriting_text: str
    lines: List[str]
    backend_used: str
    distribution_scores: Dict[str, float]
    matched_cues: Dict[str, List[str]]
    statistical_context: StatisticalContextDetails


SWIFT_VISION_OCR_SCRIPT = r"""
import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count > 1 else {
    exit(1)
}
let imagePath = args[1]
guard let nsImage = NSImage(contentsOfFile: imagePath) else {
    exit(2)
}

// Desenha a imagem sobre um fundo branco sólido com margem de 40px em todos os lados
// para evitar problemas com PNGs transparentes ou texto encostado na borda da imagem
let origSize = nsImage.size
let pad: CGFloat = 40.0
let canvasSize = NSSize(width: max(origSize.width + pad * 2, 120), height: max(origSize.height + pad * 2, 100))
let paddedImage = NSImage(size: canvasSize)
paddedImage.lockFocus()
NSColor.white.setFill()
NSBezierPath.fill(NSRect(origin: .zero, size: canvasSize))
nsImage.draw(
    in: NSRect(x: pad, y: pad, width: origSize.width, height: origSize.height),
    from: .zero,
    operation: .sourceOver,
    fraction: 1.0
)
paddedImage.unlockFocus()

guard let cgImage = paddedImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    exit(3)
}

let request = VNRecognizeTextRequest { (request, error) in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    for observation in observations {
        if let topCandidate = observation.topCandidates(1).first {
            print(topCandidate.string)
        }
    }
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])
"""


class OfflineOCREngine:
    """
    Executa OCR local na imagem manuscrita e interpreta toda a notação estatística.
    Sincroniza e aprende automaticamente qualquer nova imagem colocada em `training_data/<dist_id>/`.
    """

    _cached_learned_data: Optional[Dict[str, Any]] = None
    _cached_dir_signature: Optional[Tuple[Tuple[str, int, int], ...]] = None

    KNOWN_TRAINING_TRANSCRIPTIONS: Dict[str, str] = {
        "beta/Beta_dist1.jpeg": "f(y) = (\\Gamma(a)\\Gamma(b) / \\Gamma(a+b)) y^{a-1}(1-y)^{b-1}, Y ~ Beta(a, b)",
        "beta/Beta_dist2.jpeg": "X ~ Beta(6, 7) | \\Gamma(6+7) / (\\Gamma(6)\\Gamma(7)) x^5 (1-x)^6",
        "beta/Beta_dist3.jpeg": "f(b) = (\\Gamma(\\theta_1+\\theta_2) / (\\Gamma(\\theta_1)\\Gamma(\\theta_2))) b^{\\theta_1-1}(1-b)^{\\theta_2-1} I(0 < b < 1), B ~ Beta(\\theta_1, \\theta_2)",
        "beta/Beta_dist4.jpeg": "A ~ Beta(1, 8), f(a) = (\\Gamma(9)/(\\Gamma(1)\\Gamma(8))) a^{1-1}(1-a)^{8-1}",
        "beta/Beta_dist5.png": "Y ~ Beta(1, 1) => f(y) = (\\Gamma(2)/(\\Gamma(1)\\Gamma(1))) y^0(1-y)^0 I_{(0,1)}(y)",
        "beta/Beta_dist6.png": "P ~ Beta(0.5, 0.5) => f_P(p) = (\\Gamma(1)/\\Gamma(1/2)^2) p^{-1/2}(1-p)^{-1/2} I_{(0,1)}(p)",
        "beta/Beta_dist7.png": "P ~ Beta(0.5, 0.5) => f_P(p) = (\\Gamma(1)/\\Gamma(1/2)^2) p^{-1/2}(1-p)^{-1/2} I_{(0,1)}(p)",
        "beta/Beta_dist8.png": "W ~ Beta(4, 2) => f(w) = 20 w^3 (1-w) I(0 < w < 1)",
        "beta/Beta_dist9.jpeg": "X ~ Beta(2, 3) => f(x) = (\\Gamma(5)/(\\Gamma(2)\\Gamma(3))) x(1-x)^2 I_{(0,1)}(x)",
        "binomial/Binomial_dist1.jpeg": "f(y) = C(n, y) p^n (1-p)^{n-y}, Y ~ Bin(n, p)",
        "binomial/Binomial_dist2.jpeg": "Y ~ Binomial(76, 0.25) | C(76, y) 0.25^y (1 - 0.25)^{76-y}",
        "binomial/Binomial_dist3.jpeg": "f(x) = C(n, x) p^x (1-p)^{n-x} I(x \\in \\{0, 1, ..., n\\}), X ~ Bin(n, p)",
        "binomial/Binomial_dist4.jpeg": "J ~ Binomial(15, 0.3), p(j) = C(15, j) (0.3)^j (0.7)^{15-j}",
        "binomial/Binomial_dist5.png": "Y ~ Bin(5, 0.2) => p(y) = C(5, y) (0.2)^y (0.8)^{5-y} I_{\\{0,...,5\\}}(y)",
        "binomial/Binomial_dist7.png": "K ~ Bin(20, 0.25) => P(K = k) = C(20, k) (1/4)^k (3/4)^{20-k} I(k \\in \\{0,...,20\\})",
        "binomial/Binomial_dist8.png": "W ~ Bin(8, 0.75) => p_W(w) = (8! / (w!(8-w)!)) (0.75)^w (0.25)^{8-w} I_{\\{0,...,8\\}}(w)",
        "binomial/Binomial_dist9.jpeg": "X ~ Bin(10, 0.5) => p(x) = C(10, x) (0.5)^{10} I_{\\{0,1,...,10\\}}(x)",
        "chi_square/Chi_square_dist2.jpeg": "X ~ Qui-Quadrado(9)",
        "chi_square/Chi_square_dist3.jpeg": "f(q) = ((1/2)^{g/2} / \\Gamma(g/2)) q^{g/2-1} e^{-q/2} I_{R_+}(q), Q ~ Qui-Quadrado(g)",
        "chi_square/Chi_Square_dist4.jpeg": "M ~ Qui-Quadrado(20), f(m) = ((1/2)^{20/2} / \\Gamma(20/2)) m^{20/2-1} e^{-m/2}",
        "chi_square/Chi_square_dist5.png": "X ~ Qui-Quadrado(1) => f(x) = ((1/2)^{1/2} / \\Gamma(1/2)) x^{-1/2} e^{-x/2} I_{(0,\\infty)}(x)",
        "chi_square/Chi_square_dist6.png": "Y ~ Qui-Quadrado(2) => f_Y(y) = (1/2) e^{-y/2} I_{R_+}(y)",
        "chi_square/Chi_square_dist7.png": "Y ~ Qui-Quadrado(2) => f_Y(y) = (1/2) e^{-y/2} I_{R_+}(y)",
        "chi_square/Chi_square_dist8.png": "W ~ Qui-Quadrado(10) => f(w) = ((1/2)^5 / \\Gamma(5)) w^4 e^{-w/2} I(w > 0)",
        "chi_square/Chi_square_dist9.jpeg": "Q ~ Qui-Quadrado(4) => f(q) = ((1/2)^2 / \\Gamma(2)) q e^{-q/2} I_{R_+}(q)",
        "chi_square/Chi_square_dist10.jpg": "R_1, R_2 ~ i.i.d. Qui-Quadrado(10)",
        "exponential/Exponential_dist.jpeg": "f(y) = \\lambda e^{-\\lambda y}, Y ~ Exp(\\lambda)",
        "exponential/Exponential_dist2.jpeg": "X ~ Exponencial(6) | 6 e^{-6x}",
        "exponential/Exponential_dist3.jpeg": "f(x) = \\lambda e^{-\\lambda x} I(x > 0), X ~ Exp(\\lambda)",
        "exponential/Exponential_dist4.jpeg": "K ~ Exponencial(7), f(k) = 7 e^{-7k}",
        "exponential/Exponential_dist5.png": "T ~ Exp(0.5) => f(t) = 0.5 e^{-0.5 t} I_{(0,\\infty)}(t)",
        "exponential/Exponential_dist7.png": "Y ~ Exp(1/3) => f_Y(y) = (1/3) e^{-y/3} I(y > 0)",
        "exponential/Eponential_dist8.png": "W ~ Exp(5) => f(w) = 5 \\exp(-5w) I_{R_+}(w)",
        "exponential/Exponential_dist9.jpeg": "X ~ Exp(2) => f(x) = 2 e^{-2x} I_{R_+}(x)",
        "gamma/Gama_dist1.jpeg": "f(y) = (\\beta^\\alpha / \\Gamma(\\alpha)) y^\\alpha e^{-\\beta y}, Y ~ Gama(\\alpha, \\beta)",
        "gamma/Gama_dist2.jpeg": "Y ~ Gama(15, 19) | (19^{15} / \\Gamma(15)) y^{14} e^{-19y}",
        "gamma/Gama_dist3.jpeg": "f(r) = (\\eta^\\psi / \\Gamma(\\psi)) r^{\\psi-1} e^{-\\eta r} I(r \\in R_+), R ~ Gama(\\psi, \\eta)",
        "gamma/Gama_dist4.jpeg": "Z ~ Gama(9, 3.5), f(z) = ((3.5)^9 / \\Gamma(9)) z^{9-1} e^{-3.5 z}",
        "gamma/Gama_dist5.png": "X ~ Gama(4, 1) => f(x) = (1 / \\Gamma(4)) x^3 e^{-x} I_{R_+}(x)",
        "gamma/Gama_dist6.png": "Y ~ Gama(2, 0.5) => f_Y(y) = (0.5)^2 y e^{-0.5 y} I_{(0,\\infty)}(y)",
        "gamma/Gama_dist7.png": "Y ~ Gama(2, 0.5) => f_Y(y) = (0.5)^2 y e^{-0.5 y} I_{(0,\\infty)}(y)",
        "gamma/Gama_dist8.png": "W ~ Gama(5, 3) => f(w) = (3^5 / \\Gamma(5)) w^4 e^{-3w} I_{R_+}(w)",
        "gamma/Gama_dist9.jpeg": "R ~ Gama(3, 2) => f(r) = (2^3 / \\Gamma(3)) r^2 e^{-2r} I(r \\in R_+)",
        "geometric/Geometric_dist2.jpeg": "Z ~ Geometrica(0.67) | 0.67 (1 - 0.67)^{z-1}",
        "geometric/Geometric_dist3.jpeg": "f(x) = (1-p)^{x-1} p I(x \\ge 1), X ~ Geometrica(p)",
        "geometric/Geometric_dist4.jpeg": "Q ~ Geometrica(0.4), p(q) = 0.4 (0.6)^{q-1}",
        "geometric/Geometric_dist5.png": "Y ~ Geo(0.5) => p(y) = (0.5)^y I_{N^*}(y)",
        "geometric/Geometric_dist6.png": "K ~ Geo(0.25) => P(K = k) = (1/4) (3/4)^{k-1} I_{\\{1,2,...\\}}(k)",
        "geometric/Geometric_dist7.png": "K ~ Geo(0.25) => P(K = k) = (1/4) (3/4)^{k-1} I_{\\{1,2,...\\}}(k)",
        "geometric/Geometric_dist8.png": "W ~ Geo(0.2) => p(w) = 0.2 (0.8)^w I_{\\{0,1,2,...\\}}(w)",
        "geometric/Geometric_dist9.jpeg": "X ~ Geo(0.3) => p(x) = 0.3 (0.7)^{x-1} I_{\\{1,2,...\\}}(x)",
        "normal/Normal_dist1.jpeg": "f(y) = (1 / \\sqrt{2\\pi\\sigma^2}) \\exp\\{-1/(2\\sigma^2) (y - \\mu)^2\\}, Y ~ Normal(\\mu, \\sigma^2)",
        "normal/Normal_dist2.jpeg": "Z ~ Normal(3, 4.35) | (1 / \\sqrt{2\\pi \\cdot 4.35}) \\exp(-1/8.70 (z - 3)^2)",
        "normal/Normal_dist3.jpeg": "f(k) = (1 / \\sqrt{2\\pi v^2}) \\exp\\{-1/(2v^2) (k - m)^2\\}, K ~ Normal(m, v^2)",
        "normal/Normal_dist4.jpeg": "N ~ Normal(-2, 7), f(n) = (1 / \\sqrt{14\\pi}) \\exp\\{-(n+2)^2 / 14\\}",
        "normal/Normal_dist5.png": "X ~ Normal(10, 4) => f(x) = (1 / \\sqrt{8\\pi}) \\exp(-(x-10)^2 / 8) I_R(x)",
        "normal/Normal_dist6.png": "Y ~ Normal(-3, 9) => f_Y(y) = (1 / (3\\sqrt{2\\pi})) e^{-(y+3)^2 / 18} I_R(y)",
        "normal/Normal_dist8.png": "W ~ Normal(50, 25) => f(w) = (1 / (5\\sqrt{2\\pi})) \\exp(-(w-50)^2 / 50) I(w \\in R)",
        "normal/Normal_dist9.jpeg": "Z ~ Normal(0, 1) => f(z) = (1 / \\sqrt{2\\pi}) e^{-z^2 / 2} I_R(z)",
        "poisson/Poisson_dist1.jpeg": "f(y) = (\\lambda^y e^{-\\lambda}) / y!, Y ~ Poisson(\\lambda)",
        "poisson/Poisson_dist2.jpeg": "W ~ Poisson(0.49) | e^{-0.49} (0.49)^w / w!",
        "poisson/Poisson_dist3.jpeg": "f(x) = e^{-r} (r^x / x!) I_N(x), X ~ Poisson(r)",
        "poisson/Poisson_dist4.jpeg": "E ~ Pois(5), p(e) = (e^{-5} 5^e) / e!",
        "poisson/Poisson_dist5.png": "Y ~ Pois(0.5) => p(y) = (e^{-0.5} (0.5)^y / y!) I_N(y)",
        "poisson/Poisson_dist6.png": "K ~ Pois(10) => P(K = k) = (10^k e^{-10} / k!) I_{\\{0,1,...\\}}(k)",
        "poisson/Poisson_dist7.png": "K ~ Pois(10) => P(K = k) = (10^k e^{-10} / k!) I_{\\{0,1,...\\}}(k)",
        "poisson/Poisson_dist8.png": "W ~ Pois(2.5) => p_W(w) = (e^{-2.5} (2.5)^w / w!) I(w \\in N)",
        "poisson/Poisson_dist9.jpeg": "X ~ Pois(3) => p(x) = (e^{-3} 3^x / x!) I_{\\{0,1,2,...\\}}(x)",
        "poisson/Poisson_dist10.png": "E_1, E_2, ..., E_i ~ i.i.d. Poisson(4,5), i \\in [0, n]",
        "student_t/Student_t_dist2.jpeg": "T ~ t-student(22) | (\\Gamma(23/2)/\\Gamma(22/2)) (1/\\sqrt{22\\pi}) (1 + t^2/22)^{-23/2}",
        "student_t/student_t_dist3.jpeg": "f(v) = (\\Gamma((a+1)/2) / (\\sqrt{\\pi a}\\Gamma(a/2))) (1 + v^2/a)^{-(a+1)/2} I_R(v), V ~ t_a",
        "student_t/student_t_dist4.jpeg": "X ~ t_{n-1}, f(x) = (\\Gamma(n/2) / (\\sqrt{\\pi(n-1)}\\Gamma((n-1)/2))) (1 + x^2/(n-1))^{-n/2}",
        "student_t/student_t_dist5.png": "T ~ t(1) => f(t) = (\\Gamma(1) / (\\sqrt{\\pi}\\Gamma(1/2))) (1 + t^2)^{-1} I_R(t)",
        "student_t/student_t_dist6.png": "X ~ t(10) => f_X(x) = (\\Gamma(5.5) / (\\sqrt{10\\pi}\\Gamma(5))) (1 + x^2/10)^{-5.5} I_R(x)",
        "student_t/student_t_dist7.png": "V ~ t(5) => f(v) = (\\Gamma(3) / (\\sqrt{5\\pi}\\Gamma(5/2))) (1 + v^2/5)^{-3} I_R(v)",
        "student_t/student_t_dist8.png": "Y ~ t(3) => f(y) = (\\Gamma(2) / (\\sqrt{3\\pi}\\Gamma(3/2))) (1 + y^2/3)^{-2} I(y \\in R)",
        "student_t/student_t_dist9.jpeg": "V ~ t(5) => f(v) = (\\Gamma(3) / (\\sqrt{5\\pi}\\Gamma(5/2))) (1 + v^2/5)^{-3} I_R(v)",
        "uniform/Uniform_dist1.jpeg": "f(y) = 1 / (b - a), Y ~ U(a, b)",
        "uniform/Uniform_dist2.jpeg": "U ~ Uniforme[0.22, 0.89] | 1 / (0.89 - 0.22)",
        "uniform/Uniform_dist3.jpeg": "f(u) = (1 / (c - b)) I_{(c,b)}(u), U ~ U(b, c)",
        "uniform/Uniform_dist4.jpeg": "C ~ Unif(1, 2), f(c) = 1 / (2 - 1)",
        "uniform/Uniform_dist5.png": "Y ~ U(-2, 2) => f(y) = (1/4) I_{[-2,2]}(y)",
        "uniform/Uniform_dist6.png": "U ~ U(5, 15) => f_U(u) = (1/10) I_{(5,15)}(u)",
        "uniform/Uniform_dist7.png": "U ~ U(5, 15) => f_U(u) = (1/10) I_{(5,15)}(u)",
        "uniform/Uniform_dist8.png": "W ~ U(0, 10) => f(w) = 0.1 I(0 \\le w \\le 10)",
        "uniform/Uniform_dist9.jpeg": "X ~ U(0, 1) => f(x) = I_{[0,1]}(x)",
    }

    _rapidocr_engine: Any = None
    _rapidocr_checked: bool = False
    _easyocr_reader: Any = None
    _easyocr_checked: bool = False

    @classmethod
    def _run_rapidocr(cls, image_path: str) -> Optional[str]:
        """
        Executa RapidOCR (PP-OCRv4 ONNX multiplataforma, leve e determinístico em CPU).
        Garante o MESMO resultado em Windows, macOS e Linux, reconstruindo linhas de leitura
        da esquerda para a direita a partir das caixas delimitadoras e normalizando pontuação full-width.
        """
        if not cls._rapidocr_checked:
            cls._rapidocr_checked = True
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
                cls._rapidocr_engine = RapidOCR(box_thresh=0.3, text_score=0.3, unclip_ratio=1.8)
            except Exception:
                cls._rapidocr_engine = None
        if cls._rapidocr_engine is None:
            return None
        try:
            result, _ = cls._rapidocr_engine(image_path)
            if not result:
                return None

            def _to_ascii_punct(s: str) -> str:
                return "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)

            boxes: List[Tuple[float, float, float, str]] = []
            for entry in result:
                if len(entry) < 2:
                    continue
                box, raw_txt = entry[0], _to_ascii_punct(str(entry[1]).strip())
                if not raw_txt:
                    continue
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
                yc = 0.5 * (min(ys) + max(ys))
                x1 = min(xs)
                h = max(1.0, max(ys) - min(ys))
                boxes.append((yc, x1, h, raw_txt))

            if not boxes:
                return None

            hs = sorted(b[2] for b in boxes)
            med_h = hs[len(hs) // 2]
            rows: List[List[Tuple[float, float, float, str]]] = []
            for b in sorted(boxes, key=lambda t: (t[0], t[1])):
                placed = False
                for row in rows:
                    row_yc = sum(r[0] for r in row) / len(row)
                    if abs(b[0] - row_yc) <= max(med_h * 0.75, 18.0):
                        row.append(b)
                        placed = True
                        break
                if not placed:
                    rows.append([b])

            rows.sort(key=lambda row: sum(r[0] for r in row) / len(row))
            lines: List[str] = []
            for row in rows:
                row.sort(key=lambda r: r[1])
                lines.append(" ".join(r[3] for r in row))
            return "\n".join(lines) if lines else None
        except Exception:
            return None

    @classmethod
    def _run_easyocr(cls, image_path: str) -> Optional[str]:
        if not cls._easyocr_checked:
            cls._easyocr_checked = True
            try:
                import easyocr  # type: ignore
                cls._easyocr_reader = easyocr.Reader(["pt", "en"], gpu=False, verbose=False)
            except Exception:
                cls._easyocr_reader = None
        if cls._easyocr_reader is None:
            return None
        try:
            results = cls._easyocr_reader.readtext(image_path, detail=0)
            return "\n".join(results) if results else None
        except Exception:
            return None

    def extract_text(self, image_path: str) -> Tuple[str, str]:
        if not os.path.exists(image_path):
            return "", "arquivo_nao_encontrado"

        sidecar_text = self._read_png_text_chunks(image_path)

        # 1. Motor unificado multiplataforma (RapidOCR ONNX — garante o MESMO resultado em Windows, macOS e Linux!)
        rapid_txt = self._run_rapidocr(image_path)
        if rapid_txt:
            if sidecar_text and sidecar_text not in rapid_txt:
                rapid_txt = rapid_txt + "\n" + sidecar_text
            return rapid_txt, "RapidOCR (Rede Neural ONNX Multiplataforma)"

        # 2. Fallback: EasyOCR se instalado (instância em cache)
        easy_txt = self._run_easyocr(image_path)
        if easy_txt:
            if sidecar_text and sidecar_text not in easy_txt:
                easy_txt = easy_txt + "\n" + sidecar_text
            return easy_txt, "EasyOCR (Rede Neural Local para Manuscritos)"

        # 3. Fallback nativo do macOS (Apple Vision Framework)
        if sys.platform == "darwin" and shutil.which("swift"):
            try:
                text = self._run_macos_vision_ocr(image_path)
                if text:
                    if sidecar_text and sidecar_text not in text:
                        text = text + "\n" + sidecar_text
                    return text, "Apple Vision Framework (OCR Manuscrito Nativo macOS)"
            except Exception:
                pass

        # 4. Fallback nativo do Windows 10/11 (Windows.Media.Ocr via PowerShell)
        if sys.platform == "win32" and shutil.which("powershell"):
            try:
                text = self._run_windows_native_ocr(image_path)
                if text:
                    if sidecar_text and sidecar_text not in text:
                        text = text + "\n" + sidecar_text
                    return text, "Windows Native OCR (WinRT PowerShell)"
            except Exception:
                pass

        # 5. Tenta Tesseract CLI se instalado
        tesseract_bin = shutil.which("tesseract")
        if tesseract_bin:
            try:
                win_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
                proc = subprocess.run(
                    [tesseract_bin, image_path, "stdout", "--psm", "6"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    creationflags=win_flags,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    combined = proc.stdout.strip()
                    if sidecar_text and sidecar_text not in combined:
                        combined = combined + "\n" + sidecar_text
                    return combined, "Tesseract OCR (Motor Local CLI)"
            except Exception:
                pass

        if sidecar_text:
            return sidecar_text, "Camada de Metadados + OCR Geométrico"

        return "", "Sem camada de texto detectada"

    @staticmethod
    def _run_windows_native_ocr(image_path: str) -> Optional[str]:
        abs_path = os.path.abspath(image_path).replace("'", "''")
        ps_script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
            "Add-Type -AssemblyName System.Runtime.WindowsRuntime\n"
            "$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]\n"
            "$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType=WindowsRuntime]\n"
            "$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]\n"
            "$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]\n"
            "$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]\n"
            "$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime]\n"
            "$null = [Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType=WindowsRuntime]\n"
            "$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { "
            "$_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' "
            "})[0]\n"
            "function Await($WinRtTask, $ResultType) {\n"
            "  $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)\n"
            "  $netTask = $asTask.Invoke($null, @($WinRtTask))\n"
            "  $netTask.Wait(-1) | Out-Null\n"
            "  $netTask.Result\n"
            "}\n"
            f"$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{abs_path}')) ([Windows.Storage.StorageFile])\n"
            "$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])\n"
            "try {\n"
            "  $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])\n"
            "  $bmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])\n"
            "  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()\n"
            "  if ($null -eq $engine) {\n"
            "    $langs = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages\n"
            "    if ($langs.Count -gt 0) { $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($langs[0]) }\n"
            "  }\n"
            "  if ($null -ne $engine) {\n"
            "    $res = Await ($engine.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])\n"
            "    foreach ($line in $res.Lines) { Write-Output $line.Text }\n"
            "  }\n"
            "} finally {\n"
            "  if ($null -ne $stream) { $stream.Dispose() }\n"
            "}\n"
        )
        encoded_ps = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
        win_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-EncodedCommand", encoded_ps],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                creationflags=win_flags,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
        return None

    def _run_macos_vision_ocr(self, image_path: str) -> Optional[str]:
        cache_dir = os.path.join(tempfile.gettempdir(), "stat_vision_ocr_cache_v2")
        os.makedirs(cache_dir, exist_ok=True)
        bin_path = os.path.join(cache_dir, "macos_vision_handwriting_bin")
        swift_path = os.path.join(cache_dir, "macos_vision_handwriting.swift")

        if not os.path.exists(bin_path):
            with open(swift_path, "w", encoding="utf-8") as f:
                f.write(SWIFT_VISION_OCR_SCRIPT)
            try:
                subprocess.run(
                    ["swiftc", "-O", swift_path, "-o", bin_path],
                    capture_output=True,
                    text=True,
                    timeout=25,
                )
            except Exception:
                pass

        cmd = [bin_path, os.path.abspath(image_path)] if os.path.exists(bin_path) else ["swift", swift_path, os.path.abspath(image_path)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def compute_soft_ink_field(grid: List[List[float]]) -> Tuple[List[List[float]], int, int]:
        """
        Remove gradientes de iluminação do papel (sombra de câmera/celular) usando
        um mapa bilinear de fundo local 12x6 e extrai a intensidade contínua da tinta
        (funciona tanto para escrita a lápis fraco quanto caneta forte).
        """
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        if w == 0 or h == 0:
            return [], 0, 0

        by, bx = 6, 12
        bg_map = [[240.0] * bx for _ in range(by)]
        for iy in range(by):
            y0, y1 = int(iy * h / by), max(int((iy + 1) * h / by), int(iy * h / by) + 1)
            for ix in range(bx):
                x0, x1 = int(ix * w / bx), max(int((ix + 1) * w / bx), int(ix * w / bx) + 1)
                vals = []
                step_y = max(1, (y1 - y0) // 10)
                step_x = max(1, (x1 - x0) // 10)
                for yy in range(y0, y1, step_y):
                    row = grid[yy]
                    for xx in range(x0, x1, step_x):
                        vals.append(row[xx])
                vals.sort()
                bg_map[iy][ix] = vals[int(len(vals) * 0.78)] if vals else 240.0

        ink_grid = [[0.0] * w for _ in range(h)]
        ink_vals: List[float] = []
        mx0, mx1 = int(w * 0.03), max(int(w * 0.97), int(w * 0.03) + 1)
        my0, my1 = int(h * 0.04), max(int(h * 0.96), int(h * 0.04) + 1)

        for y in range(my0, my1):
            fy = (y / float(h)) * by - 0.5
            iy0 = max(0, min(by - 1, int(math.floor(fy))))
            iy1 = max(0, min(by - 1, iy0 + 1))
            wy = max(0.0, min(1.0, fy - iy0))
            row = grid[y]
            irow = ink_grid[y]
            for x in range(mx0, mx1):
                fx = (x / float(w)) * bx - 0.5
                ix0 = max(0, min(bx - 1, int(math.floor(fx))))
                ix1 = max(0, min(bx - 1, ix0 + 1))
                wx = max(0.0, min(1.0, fx - ix0))
                bg = (1.0 - wy) * ((1.0 - wx) * bg_map[iy0][ix0] + wx * bg_map[iy0][ix1]) + wy * (
                    (1.0 - wx) * bg_map[iy1][ix0] + wx * bg_map[iy1][ix1]
                )
                v = max(0.0, bg - row[x])
                irow[x] = v
                if (x + y) % 2 == 0:
                    ink_vals.append(v)

        ink_vals.sort()
        n = len(ink_vals)
        mean_i = sum(ink_vals) / max(n, 1)
        std_i = math.sqrt(sum((v - mean_i) ** 2 for v in ink_vals) / max(n, 1))
        p99 = ink_vals[int(n * 0.99)] if n else 20.0
        noise_floor = mean_i + 1.15 * std_i
        scale = max(p99 - noise_floor, 4.0)

        soft_ink = [[0.0] * w for _ in range(h)]
        for y in range(my0, my1):
            irow = ink_grid[y]
            srow = soft_ink[y]
            for x in range(mx0, mx1):
                val = irow[x] - noise_floor
                if val > 0.0:
                    srow[x] = min(1.0, val / scale)
        return soft_ink, w, h

    @staticmethod
    def extract_region_ink_vector(
        soft_ink: List[List[float]], x0: int, y0: int, x1: int, y1: int
    ) -> List[float]:
        """
        Extrai o vetor descritor visual (249 dimensões, normalizado L2 e invariante a
        translação/pan e escala/zoom através da caixa delimitadora justa da tinta).
        """
        xs: List[int] = []
        ys: List[int] = []
        for y in range(max(0, y0), min(len(soft_ink), y1)):
            row = soft_ink[y]
            for x in range(max(0, x0), min(len(row), x1)):
                if row[x] > 0.25:
                    xs.append(x)
                    ys.append(y)
        if len(xs) < 15:
            return []

        xs.sort()
        ys.sort()
        tx0, tx1 = xs[int(len(xs) * 0.02)], xs[int(len(xs) * 0.98)]
        ty0, ty1 = ys[int(len(ys) * 0.02)], ys[int(len(ys) * 0.98)]
        bw = max(tx1 - tx0 + 1, 8)
        bh = max(ty1 - ty0 + 1, 8)

        gx, gy = 16, 8
        cells = [0.0] * (gx * gy)
        counts = [0] * (gx * gy)
        for y in range(ty0, min(len(soft_ink), ty0 + bh)):
            cy = min(gy - 1, int((y - ty0) * gy / bh))
            row = soft_ink[y]
            for x in range(tx0, min(len(row), tx0 + bw)):
                cx = min(gx - 1, int((x - tx0) * gx / bw))
                idx = cy * gx + cx
                cells[idx] += row[x]
                counts[idx] += 1

        cell_means = [cells[i] / max(counts[i], 1) for i in range(gx * gy)]
        cm_avg = sum(cell_means) / len(cell_means)
        cm_std = math.sqrt(sum((v - cm_avg) ** 2 for v in cell_means) / len(cell_means)) + 1e-6
        vec = [(v - cm_avg) / cm_std for v in cell_means]

        for cy in range(gy):
            for cx in range(gx - 1):
                vec.append((cell_means[cy * gx + cx + 1] - cell_means[cy * gx + cx]) / cm_std)

        aspect = math.log(max(bw / float(bh), 0.1))
        vec.append(aspect * 1.5)
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-8
        return [round(v / norm, 5) for v in vec]

    @classmethod
    def compute_image_visual_signatures(cls, image_path: str) -> Dict[str, List[float]]:
        """
        Gera assinaturas visuais da imagem inteira e de sub-janelas (centro, topo, base,
        esquerda, direita) para garantir reconhecimento mesmo quando o usuário dá zoom ou
        move a imagem dentro do retângulo de análise.
        """
        if not image_path or not os.path.exists(image_path):
            return {}
        try:
            w, h, _, grid = PurePythonPNGReader.read_grayscale_grid(image_path)
            soft_ink, mw, mh = cls.compute_soft_ink_field(grid)
            if mw == 0 or mh == 0:
                return {}
            regions = {
                "full": (0, 0, mw, mh),
                "center": (int(mw * 0.06), int(mh * 0.06), int(mw * 0.94), int(mh * 0.94)),
                "left_part": (0, 0, int(mw * 0.68), mh),
                "right_part": (int(mw * 0.32), 0, mw, mh),
                "top_part": (0, 0, mw, int(mh * 0.62)),
                "bottom_part": (0, int(mh * 0.32), mw, mh),
            }
            out: Dict[str, List[float]] = {}
            for r_name, (rx0, ry0, rx1, ry1) in regions.items():
                vec = cls.extract_region_ink_vector(soft_ink, rx0, ry0, rx1, ry1)
                if vec:
                    out[r_name] = vec
            return out
        except Exception:
            return {}

    @staticmethod
    def _parse_numeric_or_fraction(val_str: str) -> float:
        s = val_str.strip().replace(",", ".")
        if "/" in s:
            parts = s.split("/", 1)
            return float(parts[0]) / max(float(parts[1]), 1e-9)
        if "-" in s[1:]:
            parts = s.split("-", 1)
            return float(parts[0]) - float(parts[1])
        return float(s)

    @staticmethod
    def _compute_file_sha256(image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _resolve_training_dir() -> str:
        """
        Localiza a pasta `training_data` de forma 100% dinâmica em QUALQUER sistema operacional
        (Windows, macOS ou Linux) e em qualquer modo de execução (pacote Python, arquivo único
        consolidado, `StatMath.app` ou executável PyInstaller), sem nenhum caminho fixo de máquina.
        """
        search_roots: List[str] = []
        if getattr(sys, "frozen", False):
            # Executável empacotado (ex.: PyInstaller .exe / .app)
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            search_roots.extend([exe_dir, os.path.dirname(exe_dir)])
            meipass = getattr(sys, "_MEIPASS", "")
            if meipass:
                search_roots.append(meipass)
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            search_roots.extend([here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))])
        except NameError:
            pass
        search_roots.extend([os.getcwd(), os.path.dirname(os.getcwd())])

        seen = set()
        candidates: List[str] = []
        for root in search_roots:
            if not root:
                continue
            cand = os.path.normpath(os.path.join(root, "training_data"))
            if cand not in seen:
                seen.add(cand)
                candidates.append(cand)

        for cand in candidates:
            if os.path.isdir(cand):
                return cand
        return candidates[0] if candidates else os.path.join(os.getcwd(), "training_data")

    @classmethod
    def _load_learned_data(cls) -> Dict[str, Any]:
        """
        Carrega o banco de padrões treinados (`learned_patterns.json`) e sincroniza
        AUTOMATICAMENTE qualquer nova imagem adicionada às subpastas de `training_data/<dist_id>/`.
        Assim, toda imagem colocada em `training_data/` já treina a IA automaticamente!
        """
        training_dir = cls._resolve_training_dir()
        learned_file = os.path.join(training_dir, "learned_patterns.json")
        if not os.path.isdir(training_dir):
            return {}

        # Calcula assinatura rápida do diretório (nome relativo, tamanho, mtime)
        dir_items: List[Tuple[str, int, int]] = []
        for dist_id in sorted(DISTRIBUTIONS_DB.keys()):
            folder = os.path.join(training_dir, dist_id)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    fpath = os.path.join(folder, fname)
                    try:
                        st = os.stat(fpath)
                        dir_items.append((f"{dist_id}/{fname}", int(st.st_size), int(st.st_mtime)))
                    except OSError:
                        pass
        current_dir_sig = tuple(dir_items)

        if cls._cached_learned_data is not None and cls._cached_dir_signature == current_dir_sig:
            return cls._cached_learned_data

        try:
            with open(learned_file, "r", encoding="utf-8") as f:
                learned_data = json.load(f)
        except Exception:
            learned_data = {"learned_ocr_aliases": {}, "learned_visual_signatures": [], "training_history": []}

        learned_aliases: Dict[str, List[str]] = learned_data.setdefault("learned_ocr_aliases", {})
        visual_list: List[Dict[str, Any]] = learned_data.setdefault("learned_visual_signatures", [])
        by_file: Dict[str, Dict[str, Any]] = {
            entry.get("file", ""): entry for entry in visual_list if entry.get("file")
        }

        active_rel_keys = {item[0] for item in dir_items}
        changed = False

        # Remove assinaturas de arquivos que foram apagados da pasta training_data
        if any(k not in active_rel_keys for k in list(by_file.keys())):
            visual_list = [e for e in visual_list if e.get("file") in active_rel_keys]
            by_file = {e.get("file", ""): e for e in visual_list if e.get("file")}
            changed = True

        # Treina automaticamente qualquer imagem nova ou modificada em training_data/<dist_id>/
        for rel_key, fsize, fmtime in dir_items:
            existing = by_file.get(rel_key)
            fpath = os.path.join(training_dir, rel_key)
            if (
                existing is not None
                and existing.get("file_size") == fsize
                and existing.get("signatures")
            ):
                # Atualiza transcrição se estiver no catálogo de referência
                ref_tr = cls.KNOWN_TRAINING_TRANSCRIPTIONS.get(rel_key)
                if ref_tr and existing.get("transcription") != ref_tr:
                    existing["transcription"] = ref_tr
                    changed = True
                if not existing.get("sha256"):
                    existing["sha256"] = cls._compute_file_sha256(fpath)
                    changed = True
                continue

            dist_id = rel_key.split("/", 1)[0]
            spec = DISTRIBUTIONS_DB.get(dist_id)
            if not spec:
                continue
            sigs = cls.compute_image_visual_signatures(fpath)
            if not sigs:
                continue
            sidecar_ocr = cls._read_png_text_chunks(fpath)
            auto_ocr = ""
            if rel_key not in cls.KNOWN_TRAINING_TRANSCRIPTIONS and not sidecar_ocr.strip():
                try:
                    raw_auto, _ = cls().extract_text(fpath)
                    auto_ocr = cls.normalize_handwritten_ocr(raw_auto).strip()
                except Exception:
                    auto_ocr = ""
            transcription = (
                cls.KNOWN_TRAINING_TRANSCRIPTIONS.get(rel_key)
                or sidecar_ocr.strip()
                or auto_ocr
                or spec.canonical_single_rv_latex
            )
            new_entry = {
                "dist_id": dist_id,
                "file": rel_key,
                "file_size": fsize,
                "file_mtime": fmtime,
                "sha256": cls._compute_file_sha256(fpath),
                "transcription": transcription,
                "signatures": sigs,
            }
            if rel_key in by_file:
                idx = visual_list.index(by_file[rel_key])
                visual_list[idx] = new_entry
            else:
                visual_list.append(new_entry)
            by_file[rel_key] = new_entry

            first_line = transcription.splitlines()[0].strip()
            if len(first_line) >= 4:
                alias_list = learned_aliases.setdefault(dist_id, [])
                if first_line not in alias_list:
                    alias_list.append(first_line)
            changed = True

        learned_data["learned_visual_signatures"] = visual_list
        if changed:
            try:
                with open(learned_file, "w", encoding="utf-8") as f:
                    json.dump(learned_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        cls._cached_learned_data = learned_data
        cls._cached_dir_signature = current_dir_sig
        return learned_data

    @classmethod
    def _load_learned_patterns(cls) -> Dict[str, List[str]]:
        return cls._load_learned_data().get("learned_ocr_aliases", {})

    @classmethod
    def match_learned_visual_signature(cls, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Compara a imagem analisada (ou o recorte com zoom/pan feito pelo usuário)
        contra o banco de assinaturas visuais aprendidas da pasta `training_data/`.
        Também calcula a votação k-NN por classe (`class_knn_scores`) para generalizar
        melhor o entendimento visual de novas imagens manuscritas.
        """
        if not image_path or not os.path.exists(image_path):
            return None
        learned_data = cls._load_learned_data()
        visual_db: List[Dict[str, Any]] = learned_data.get("learned_visual_signatures", [])
        if not visual_db:
            return None

        query_sha256 = cls._compute_file_sha256(image_path)
        query_sigs: Dict[str, List[float]] = {}
        if query_sha256:
            for e in visual_db:
                if e.get("sha256") == query_sha256 and e.get("signatures"):
                    query_sigs = e["signatures"]
                    break
        if not query_sigs:
            query_sigs = cls.compute_image_visual_signatures(image_path)
        query_full = query_sigs.get("full") or query_sigs.get("center")
        if not query_full and not query_sha256:
            return None

        best_entry: Optional[Dict[str, Any]] = None
        best_sim = -1.0
        second_best_sim = -1.0
        best_dist_id = ""
        per_class_sims: Dict[str, List[float]] = {d: [] for d in DISTRIBUTIONS_DB}

        for entry in visual_db:
            if query_sha256 and entry.get("sha256") == query_sha256:
                max_sim = 1.0
            else:
                entry_sigs: Dict[str, List[float]] = entry.get("signatures", {})
                entry_full = entry_sigs.get("full")
                if not query_full or not entry_full or len(entry_full) != len(query_full):
                    continue

                # Similaridade principal (full vs full e center vs full)
                sim_full = sum(a * b for a, b in zip(query_full, entry_full))
                max_sim = sim_full
                if "center" in query_sigs and len(query_sigs["center"]) == len(entry_full):
                    sim_c = sum(a * b for a, b in zip(query_sigs["center"], entry_full))
                    if sim_c > max_sim:
                        max_sim = sim_c

                # Também compara contra sub-regiões do treino caso o usuário tenha dado zoom forte em parte da fórmula
                for sub_key in ("center", "left_part", "right_part", "top_part", "bottom_part"):
                    sub_vec = entry_sigs.get(sub_key)
                    if sub_vec and len(sub_vec) == len(query_full):
                        sub_sim = sum(a * b for a, b in zip(query_full, sub_vec)) - 0.04
                        if sub_sim > max_sim:
                            max_sim = sub_sim

            d_id = entry.get("dist_id", "")
            if d_id in per_class_sims:
                per_class_sims[d_id].append(max_sim)

            if max_sim > best_sim:
                if d_id != best_dist_id:
                    second_best_sim = best_sim
                best_sim = max_sim
                best_entry = entry
                best_dist_id = d_id
            elif d_id != best_dist_id and max_sim > second_best_sim:
                second_best_sim = max_sim

        # Calcula pontuação k-NN estrutural por família de distribuição (média dos top-3 protótipos mais próximos)
        class_knn_scores: Dict[str, float] = {}
        for d_id, sim_list in per_class_sims.items():
            if not sim_list:
                class_knn_scores[d_id] = 0.0
            else:
                top_k = sorted(sim_list, reverse=True)[:3]
                avg_top3 = sum(top_k) / len(top_k)
                class_knn_scores[d_id] = round(max(0.0, top_k[0] * 0.7 + avg_top3 * 0.3), 4)

        if best_entry and best_sim >= 0.38:
            return {
                "dist_id": best_entry["dist_id"],
                "file": best_entry.get("file", ""),
                "similarity": round(best_sim, 4),
                "margin": round(best_sim - max(0.0, second_best_sim), 4),
                "transcription": best_entry.get("transcription", ""),
                "class_knn_scores": class_knn_scores,
            }
        return None

    @staticmethod
    def _read_png_text_chunks(image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                return ""
            pos = 8
            texts = []
            while pos + 8 <= len(data):
                length = int.from_bytes(data[pos:pos + 4], "big")
                chunk_type = data[pos + 4:pos + 8]
                chunk_data = data[pos + 8:pos + 8 + length]
                if chunk_type == b"tEXt":
                    parts = chunk_data.split(b"\x00", 1)
                    if len(parts) == 2:
                        val = parts[1].decode("utf-8", errors="ignore")
                        texts.append(val)
                pos += 12 + length
            return "\n".join(texts)
        except Exception:
            return ""

    @staticmethod
    def normalize_handwritten_ocr(raw_text: str) -> str:
        """
        Corrige confusões comuns de OCR (Apple Vision, EasyOCR, Tesseract) quando o usuário
        escreve notação estatística e funções densidade de probabilidade à mão:
        - '1.1.d' ou 'l.l.d' -> 'i.i.d.'
        - '∼' ou '≈' -> '~'
        - Funções indicadoras I_{(0,1)}(y) mal lidas como 'N(0,1)(y)', 'U(0,1)(y)', '11(0,1)(y)', 'II(0,1)(y)'
        - Dígitos manuscritos dentro de argumentos: 'l'/'I'/'|' -> '1', 'O' -> '0', 'S' -> '5'
        - Função Gama manuscrita: 'r(2)', 'T(5)', 'Γ(2)' -> '\\Gamma(2)', '\\Gamma(5)'
        - Notações Qui-Quadrado e t-Student manuscritas: '~ X2 10' -> '~ Qui-Quadrado(10)', '~ t10' -> '~ t(10)'
        """
        if not raw_text:
            return ""
        # Converte pontuação full-width Unicode (ex.: '（', '）', '，', '；') para ASCII padrão
        t = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in raw_text)
        t = t.replace("∼", "~").replace("≈", "~").replace("˜", "~")
        t = re.sub(r"\b[1lI]\.[1lI]\.d\.?\b", "i.i.d.", t, flags=re.IGNORECASE)
        t = re.sub(r"\biid\b", "i.i.d.", t, flags=re.IGNORECASE)

        # 1. Corrige funções indicadoras I_{(a,b)}(y) no final da fórmula mal lidas como N(0,1)(y), U(0,1)(y), 11(0,1)(y), II(0,1)(y)
        t = re.sub(
            r"(?:\bII|\b11|\bll|\|\||\bI|\bN|\bU)\s*_?\s*\{?\s*([\(\[])\s*([-+]?\d*[\.,]?\d+)\s*[,;]\s*([-+]?\d*[\.,]?\d+|inf|\\infty|∞)\s*([\)\]])\s*\}?\s*\(\s*([a-z])\s*\)",
            r"I_{\1\2,\3\4}(\5)",
            t,
            flags=re.IGNORECASE,
        )
        # Se N(0,1) ou 11(0,1) ou II(0,1) aparece à direita de '=' dentro de uma equação f(x) = ... (sem '~' antes), é a indicadora I_{(0,1)}
        cleaned_lines: List[str] = []
        for line in t.splitlines():
            if "=" in line:
                lhs, rhs = line.split("=", 1)
                if "~" not in rhs and "\\sim" not in rhs:
                    rhs = re.sub(
                        r"(?:\bII|\b11|\bll|\|\||\bN)\s*_?\s*\{?\s*([\(\[])\s*([-+]?\d*[\.,]?\d+)\s*[,;]\s*([-+]?\d*[\.,]?\d+)\s*([\)\]])\s*\}?",
                        r"I_{\1\2,\3\4}",
                        rhs,
                        flags=re.IGNORECASE,
                    )
                line = lhs + "=" + rhs
            cleaned_lines.append(line)
        t = "\n".join(cleaned_lines)

        # 2. Corrige decimais e frações manuscritas ('O.5' -> '0.5', '0.S' -> '0.5', 'O.S' -> '0.5', 'l/2' -> '1/2')
        t = re.sub(r"\b[Oo]\s*[\.,]\s*[Ss]\b", "0.5", t)
        t = re.sub(r"\b[Oo]\s*[\.,]\s*(\d+)\b", r"0.\1", t)
        t = re.sub(r"\b(\d+)\s*[\.,]\s*[Ss]\b", r"\1.5", t)
        t = re.sub(r"\b(\d+)\s*[\.,]\s*[Oo]\b", r"\1.0", t)
        t = re.sub(r"\b[lI\|]\s*/\s*(\d+)\b", r"1/\1", t)
        t = re.sub(r"\b(\d+)\s*/\s*[lI\|]\b", r"\1/1", t)
        t = re.sub(r"\(\s*[lI\|]\s*-\s*([a-z])\s*\)", r"(1-\1)", t)

        # 2b. Normaliza grafias cursivas de nomes de distribuições mal lidas pelo Apple Vision
        # Ex.: "Poinon (4,5)", "loisson (4,5)", "toisson (4,5)", "Poissan (4,5)", "Poimon (4,5)" -> "Poisson(4,5)"
        t = re.sub(
            r"\b(?:[ptlcrfb1I\|]o[i1l](?:ss|s|n|nn|m|rs|rr)[oa][nm]|p[o0][i1l]s(?:s[o0][nm])?|paisson)\b",
            "Poisson",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r"\b(?:n[o0]r(?:m|in|n)a[l1I]|n[o0]mal)\b", "Normal", t, flags=re.IGNORECASE)
        t = re.sub(
            r"\b(?:[b8][i1l]n[o0](?:m|in|n)[i1l]?a[l1I]|[b8][i1l]n[o0]m)\b",
            "Binomial",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"\b(?:e[x]?p[o0]nenc[i1l]?a[l1I]|e[x]?p[o0]nent[i1l]?a[l1I])\b",
            "Exponencial",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"\b(?:[g6]e[o0](?:m|in|n)e[t1l]r[i1l]ca?)\b",
            "Geometrica",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r"\b(?:un[i1l]f[o0]r(?:m|in|n)e?)\b", "Uniforme", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(?:c[ao]uch[yv1li]|c[ao]vch[yv]|l[o0]rentz)\b", "Cauchy", t, flags=re.IGNORECASE)

        # 2c. Normaliza 'iid' empilhado sobre til lido como 'id', 'iid', 'i.d' antes do nome da distribuição
        t = re.sub(
            r"(?:\s*~\s*)?\b(?:i\.?\s*i\.?\s*d\.?|i\.?\s*d\.?|iid|id|üd)\b(?:\s*~\s*)?(?=\s*(?:Beta|Bela|Bcta|8eta|Beto|Gama|Gamma|6ama|Cama|Ganma|Erlang|Bin|B1n|8in|Bim|Binom|Binomial|Bern|Bernoulli|Pois|P0is|Po1s|Pols|Poisson|Paisson|Exp|Exponencial|Exponential|Eponential|Ex9|Geo|6eo|Ge0|Gco|Geom|Geometrica|Geométrica|Geometric|N|Norm|Normal|N0rmal|U|Unif|Un1f|Uniforme|Uniform|Chi2|Chisq|Qui-?Quadrado|Chi-?Square|t-?student|student|t|Weibull|Lognormal|Log-?Normal|Lnorm|Cauchy|Lorentz)\b)",
            " ~ i.i.d. ",
            t,
            flags=re.IGNORECASE,
        )

        # 2d. Normaliza declarações de domínio de índice mal lidas pelo OCR (ex.: "i € l0, m]" -> "i \in [0, n]")
        def _normalize_index_domain(m: re.Match) -> str:
            idx_var = m.group(1)
            l_raw = m.group(2)
            l_brk = "[" if l_raw in ("[", "l", "I", "|") else ("{" if l_raw == "{" else "(")
            s_raw = m.group(3)
            start_val = "0" if s_raw in ("0", "O", "o") else ("1" if s_raw in ("1", "l", "I") else s_raw)
            end_raw = m.group(4).strip()
            end_val = "n" if end_raw in ("n", "m", "N", "M") else end_raw
            r_raw = m.group(5)
            r_brk = "]" if r_raw in ("]", "l", "I", "|") else ("}" if r_raw == "}" else ")")
            return f"{idx_var} \\in {l_brk}{start_val}, {end_val}{r_brk}"

        t = re.sub(
            r"\b([ijk])\s*(?:\\in|∈|€|\bem\b|\bin\b|[eE])\s*([\[\(\{lI\|])\s*([01Oo])\s*[,;]\s*(?:(?:\.{2,}|…)\s*[,;]\s*)?([nmNM]|\d+)\s*([\]\)\}lI\|])",
            _normalize_index_domain,
            t,
        )

        # 2e. Normaliza sequências de variáveis aleatórias indexadas com subscritos ruidosos
        # Ex.: "Ex, E.,..., E, ~ i.i.d." -> "E_1, E_2, ..., E_i ~ i.i.d."
        def _normalize_indexed_rv_seq(m: re.Match) -> str:
            letter = m.group(1)
            last_sub_raw = (m.group(2) or "").strip()
            has_i_domain = bool(re.search(r"\bi\s*(?:\\in|∈|=)", t))
            if last_sub_raw.lower() == "i" or (not last_sub_raw and has_i_domain):
                end_sub = "i"
            else:
                end_sub = "n"
            return f"{letter}_1, {letter}_2, ..., {letter}_{end_sub}"

        t = re.sub(
            r"\b([A-Z])(?:_?[1lIx]|\.)?\s*,\s*\1(?:_?[2z]|\.)?\s*,\s*(?:\.{2,}|…|\\dots)\s*,\s*\1(?:_?([iInN])\b|[,\.])?",
            _normalize_indexed_rv_seq,
            t,
        )

        # 2f. Normaliza listas finitas de variáveis aleatórias indexadas sem sublinhado (ex.: "R1, R2" -> "R_1, R_2", "X1, X2, X3" -> "X_1, X_2, X_3")
        t = re.sub(
            r"\b([A-Z])(\d)\s*,\s*\1(\d)(?:\s*,\s*\1(\d))?\b",
            lambda m: f"{m.group(1)}_{m.group(2)}, {m.group(1)}_{m.group(3)}" + (f", {m.group(1)}_{m.group(4)}" if m.group(4) else ""),
            t,
        )

        # 2g. Normaliza Qui-Quadrado manuscrito quebrado em linhas ou lido horizontalmente pelo OCR
        # Ex.: "R_1, R_2\nX 2\n10" -> "R_1, R_2 ~ Qui-Quadrado(10)"
        # Ex.: "Q X 7 2" (onde o OCR leu Q ~ \chi^2_7 na ordem horizontal) -> "Q ~ Qui-Quadrado(7)"
        t = re.sub(
            r"(\b[A-Z]_\d(?:\s*,\s*[A-Z]_\d)*|\b[A-Z]\b)\s*(?:~|\\sim)?\s*(?:i\.?i\.?d\.?)?\s*[\r\n]+\s*(?:[+xXχ]|\\chi)\s*\^?\s*2\s*[\r\n\s]+(\d+(?:[\.,]\d+)?)\b",
            r"\1 ~ Qui-Quadrado(\2)",
            t,
        )
        t = re.sub(
            r"(\b[A-Z]\b)\s+(?:~|\\sim)?\s*[xXχ]\s+(\d{1,3})\s+2\b",
            r"\1 ~ Qui-Quadrado(\2)",
            t,
        )

        # 3. Normaliza dígitos manuscritos ('l'/'I'/'|'/'/' -> '1', 'O' -> '0', 'S' -> '5') e vírgula decimal brasileira (ex.: Poisson(4,5) -> Poisson(4.5))
        dist_fn_pat = re.compile(
            r"(\b(?:Beta|Bela|Bcta|8eta|Beto|Gama|Gamma|6ama|Cama|Ganma|Erlang|Bin|B1n|8in|Bim|Binom|Binomial|Bern|Bernoulli|Pois|P0is|Po1s|Pols|Poisson|Paisson|Exp|Exponencial|Exponential|Eponential|Ex9|Geo|6eo|Ge0|Gco|Geom|Geometrica|Geométrica|Geometric|N|Norm|Normal|N0rmal|U|Unif|Un1f|Uniforme|Uniform|Chi2|Chisq|Qui-?Quadrado|Chi-?Square|t-?student|student|t|Weibull|Lognormal|Log-?Normal|Lnorm|Cauchy|Lorentz|C|r|T|F|Γ)|\\Gamma)\s*([\(\[])([^\)\]]+)([\)\]])",
            flags=re.IGNORECASE,
        )
        one_param_fn_pat = re.compile(
            r"^(?:Pois|P0is|Po1s|Pols|Poisson|Paisson|Exp|Exponencial|Exponential|Eponential|Ex9|"
            r"Geo|6eo|Ge0|Gco|Geom|Geometrica|Geométrica|Geometric|Bern|Bernoulli|"
            r"Chi2|Chisq|Qui-?Quadrado|Chi-?Square|t-?student|student|t|r|T|F|Γ|\\Gamma)$",
            flags=re.IGNORECASE,
        )

        def _clean_arg_token(tok: str) -> str:
            s = tok.strip()
            if s in ("l", "I", "|", "/"):
                return "1"
            if s in ("O", "o"):
                return "0"
            if s in ("S", "s"):
                return "5"
            s = re.sub(r"^[lI\|](\d+)$", r"1\1", s)
            s = re.sub(r"^(\d+)[Oo]$", r"\g<1>0", s)
            s = re.sub(r"^(\d+)[Ss]$", r"\g<1>5", s)
            s = re.sub(r"^[Oo][\.,](\d+)$", r"0.\1", s)
            s = re.sub(r"^[Oo][\.,][Ss]$", "0.5", s)
            s = re.sub(r"^(\d+)[\.,][Ss]$", r"\1.5", s)
            s = re.sub(r"^[lI\|]\s*/\s*(\d+)$", r"1/\1", s)
            return s

        def _replace_dist_args(m: re.Match) -> str:
            fn_name, lp, inner, rp = m.group(1), m.group(2), m.group(3), m.group(4)
            if one_param_fn_pat.match(fn_name):
                # Em distribuições de 1 único parâmetro (ex.: Poisson(4,5), Exp(2,5), Geo(0,4)),
                # a vírgula entre dois números inteiros é o separador decimal brasileiro!
                raw_parts = [_clean_arg_token(p) for p in inner.split(",")]
                if len(raw_parts) == 2 and re.match(r"^[-+]?\d+$", raw_parts[0]) and re.match(r"^\d+$", raw_parts[1]):
                    return f"{fn_name}{lp}{raw_parts[0]}.{raw_parts[1]}{rp}"
                return f"{fn_name}{lp}{_clean_arg_token(inner)}{rp}"

            if ";" in inner:
                parts = [_clean_arg_token(p.replace(",", ".")) for p in inner.split(";")]
                return f"{fn_name}{lp}{', '.join(parts)}{rp}"

            comma_parts = [_clean_arg_token(p) for p in inner.split(",")]
            if len(comma_parts) == 4:
                p1 = f"{comma_parts[0]}.{comma_parts[1]}"
                p2 = f"{comma_parts[2]}.{comma_parts[3]}"
                return f"{fn_name}{lp}{p1}, {p2}{rp}"
            if len(comma_parts) == 3:
                if comma_parts[0] in ("0", "-0"):
                    p1 = f"{comma_parts[0]}.{comma_parts[1]}"
                    p2 = comma_parts[2]
                else:
                    p1 = comma_parts[0]
                    p2 = f"{comma_parts[1]}.{comma_parts[2]}"
                return f"{fn_name}{lp}{p1}, {p2}{rp}"

            return f"{fn_name}{lp}{', '.join(comma_parts)}{rp}"

        t = dist_fn_pat.sub(_replace_dist_args, t)

        # 4. Normaliza função Gama manuscrita: 'Γ(...)' ou 'r(2)', 'T(6)' -> '\Gamma(...)'
        t = re.sub(r"Γ\s*\(", r"\\Gamma(", t)
        t = re.sub(
            r"(?<!~)\b[rTF]\s*\(\s*(\d+(?:[\.,]\d+)?(?:\s*[+/]\s*\d+(?:[\.,]\d+)?)?)\s*\)",
            r"\\Gamma(\1)",
            t,
        )

        # 5. Normaliza Qui-Quadrado e t-Student após '~' ('~ X2 10' -> '~ Qui-Quadrado(10)', '~ t10' -> '~ t(10)')
        t = re.sub(
            r"(~|\\sim)\s*(?:\\chi|[xXχ])\s*\^?\s*2\s*_?\s*[\(\[\{]?\s*(\d+(?:[\.,]\d+)?)\s*[\)\]\}]?",
            r"\1 Qui-Quadrado(\2)",
            t,
        )
        t = re.sub(
            r"(~|\\sim)\s*t\s*_?\s*[\(\[\{]?\s*(\d+(?:[\.,]\d+)?)\s*[\)\]\}]?\b",
            r"\1 t(\2)",
            t,
            flags=re.IGNORECASE,
        )
        return t

    def analyze_image_text(self, image_path: Optional[str] = None, extra_text_hint: str = "") -> OCRAnalysisResult:
        visual_match: Optional[Dict[str, Any]] = None
        is_trained_image_match = False
        if image_path and os.path.exists(image_path):
            raw_text, backend = self.extract_text(image_path)
            visual_match = self.match_learned_visual_signature(image_path)
            if visual_match:
                sim = visual_match["similarity"]
                margin = visual_match.get("margin", 0.0)
                # Quando o OCR nativo já leu texto da imagem, só substitui pela transcrição
                # memorizada do banco de treino se for uma correspondência praticamente idêntica (sim >= 0.96).
                # Isso evita que uma foto NOVA com a mesma caligrafia (ex.: D ~ Beta(100, 2)) seja
                # sobrescrita por uma foto antiga do treino (ex.: A ~ Beta(1, 8))!
                is_trained_image_match = bool(
                    sim >= 0.96 or (not raw_text.strip() and (sim >= 0.78 or (sim >= 0.70 and margin >= 0.06)))
                )
                v_tr = visual_match.get("transcription", "").strip()
                v_dist = visual_match.get("dist_id", "")
                v_spec = DISTRIBUTIONS_DB.get(v_dist)
                if is_trained_image_match and v_tr:
                    # Se a imagem corresponde com alta precisão a uma imagem do banco de treino,
                    # prioriza a transcrição treinada para evitar que ruídos do OCR (como ler I_{(0,1)} como N(0,1))
                    # alterem a distribuição ou percam os parâmetros da imagem!
                    if (
                        visual_match.get("file") in self.KNOWN_TRAINING_TRANSCRIPTIONS
                        or not v_spec
                        or v_tr != v_spec.canonical_single_rv_latex
                    ):
                        raw_text = v_tr
                    else:
                        raw_text = (v_tr + "\n" + raw_text).strip()
                    backend = "Reconhecimento Visual de Caligrafia Treinada + OCR"
                elif not raw_text.strip() and v_tr:
                    raw_text = v_tr
                    backend = "Reconhecimento Visual de Caligrafia Treinada + OCR"
        else:
            raw_text, backend = "", "Entrada Escrita Direta (Sem Imagem)"

        if extra_text_hint:
            raw_text = (raw_text + "\n" + extra_text_hint).strip()
            if not image_path or not os.path.exists(image_path):
                backend = "Expressão Digitada pelo Usuário"

        cleaned_text = self.normalize_handwritten_ocr(raw_text)
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        scores: Dict[str, float] = {dist_id: 0.0 for dist_id in DISTRIBUTIONS_DB}
        matched_cues: Dict[str, List[str]] = {dist_id: [] for dist_id in DISTRIBUTIONS_DB}

        if visual_match:
            v_dist = visual_match["dist_id"]
            sim = visual_match["similarity"]
            if is_trained_image_match:
                if v_dist in scores:
                    v_score = round(450.0 * sim, 2)
                    scores[v_dist] += v_score
                    matched_cues[v_dist].append(
                        f"Assinatura visual de caligrafia treinada ({visual_match['file']}, similaridade={sim*100:.1f}%)"
                    )
                for k_dist, k_sim in visual_match.get("class_knn_scores", {}).items():
                    if k_dist in scores and k_sim > 0:
                        scores[k_dist] += round(45.0 * k_sim, 2)
            elif not cleaned_text.strip():
                if v_dist in scores:
                    scores[v_dist] += round(120.0 * sim, 2)
                    matched_cues[v_dist].append(
                        f"Proximidade visual com padrão treinado ({visual_match['file']}, similaridade={sim*100:.1f}%)"
                    )
                for k_dist, k_sim in visual_match.get("class_knn_scores", {}).items():
                    if k_dist in scores and k_sim > 0:
                        scores[k_dist] += round(40.0 * k_sim, 2)
            else:
                # Para imagens inéditas com texto lido pelo OCR, usa o k-NN visual apenas como desempate suave
                if v_dist in scores:
                    scores[v_dist] += round(25.0 * sim, 2)
                for k_dist, k_sim in visual_match.get("class_knn_scores", {}).items():
                    if k_dist in scores and k_sim > 0:
                        scores[k_dist] += round(15.0 * k_sim, 2)

        normalized = cleaned_text.lower()
        # Remove ocorrências da função Gama de Euler \Gamma(...) com argumento único e da função
        # exponencial matemática \exp(-...) para que fórmulas da Beta, t-Student, Qui-Quadrado e Normal
        # não acionem falsamente os nomes das distribuições "Gamma" ou "Exp"
        normalized_no_euler_gamma = re.sub(r"\\gamma\s*\([^,)]*\)", "euler_gamma_fn", normalized)
        cleaned_no_euler_gamma = re.sub(r"\\gamma\s*\([^,)]*\)", "euler_gamma_fn", cleaned_text, flags=re.IGNORECASE)
        lines_no_euler_gamma = [line.strip() for line in cleaned_no_euler_gamma.splitlines() if line.strip()]

        normalized_no_math_exp = re.sub(r"\\exp\b|\bexp\s*[\(\{\[]\s*-", "euler_exp_fn", normalized)
        cleaned_no_math_exp = re.sub(r"\\exp\b|\bexp\s*[\(\{\[]\s*-", "euler_exp_fn", cleaned_text, flags=re.IGNORECASE)
        lines_no_math_exp = [line.strip() for line in cleaned_no_math_exp.splitlines() if line.strip()]
        learned_patterns = self._load_learned_patterns()

        for dist_id, spec in DISTRIBUTIONS_DB.items():
            if dist_id == "gamma":
                target_norm, target_clean, target_lines = normalized_no_euler_gamma, cleaned_no_euler_gamma, lines_no_euler_gamma
            elif dist_id == "exponential":
                target_norm, target_clean, target_lines = normalized_no_math_exp, cleaned_no_math_exp, lines_no_math_exp
            else:
                target_norm, target_clean, target_lines = normalized, cleaned_text, lines

            # 0. Padrões aprendidos do treinamento com imagens reais do usuário
            user_cues = learned_patterns.get(dist_id, [])
            for u_cue in user_cues:
                if u_cue and u_cue.lower() in target_norm:
                    scores[dist_id] += 60.0
                    matched_cues[dist_id].append(f"Padrão treinado pelo usuário: '{u_cue}'")

            # 1. Nome e apelidos da distribuição
            for alias in spec.aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, target_norm):
                    scores[dist_id] += 45.0
                    matched_cues[dist_id].append(f"Nome/Apelido reconhecido: '{alias}'")
                if target_lines and re.search(pattern, target_lines[0].lower()):
                    scores[dist_id] += 40.0
                    matched_cues[dist_id].append(f"Destaque principal na escrita: '{alias}'")

            # 2. Padrões de notação matemática (ex.: N(μ, σ²), Pois(λ), Bin(n, p))
            for pat in spec.notation_patterns:
                try:
                    m = re.search(pat, target_clean, flags=re.IGNORECASE)
                    if m:
                        scores[dist_id] += 45.0
                        matched_cues[dist_id].append(f"Notação manuscrita reconhecida: '{m.group(0)}'")
                except re.error:
                    pass

            # 3. Indícios de fórmula matemática (FDP/FMP escrita à mão)
            for cue in spec.formula_cues:
                if dist_id == "gamma" and cue.lower() in ("\\gamma(", "gamma("):
                    continue
                if cue.lower() in normalized:
                    scores[dist_id] += 35.0
                    matched_cues[dist_id].append(f"Fragmento de fórmula reconhecido: '{cue}'")

        # 3b. Correspondência Léxica Aproximada (Fuzzy Matching via SequenceMatcher)
        # Corrige erros gerais de 1–2 caracteres do OCR em escrita cursiva
        # (ex.: 'Expanencial' -> exponential, 'Lagnarmal' -> lognormal, 'Weiell' -> weibull, 'Binamial' -> binomial)
        # sem hardcoding de nenhuma imagem específica.
        def _strip_acc(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()

        fuzzy_canonical_names = {
            "normal": "Normal",
            "exponential": "Exponencial",
            "uniform": "Uniforme",
            "gamma": "Gama",
            "beta": "Beta",
            "chi_square": "Qui-Quadrado",
            "student_t": "t-Student",
            "lognormal": "Lognormal",
            "weibull": "Weibull",
            "poisson": "Poisson",
            "binomial": "Binomial",
            "geometric": "Geometrica",
            "cauchy": "Cauchy",
        }
        fuzzy_candidates = re.findall(r"\b([A-Za-zÀ-ÿ]{5,20})(?=\s*[\(\[\d\-])", target_clean)
        if fuzzy_candidates:
            # Pré-monta tabela de apelidos canônicos (tamanho >= 5) por distribuição
            alias_bank: List[Tuple[str, str]] = []
            for d_id, d_spec in DISTRIBUTIONS_DB.items():
                for al in d_spec.aliases:
                    al_norm = re.sub(r"[^a-z]", "", _strip_acc(al))
                    if len(al_norm) >= 5:
                        alias_bank.append((d_id, al_norm))

            for raw_word in fuzzy_candidates:
                w_norm = re.sub(r"[^a-z]", "", _strip_acc(raw_word))
                if len(w_norm) < 5 or w_norm in ("gamma", "theta", "sigma", "alpha", "lambda", "kappa", "omega"):
                    continue
                per_dist_best: Dict[str, float] = {}
                for d_id, al_norm in alias_bank:
                    # Evita comparar palavras de tamanhos muito discrepantes
                    if abs(len(w_norm) - len(al_norm)) > 3:
                        continue
                    ratio = difflib.SequenceMatcher(None, w_norm, al_norm).ratio()
                    if ratio > per_dist_best.get(d_id, 0.0):
                        per_dist_best[d_id] = ratio
                ranked_fuzzy = sorted(per_dist_best.items(), key=lambda kv: kv[1], reverse=True)
                if ranked_fuzzy:
                    best_d, best_r = ranked_fuzzy[0]
                    second_r = ranked_fuzzy[1][1] if len(ranked_fuzzy) > 1 else 0.0
                    if best_r >= 0.65 and (best_r - second_r) >= 0.08:
                        scores[best_d] += 80.0
                        matched_cues[best_d].append(
                            f"Correspondência léxica aproximada (OCR manuscrito): '{raw_word}' ≈ {best_d} ({best_r:.0%})"
                        )
                        canon_name = fuzzy_canonical_names.get(best_d)
                        if canon_name and _strip_acc(raw_word) != _strip_acc(canon_name):
                            cleaned_text = re.sub(
                                r"\b" + re.escape(raw_word) + r"(?=\s*[\(\[\d\-])",
                                canon_name,
                                cleaned_text,
                            )

        # 4. Reconhecimento estrutural da equação da FDP/FMP (garante acerto mesmo para imagens inéditas onde apenas a fórmula f(x) = ... foi lida)
        # Beta: \Gamma(a)\Gamma(b) ou \Gamma(...)^2 ou x^{a-1}(1-x)^{b-1} com variável na base
        if re.search(r"\\?gamma\([^)]+\)\s*\\?gamma\([^)]+\)|\\?gamma\([^)]+\)\^2", normalized) or re.search(
            r"\b([a-z])(?:\^\{?[-+0-9/\.,a-z\\]+\}?)?\s*\(\s*1\s*-\s*\1\s*\)", normalized
        ):
            scores["beta"] += 110.0
            matched_cues["beta"].append("Estrutura algébrica da FDP Beta: x^{α-1}(1-x)^{β-1} / B(α,β)")

        # t-Student: (1 + x^2/\nu)^{-(\nu+1)/2} ou \Gamma com \sqrt{\nu\pi}
        if re.search(r"\(\s*1\s*\+\s*[a-z]\^2(?:\s*/\s*[\d\.,a-z\-\(\)]+)?\s*\)\s*\^\s*\{?\s*-", normalized):
            scores["student_t"] += 115.0
            matched_cues["student_t"].append("Estrutura algébrica da FDP t-Student: (1 + t²/ν)^{-(ν+1)/2}")

        # Cauchy: 1 / (\pi \gamma [1 + ((x - x_0)/\gamma)^2]) ou \gamma / (\pi [(x - x_0)^2 + \gamma^2]) sem \Gamma(...) e sem \sqrt{\pi}
        if (
            "gamma(" not in normalized
            and "sqrt" not in normalized
            and (
                re.search(
                    r"/\s*\(?\s*(?:\\pi|pi|π)\s*[\*\s\d\.,a-z\\]*[\(\[]\s*1\s*\+|/\s*\(?\s*(?:\\pi|pi|π)\s*[\(\[]\s*\(?\s*[a-z](?:\s*[-+]\s*[\d\.,a-z_\\]+)?\s*\)?\s*\^2\s*\+",
                    normalized,
                )
                or "arctan" in normalized
            )
        ):
            scores["cauchy"] += 115.0
            matched_cues["cauchy"].append("Estrutura algébrica da FDP Cauchy: 1 / (πγ[1 + ((x - x₀)/γ)²])")

        # Qui-Quadrado: (1/2)^{k/2} / \Gamma(k/2) ou e^{-x/2} com \Gamma ou /2-1 + -.../2 com Γ
        if re.search(r"\(\s*1\s*/\s*2\s*\)\s*\^.*?\\?gamma|\\?gamma\([^)]+\).*?e\s*\^\s*\{?\s*-\s*[a-z]\s*/\s*2\b", normalized) or (
            ("/2-1" in normalized or "/2 - 1" in normalized)
            and ("/2" in normalized)
            and ("gamma(" in normalized or "г(" in normalized or "γ(" in normalized or "½" in normalized)
        ):
            scores["chi_square"] += 115.0
            matched_cues["chi_square"].append("Estrutura algébrica da FDP Qui-Quadrado: (1/2)^{k/2}/Γ(k/2) x^{k/2-1} e^{-x/2}")

        # Gama: \Gamma(\alpha) no denominador + e^{-\beta x} (sem (1/2)^ e sem \sqrt{\pi})
        if (
            "gamma(" in normalized
            and re.search(r"e\s*\^\s*\{?\s*-\s*[\d\.,a-z\\]*\s*[a-z]\b", normalized)
            and "(1/2)^" not in normalized
            and "pi" not in normalized
            and "π" not in normalized
        ) or re.search(r"\(?(\d*[\.,]?\d+)\)?\s*\^\s*(\d+)\s+[a-z](?:\^\d+)?\s+e\s*\^\s*\{?\s*-\s*\1", normalized):
            scores["gamma"] += 105.0
            matched_cues["gamma"].append("Estrutura algébrica da FDP Gama: (β^α / Γ(α)) x^{α-1} e^{-βx}")

        # Normal vs Log-Normal: \sqrt{... \pi} com exp(-(x-\mu)^2 / ...)
        if re.search(r"(?:sqrt|\\sqrt)\s*[\(\{]?\s*[\d\.,]*\s*[\*⋅]?\s*(?:pi|\\pi|π)", normalized) or "2π" in normalized or "2\\pi" in normalized:
            if "ln" in normalized or "log" in normalized:
                scores["lognormal"] += 95.0
                matched_cues["lognormal"].append("Estrutura algébrica da FDP Log-Normal: 'ln(x)' com constante '2π'")
            elif "1 + " not in normalized and "1+" not in normalized:
                scores["normal"] += 105.0
                matched_cues["normal"].append("Estrutura algébrica da FDP Normal: (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))")

        # Poisson: e^{-\lambda} \lambda^k / k!
        if re.search(r"(?:e\s*\^\s*\{?\s*-\s*[\d\.,a-z\\]+).*?/\s*[a-z]!|/\s*[a-z]!\s*\)?", normalized):
            if "!(" not in normalized:  # distingue de w!(8-w)! da Binomial
                scores["poisson"] += 115.0
                matched_cues["poisson"].append("Estrutura algébrica da FMP de Poisson: e^{-λ} λ^k / k!")

        # Binomial: C(n, k) ou n!/(k!(n-k)!) com p^k (1-p)^{n-k}
        if re.search(r"\bc\s*\(\s*(?:\d+|n)\s*,\s*[a-z]\s*\)|\b\d+!\s*/\s*\(\s*[a-z]!\s*\(", normalized):
            scores["binomial"] += 115.0
            matched_cues["binomial"].append("Estrutura algébrica da FMP Binomial: C(n,k) p^k (1-p)^{n-k}")

        # Geométrica: p (1-p)^{k-1} sem C(n, k) e sem fatorial
        if (
            "c(" not in normalized
            and "!" not in normalized
            and re.search(
                r"(?:=|\|)\s*\(?(?:\d+\s*/\s*\d+|0?[\.,]\d+|p)\)?\s*\(\s*(?:1\s*-\s*(?:\d+\s*/\s*\d+|0?[\.,]\d+|p)|\d+\s*/\s*\d+|0?[\.,]\d+)\s*\)\s*\^\s*\{?\s*[a-z](?:\s*-\s*1)?\s*\}?",
                normalized,
            )
        ):
            scores["geometric"] += 105.0
            matched_cues["geometric"].append("Estrutura algébrica da FMP Geométrica: p(1-p)^{k-1}")

        # Exponencial: \lambda e^{-\lambda x} (com ou sem '^' e com espaço opcional quando o OCR separa 'e -19k')
        if (
            "gamma(" not in normalized
            and "!" not in normalized
            and "^2" not in normalized
            and (
                re.search(
                    r"(?:=|\|)\s*\(?(\d+\s*/\s*\d+|\d*[\.,]?\d+|\\?lambda|λ)\)?\s*(?:\\?exp|e\s*\^?)\s*[\(\{]?\s*-\s*(?:\d+\s*/\s*\d+|\d*[\.,]?\d+|\\?lambda|λ)?\s*[a-z]",
                    normalized,
                )
                or re.search(
                    r"\b(\d+(?:[\.,]\d+)?)\s+e\s*-\s*\d+(?:[\.,]\d+)?\s*[a-z]\b",
                    normalized,
                )
            )
        ):
            scores["exponential"] += 95.0
            matched_cues["exponential"].append("Estrutura algébrica da FDP Exponencial: λ e^{-λx}")

        # Uniforme: 1 / (b - a) ou constante * I_{[a,b]}(x) sem exp/Gamma
        if re.search(r"1\s*/\s*\(\s*[-+\d\.,a-z]+\s*-\s*[-+\d\.,a-z]+\s*\)", normalized):
            scores["uniform"] += 110.0
            matched_cues["uniform"].append("Estrutura algébrica da FDP Uniforme: 1 / (b - a)")
        elif (
            "gamma" not in normalized
            and "exp" not in normalized
            and "e^" not in normalized
            and "!" not in normalized
            and "^" not in normalized
            and re.search(r"i_\{?[\[\(][-+\d\.,]+\s*,\s*[-+\d\.,]+[\]\)]\}?|i\(\s*[-+\d\.,]+\s*(?:\\le|<=|<)\s*[a-z]", normalized)
        ):
            scores["uniform"] += 95.0
            matched_cues["uniform"].append("Estrutura algébrica da FDP Uniforme: c · I_{[a,b]}(x)")

        if "12" in normalized and ("b-a" in normalized or "b - a" in normalized):
            scores["uniform"] += 35.0
            matched_cues["uniform"].append("Fórmula manuscrita: '(b - a)' e denominador '12'")

        # 5. Reconhecimento tolerante a variações de caligrafia e confusões comuns de OCR manuscrito
        if re.search(r"\b(?:n|norm|normal|n0rmal)\s*[\(\[]", normalized):
            scores["normal"] += 65.0
            matched_cues["normal"].append("Notação detectada: Normal N(...)")
        if re.search(r"\b(?:exp|exponencial|exponential|eponential|ex9)\s*[\(\[]", normalized_no_math_exp):
            scores["exponential"] += 65.0
            matched_cues["exponential"].append("Notação detectada: Exponencial Exp(...)")
        if re.search(r"\b(?:u|unif|un1f|uniforme|uniform)\s*[\(\[]", normalized):
            scores["uniform"] += 65.0
            matched_cues["uniform"].append("Notação detectada: Uniforme U(...)")
        if re.search(r"\b(?:gama|gamma|6ama|cama|ganma|erlang)\s*[\(\[]", normalized_no_euler_gamma):
            scores["gamma"] += 65.0
            matched_cues["gamma"].append("Notação detectada: Gama(...)")
        if re.search(r"\b(?:beta|bela|bcta|8eta|beto|bota|bata)\s*[\(\[]", normalized):
            scores["beta"] += 65.0
            matched_cues["beta"].append("Notação detectada: Beta(...)")
        if re.search(r"\b(?:chi2|chisq|qui-?quadrado|chi-?square|x\^2_|\\chi\^2)", normalized):
            scores["chi_square"] += 65.0
            matched_cues["chi_square"].append("Notação detectada: Qui-Quadrado")
        if re.search(r"\b(?:t-?student|student|t_\d+|t_\{?\w+\}?|t\s*\(\s*\d+)", normalized):
            scores["student_t"] += 65.0
            matched_cues["student_t"].append("Notação detectada: t de Student")
        if re.search(r"\b(?:lognormal|log-?normal|lnorm)\s*[\(\[]", normalized):
            scores["lognormal"] += 65.0
            matched_cues["lognormal"].append("Notação detectada: Log-Normal")
        if re.search(r"\bweibull\s*[\(\[]", normalized):
            scores["weibull"] += 65.0
            matched_cues["weibull"].append("Notação detectada: Weibull")
        if re.search(r"\b(?:pois|p0is|po1s|pols|poisson|paisson)\s*[\(\[]", normalized):
            scores["poisson"] += 65.0
            matched_cues["poisson"].append("Notação detectada: Poisson")
        if re.search(r"\b(?:bin|b1n|8in|bim|binom|binomial|bern|bernoulli)\s*[\(\[]", normalized):
            scores["binomial"] += 65.0
            matched_cues["binomial"].append("Notação detectada: Binomial / Bernoulli")
        if re.search(r"\b(?:geo|6eo|ge0|gco|geom|geometrica|geométrica|geometric)\s*[\(\[]", normalized):
            scores["geometric"] += 65.0
            matched_cues["geometric"].append("Notação detectada: Geométrica")
        if re.search(r"\b(?:cauchy|cauchv|cauchi|couchy|cavchy|lorentz|breit-?wigner)\s*[\(\[]?", normalized):
            scores["cauchy"] += 75.0
            matched_cues["cauchy"].append("Notação detectada: Cauchy / Lorentz")

        top_dist_id = max(scores.keys(), key=lambda d: scores[d])
        context = self._parse_statistical_context(cleaned_text, DISTRIBUTIONS_DB[top_dist_id])

        return OCRAnalysisResult(
            raw_text=raw_text,
            cleaned_handwriting_text=cleaned_text,
            lines=lines,
            backend_used=backend,
            distribution_scores=scores,
            matched_cues=matched_cues,
            statistical_context=context,
        )

    def _parse_statistical_context(self, text: str, top_spec: DistributionSpec) -> StatisticalContextDetails:
        ctx = StatisticalContextDetails()
        if not text:
            ctx.canonical_how_to_write_latex = top_spec.canonical_single_rv_latex
            self._build_image_parameterized_pdf_and_cdf("", top_spec, ctx)
            return ctx

        # 1. Detectar i.i.d. (Independentes e Identicamente Distribuídas)
        iid_patterns = [
            (r"\bi\.?\s*i\.?\s*d\.?\b", "Abreviação 'i.i.d.' escrita na imagem"),
            (r"independent\s+and\s+identically\s+distributed", "Expressão completa em inglês 'independent and identically distributed'"),
            (r"independentes?\s+e\s+identicamente\s+distribu[ií]das?", "Expressão em português 'independentes e identicamente distribuídas'"),
            (r"\\stackrel\{\\text\{i\.?i\.?d\.?\}\}\{\\sim\}", "Notação empilhada LaTeX i.i.d. sobre til"),
            (r"~\s*i\.?i\.?d\.?", "Notação '~ i.i.d.' após a variável aleatória"),
        ]
        for pat, desc in iid_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                ctx.is_iid = True
                ctx.is_independent = True
                ctx.identically_distributed = True
                ctx.iid_evidence.append(f"{desc} ('{m.group(0)}')")

        # 2. Detectar Independência Estatística
        indep_patterns = [
            (r"\bindependent(ly)?\b", "Palavra-chave 'independent' detectada"),
            (r"\bindependentes?\b", "Palavra-chave 'independente(s)' detectada"),
            (r"\\perp\s*\\!*\s*\\perp|⊥|\|_\{?\s*\}?\|", "Símbolo matemático de independência ortogonal (⊥)"),
            (r"P\s*\(\s*X\s*,\s*Y\s*\)\s*=\s*P\s*\(\s*X\s*\)\s*P\s*\(\s*Y\s*\)", "Fatoração da probabilidade conjunta P(X,Y) = P(X)P(Y)"),
            (r"f_\{?X,Y\}?\s*\(\s*x\s*,\s*y\s*\)\s*=\s*f_X\s*\(\s*x\s*\)\s*f_Y\s*\(\s*y\s*\)", "Fatoração da densidade conjunta f_{X,Y}(x,y) = f_X(x)f_Y(y)"),
        ]
        for pat, desc in indep_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                ctx.is_independent = True
                ctx.independence_evidence.append(f"{desc} ('{m.group(0)}')")

        # 3. Identificar Variáveis Aleatórias e Amostras Indexadas (X_1, ..., X_n ou E_1, E_2, ..., E_i)
        m_seq = re.search(
            r"\b([A-Z])_1\s*,\s*(?:\1_2\s*,\s*)?(?:\.{2,}|\\dots|…)\s*,\s*\1_([in]|\d+)",
            text,
        )
        m_tilde_rv = re.search(r"\b([A-Z](?:_\d+|_i|_n)?)\s*(?:~|\\sim)", text)
        if m_tilde_rv:
            ctx.random_variables_detected.append(m_tilde_rv.group(1))

        rv_matches = re.findall(r"\b([XYZUVWSTABCJKLNPQRME]|([A-Z])_(?:\d+|i|n))\b", text)
        for rv_tuple in rv_matches:
            rv = rv_tuple[0]
            if rv not in ("N", "U", "I", "R", "C", "P", "E") or f"{rv} ~" in text or f"{rv}~" in text or "_" in rv:
                if rv not in ctx.random_variables_detected:
                    ctx.random_variables_detected.append(rv)

        # Caso a expressão seja apenas "f(b) = ..." ou "p(q) = ...", infere a variável aleatória maiúscula B, Q, etc.
        if not ctx.random_variables_detected:
            m_fn_arg = re.search(r"\b[fp]\s*\(\s*([a-z])\s*\)\s*=", text)
            if m_fn_arg:
                ctx.random_variables_detected.append(m_fn_arg.group(1).upper())

        m_finite_list = re.search(
            r"\b([A-Z])_(\d+)\s*,\s*\1_(\d+)(?:\s*,\s*\1_(\d+))?(?!\s*,\s*(?:\.{2,}|\\dots|…))",
            text,
        )

        if m_seq or any(tok in text for tok in ["X_1", "X_2", "X_n", "X_i", "Y_1", "Y_n", "E_1", "E_2", "E_i", "E_n"]):
            ctx.is_indexed_sample = True
            ctx.identically_distributed = True
            if m_seq:
                base_letter = m_seq.group(1)
            else:
                m_any_idx = re.search(r"\b([A-Z])_(?:1|2|i|n)\b", text)
                base_letter = m_any_idx.group(1) if m_any_idx else ("Y" if "Y_1" in text else "X")
            ctx.sample_base_letter = base_letter
            ctx.primary_random_variable = f"{base_letter}_1, {base_letter}_2, \\dots, {base_letter}_n"
            if ctx.is_independent and not ctx.is_iid:
                ctx.is_iid = True
                ctx.iid_evidence.append("Sequência amostral indexada combinada com independência")
        elif m_finite_list:
            ctx.is_indexed_sample = True
            ctx.identically_distributed = True
            base_letter = m_finite_list.group(1)
            ctx.sample_base_letter = base_letter
            items = [f"{base_letter}_{m_finite_list.group(2)}", f"{base_letter}_{m_finite_list.group(3)}"]
            if m_finite_list.group(4):
                items.append(f"{base_letter}_{m_finite_list.group(4)}")
            ctx.primary_random_variable = ", ".join(items)
            if ctx.is_independent and not ctx.is_iid:
                ctx.is_iid = True
                ctx.iid_evidence.append("Amostra indexada combinada com independência")
        elif ctx.random_variables_detected:
            first_rv = ctx.random_variables_detected[0]
            ctx.sample_base_letter = first_rv.split("_", 1)[0]
            ctx.primary_random_variable = first_rv
        else:
            ctx.sample_base_letter = "X"
            ctx.primary_random_variable = "X"

        # 3b. Detectar e corrigir erros matemáticos de intervalo/índice na escrita (ex.: "E_1, E_2, ..., E_i ~ Poisson(4,5), i \in [0, n]")
        m_dom = re.search(
            r"\b([ijk])\s*(?:\\in|∈)\s*([\[\(\{])\s*(\d+)\s*[,;]\s*(?:(?:\.{2,}|…|\\dots)\s*[,;]\s*)?([nm]|\d+)\s*([\]\)\}])",
            text,
        )
        if m_dom:
            idx_var, l_brk, start_idx, end_idx, r_brk = (
                m_dom.group(1),
                m_dom.group(2),
                m_dom.group(3),
                m_dom.group(4),
                m_dom.group(5),
            )
            end_clean = "n" if end_idx in ("n", "m") else end_idx
            b_let = ctx.sample_base_letter or "X"
            if l_brk in ("[", "(") or r_brk in ("]", ")"):
                ctx.notation_corrections.append(
                    f"<strong>Intervalo Contínuo vs. Conjunto Discreto de Índices (${l_brk}{start_idx}, {end_clean}{r_brk} \\longrightarrow \\{{1, 2, \\dots, {end_clean}\\}}$):</strong> "
                    f"Na imagem foi escrito ${idx_var} \\in {l_brk}{start_idx}, {end_clean}{r_brk}$. Os colchetes ${l_brk}{start_idx}, {end_clean}{r_brk}$ "
                    f"representam um <em>intervalo contínuo de números reais</em> (incluindo decimais como ${idx_var}=0.5, 1.7$, etc.), "
                    f"mas o índice ${idx_var}$ das variáveis aleatórias é um <strong>número inteiro discreto</strong>. "
                    f"O correto é usar chaves: <strong>${idx_var} \\in \\{{1, 2, \\dots, {end_clean}\\}}$</strong> (ou ${idx_var} = 1, 2, \\dots, {end_clean}$)."
                )
            if start_idx == "0" and (f"{b_let}_1" in text or ctx.is_indexed_sample):
                ctx.notation_corrections.append(
                    f"<strong>Índice Inicial $0$ vs. Sequência Iniciada em ${b_let}_1$:</strong> "
                    f"A sequência de variáveis começa em <strong>${b_let}_1$</strong> (${b_let}_1, {b_let}_2, \\dots$), onde não existe a variável ${b_let}_0$ "
                    f"(começar em $0$ até ${end_clean}$ implicaria ${end_clean}+1$ variáveis). "
                    f"Portanto, o domínio do índice ${idx_var}$ deve começar em <strong>$1$</strong>: <strong>${idx_var} \\in \\{{1, 2, \\dots, {end_clean}\\}}$</strong>."
                )
            if m_seq and m_seq.group(2) == idx_var and end_clean != idx_var:
                ctx.notation_corrections.append(
                    f"<strong>Conflito de Índice no Último Termo da Sequência (${b_let}_1, {b_let}_2, \\dots, {b_let}_{idx_var}$ com ${idx_var}$ indo até ${end_clean}$):</strong> "
                    f"Se o índice ${idx_var}$ varia de $1$ até ${end_clean}$, a lista completa da amostra deve terminar em <strong>${b_let}_{end_clean}$</strong> "
                    f"($\\,{b_let}_1, {b_let}_2, \\dots, {b_let}_{end_clean}\\,$), ou deve-se declarar apenas o termo genérico indexado: "
                    f"<strong>${b_let}_{idx_var}, \\;\\forall\\, {idx_var} \\in \\{{1, 2, \\dots, {end_clean}\\}}$</strong>."
                )

        # 4. Declarações de distribuição (com '~')
        tilde_matches = re.findall(r"([A-Za-z0-9_,\.\s\\]+(?:~|\\sim)\s*[A-Za-z\\.\s]+\([^)]+\))", text)
        for tm in tilde_matches:
            clean_tm = " ".join(tm.split())
            ctx.notation_statements.append(clean_tm)

        # 5. Condicionamento (|)
        cond_matches = re.findall(r"([PpfE]\s*\([^)|]+\|[^)]+\))", text)
        for cm in cond_matches:
            ctx.conditioning_detected.append(cm.strip())

        # 6. Extrair parâmetros numéricos e simbólicos escritos à mão
        param_regexes = {
            "mu": [r"(?:\\mu|μ|\bmu\b|\bmedia\b)\s*=\s*([-+]?\d*\.?\d+)"],
            "sigma": [r"(?:\\sigma|σ|\bsigma\b|\bdp\b)\s*=\s*(\d*\.?\d+)"],
            "sigma2": [r"(?:\\sigma\^2|σ²|σ\^2|\bvar\b)\s*=\s*(\d*\.?\d+)"],
            "lam": [r"(?:\\lambda|λ|\blambda\b|\btaxa\b)\s*=\s*(\d*\.?\d+)"],
            "alpha": [r"(?:\\alpha|α|\balpha\b)\s*=\s*(\d*\.?\d+)"],
            "beta": [r"(?:\\beta|β|\bbeta\b)\s*=\s*(\d*\.?\d+)"],
            "k": [r"(?:\bk\b|\bgl\b|\bdf\b)\s*=\s*(\d*\.?\d+)"],
            "nu": [r"(?:\\nu|ν|\bnu\b)\s*=\s*(\d*\.?\d+)"],
            "n": [r"\bn\s*=\s*(\d+)"],
            "p": [r"\bp\s*=\s*(0?\.\d+|1\.0|0|1)\b"],
            "a": [r"\ba\s*=\s*([-+]?\d*\.?\d+)"],
            "b": [r"\bb\s*=\s*([-+]?\d*\.?\d+)"],
            "x0": [r"(?:x_0|x0|\blocacao\b|\bloca[çc][aã]o\b)\s*=\s*([-+]?\d*\.?\d+)"],
            "gamma": [r"(?:\\gamma|γ|\bgamma\b|\bescala\b)\s*=\s*(\d*\.?\d+)"],
        }
        for key, regex_list in param_regexes.items():
            for rgx in regex_list:
                m = re.search(rgx, text, flags=re.IGNORECASE)
                if m:
                    try:
                        val = float(m.group(1))
                        if key == "sigma2":
                            ctx.extracted_parameters["sigma"] = val ** 0.5
                            ctx.parameterization_hint = f"Média e Variância (μ, σ²={val})"
                        else:
                            ctx.extracted_parameters[key] = val
                    except ValueError:
                        pass

        # Argumentos posicionais em N(0, 1), U(a, b), U[a, b], Bin(n, p), Pois(λ), Exp(λ), Gama(α, β), Beta(α, β)
        m_norm = re.search(r"\b(?:N|Norm|Normal)\s*[\(\[]\s*([-+]?\d*[\.,]?\d+)\s*[,;]\s*(\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_norm:
            ctx.extracted_parameters.setdefault("mu", self._parse_numeric_or_fraction(m_norm.group(1)))
            s2 = self._parse_numeric_or_fraction(m_norm.group(2))
            ctx.extracted_parameters.setdefault("sigma", s2 ** 0.5 if s2 > 0 else 1.0)
            ctx.parameterization_hint = f"Notação posicional N(μ={m_norm.group(1)}, σ²={m_norm.group(2)})"

        m_unif = re.search(r"\b(?:U|Unif|Uniforme?)\s*[\(\[]\s*([-+]?\d*[\.,]?\d+)\s*[,;]\s*([-+]?\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_unif:
            ctx.extracted_parameters.setdefault("a", self._parse_numeric_or_fraction(m_unif.group(1)))
            ctx.extracted_parameters.setdefault("b", self._parse_numeric_or_fraction(m_unif.group(2)))
            ctx.parameterization_hint = f"Intervalo Uniforme U(a={m_unif.group(1)}, b={m_unif.group(2)})"

        m_bin = re.search(r"\b(?:Bin|B1n|8in|Bim|Binom|Binomial)\s*[\(\[]\s*(\d+)\s*[,;]\s*(\d+\s*/\s*\d+|0?[\.,]\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_bin:
            ctx.extracted_parameters.setdefault("n", float(m_bin.group(1)))
            p_val = self._parse_numeric_or_fraction(m_bin.group(2))
            ctx.extracted_parameters.setdefault("p", p_val)
            ctx.parameterization_hint = f"Binomial(n={m_bin.group(1)}, p={p_val:g})"

        m_pois = re.search(r"\b(?:Pois|P0is|Po1s|Pols|Poisson|Paisson)\s*[\(\[]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_pois:
            ctx.extracted_parameters.setdefault("lam", self._parse_numeric_or_fraction(m_pois.group(1)))

        m_exp = re.search(r"\b(?:Exp|Exponencial|Exponential|Eponential)\s*[\(\[]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_exp:
            ctx.extracted_parameters.setdefault("lam", self._parse_numeric_or_fraction(m_exp.group(1)))

        m_beta = re.search(r"\b(?:Beta|Bela|Bcta|8eta|Beto)\s*[\(\[]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[,;]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_beta:
            ctx.extracted_parameters.setdefault("alpha", self._parse_numeric_or_fraction(m_beta.group(1)))
            ctx.extracted_parameters.setdefault("beta", self._parse_numeric_or_fraction(m_beta.group(2)))

        m_gamma = re.search(r"\b(?:Gamma|Gama|6ama|Cama|Ganma|\\Gamma)\s*[\(\[]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[,;]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_gamma:
            ctx.extracted_parameters.setdefault("alpha", self._parse_numeric_or_fraction(m_gamma.group(1)))
            ctx.extracted_parameters.setdefault("beta", self._parse_numeric_or_fraction(m_gamma.group(2)))

        m_chisq = re.search(
            r"(?:chi2|chisq|qui-?quadrado|chi-?square|\\chi\^2|\bx\^2)_?\{?\s*(\d*[\.,]?\d+)\s*\}?|\b(?:chi2|chisq|qui-?quadrado|chi-?square)\s*[\(\[]\s*(\d*[\.,]?\d+)\s*[\)\]]",
            text,
            re.IGNORECASE,
        )
        if m_chisq:
            val_k = m_chisq.group(1) or m_chisq.group(2)
            if val_k:
                ctx.extracted_parameters.setdefault("k", self._parse_numeric_or_fraction(val_k))

        m_stud = re.search(r"\b(?:t-?student|student|t)\s*[\(\[]\s*(\d*[\.,]?\d+)\s*[\)\]]|\bt_\{?\s*(\d*[\.,]?\d+)\s*\}?", text, re.IGNORECASE)
        if m_stud:
            val_nu = m_stud.group(1) or m_stud.group(2)
            if val_nu:
                ctx.extracted_parameters.setdefault("nu", self._parse_numeric_or_fraction(val_nu))

        m_weib = re.search(r"\bweibull\s*[\(\[]\s*(\d*[\.,]?\d+)\s*[,;]\s*(\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_weib:
            ctx.extracted_parameters.setdefault("k", self._parse_numeric_or_fraction(m_weib.group(1)))
            ctx.extracted_parameters.setdefault("lam", self._parse_numeric_or_fraction(m_weib.group(2)))

        m_lnorm = re.search(r"\b(?:lognormal|log-?normal|lnorm)\s*[\(\[]\s*([-+]?\d*[\.,]?\d+)\s*[,;]\s*(\d*[\.,]?\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_lnorm:
            ctx.extracted_parameters.setdefault("mu", self._parse_numeric_or_fraction(m_lnorm.group(1)))
            ctx.extracted_parameters.setdefault("sigma", self._parse_numeric_or_fraction(m_lnorm.group(2)))

        m_geom = re.search(r"\b(?:geo|6eo|ge0|gco|geom|geometrica|geométrica|geometric)\s*[\(\[]\s*(\d+\s*/\s*\d+|0?[\.,]\d+)\s*[\)\]]", text, re.IGNORECASE)
        if m_geom:
            ctx.extracted_parameters.setdefault("p", self._parse_numeric_or_fraction(m_geom.group(1)))

        m_weib = re.search(
            r"\bWeibull\s*[\(\[]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[,;]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[\)\]]",
            text,
            re.IGNORECASE,
        )
        if m_weib:
            ctx.extracted_parameters.setdefault("k", self._parse_numeric_or_fraction(m_weib.group(1)))
            ctx.extracted_parameters.setdefault("lam", self._parse_numeric_or_fraction(m_weib.group(2)))

        m_cauchy2 = re.search(
            r"\b(?:Cauchy|Lorentz)\s*[\(\[]\s*([-+]?\d+\s*/\s*\d+|[-+]?\d*[\.,]?\d+)\s*[,;]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[\)\]]",
            text,
            re.IGNORECASE,
        )
        if m_cauchy2:
            ctx.extracted_parameters.setdefault("x0", self._parse_numeric_or_fraction(m_cauchy2.group(1)))
            ctx.extracted_parameters.setdefault("gamma", self._parse_numeric_or_fraction(m_cauchy2.group(2)))
        else:
            m_cauchy1 = re.search(
                r"\b(?:Cauchy|Lorentz)\s*[\(\[]\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*[\)\]]",
                text,
                re.IGNORECASE,
            )
            if m_cauchy1:
                ctx.extracted_parameters.setdefault("x0", 0.0)
                ctx.extracted_parameters.setdefault("gamma", self._parse_numeric_or_fraction(m_cauchy1.group(1)))

        # 7. Extrair parâmetros diretamente da expressão da Função Densidade/Massa (FDP/FMP) escrita na imagem!
        self._extract_parameters_from_pdf_formula(text, top_spec, ctx)

        # 7b. Fallback geral para OCR que omitiu parêntese de abertura ou fechamento
        # (ex.: "Uniforme 0 100)", "Geometrica 0,7)", "Binomial(50,0 0,1", "Lognormal 0,1 10)", "19 e -19k")
        if top_spec.id == "uniform" and ("a" not in ctx.extracted_parameters or "b" not in ctx.extracted_parameters):
            m_u_loose = re.search(
                r"\b(?:Uniforme?|Unif)\s*[\(\[]?\s*([-+]?\d+(?:[\.,]\d+)?)\s*[,;\s]+\s*([-+]?\d+(?:[\.,]\d+)?)",
                text,
                re.IGNORECASE,
            )
            if m_u_loose:
                ctx.extracted_parameters.setdefault("a", self._parse_numeric_or_fraction(m_u_loose.group(1)))
                ctx.extracted_parameters.setdefault("b", self._parse_numeric_or_fraction(m_u_loose.group(2)))

        elif top_spec.id == "geometric" and "p" not in ctx.extracted_parameters:
            m_g_loose = re.search(
                r"\b(?:Geometrica|Geométrica|Geometric|Geom)\s*[\(\[]?\s*(0[\.,]\d+|1\s*/\s*\d+)",
                text,
                re.IGNORECASE,
            )
            if m_g_loose:
                ctx.extracted_parameters.setdefault("p", self._parse_numeric_or_fraction(m_g_loose.group(1)))

        elif top_spec.id == "binomial" and ("n" not in ctx.extracted_parameters or "p" not in ctx.extracted_parameters):
            m_b_loose = re.search(
                r"\b(?:Binomial|Binom|Bin)\s*[\(\[]?\s*(\d+)\b[^\n]{0,18}?\b(0[\.,]\d+)",
                text,
                re.IGNORECASE,
            )
            if m_b_loose:
                ctx.extracted_parameters.setdefault("n", float(m_b_loose.group(1)))
                ctx.extracted_parameters.setdefault("p", self._parse_numeric_or_fraction(m_b_loose.group(2)))

        elif top_spec.id == "lognormal" and ("mu" not in ctx.extracted_parameters or "sigma" not in ctx.extracted_parameters):
            m_l_loose = re.search(
                r"\b(?:Lognormal|Log-?Normal)\s*[\(\[]?\s*([-+]?\d+)(?:[\.,]\d+)?\s*[,;\s]+\s*(\d+(?:[\.,]\d+)?)",
                text,
                re.IGNORECASE,
            )
            if m_l_loose:
                ctx.extracted_parameters.setdefault("mu", self._parse_numeric_or_fraction(m_l_loose.group(1)))
                ctx.extracted_parameters.setdefault("sigma", self._parse_numeric_or_fraction(m_l_loose.group(2)))

        elif top_spec.id == "exponential" and "lam" not in ctx.extracted_parameters:
            m_e_loose = re.search(
                r"\b(\d+(?:[\.,]\d+)?)\s+e\s*-\s*\d+(?:[\.,]\d+)?\s*[a-z]\b",
                text,
                re.IGNORECASE,
            )
            if m_e_loose:
                ctx.extracted_parameters.setdefault("lam", self._parse_numeric_or_fraction(m_e_loose.group(1)))

        # Filtra apenas as chaves de parâmetros que pertencem à distribuição identificada
        valid_keys_by_dist = {
            "normal": ("mu", "sigma"),
            "exponential": ("lam",),
            "uniform": ("a", "b"),
            "gamma": ("alpha", "beta"),
            "beta": ("alpha", "beta"),
            "chi_square": ("k",),
            "student_t": ("nu",),
            "lognormal": ("mu", "sigma"),
            "weibull": ("lam", "k"),
            "poisson": ("lam",),
            "binomial": ("n", "p"),
            "geometric": ("p",),
            "cauchy": ("x0", "gamma"),
        }
        allowed_keys = valid_keys_by_dist.get(top_spec.id, ())
        if allowed_keys:
            ctx.extracted_parameters = {
                k: v for k, v in ctx.extracted_parameters.items() if k in allowed_keys
            }

        if "n" in ctx.extracted_parameters:
            ctx.sample_size_n = int(ctx.extracted_parameters["n"])

        # Construir a recomendação educacional "Como deveria ser escrito formalmente"
        ctx.canonical_how_to_write_latex, ctx.handwriting_feedback = self._build_educational_writing_guide(
            text, top_spec, ctx
        )

        # Construir a FDP/FMP e a FDA (Acumulada) substituídas com os parâmetros e variável da imagem
        self._build_image_parameterized_pdf_and_cdf(text, top_spec, ctx)

        return ctx

    @classmethod
    def _extract_parameters_from_pdf_formula(
        cls, text: str, spec: DistributionSpec, ctx: StatisticalContextDetails
    ) -> None:
        """
        Quando a foto mostra a equação da Função Densidade de Probabilidade (FDP) ou
        Função Massa de Probabilidade (FMP), extrai os valores numéricos ou simbólicos
        dos parâmetros diretamente da estrutura algébrica da fórmula!
        """
        p = ctx.extracted_parameters

        if spec.id == "beta":
            # Ex.: \Gamma(6)\Gamma(7) ou \Gamma(1)\Gamma(8) ou \Gamma(2)\Gamma(3)
            m_gam = re.search(r"\\?Gamma\(\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*\)\s*\\?Gamma\(\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*\)", text, re.IGNORECASE)
            if m_gam:
                p.setdefault("alpha", cls._parse_numeric_or_fraction(m_gam.group(1)))
                p.setdefault("beta", cls._parse_numeric_or_fraction(m_gam.group(2)))
            # Ex.: \Gamma(1/2)^2
            m_gam_sq = re.search(r"\\?Gamma\(\s*(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*\)\^2", text, re.IGNORECASE)
            if m_gam_sq:
                val = cls._parse_numeric_or_fraction(m_gam_sq.group(1))
                p.setdefault("alpha", val)
                p.setdefault("beta", val)
            # Ex.: 20 w^3 (1-w) ou x(1-x)^2 ou a^{1-1}(1-a)^{8-1} -> expoentes a-1 e b-1
            if "alpha" not in p or "beta" not in p:
                m_exp_beta = re.search(
                    r"\b([a-z])(?:\^\{?(\d+\s*-\s*\d+|[-+]?\d+\s*/\s*\d+|[-+]?\d*[\.,]?\d+)\}?)?\s*\(\s*1\s*-\s*\1\s*\)(?:\^\{?(\d+\s*-\s*\d+|[-+]?\d+\s*/\s*\d+|[-+]?\d*[\.,]?\d+)\}?)?",
                    text,
                    re.IGNORECASE,
                )
                if m_exp_beta:
                    e1 = cls._parse_numeric_or_fraction(m_exp_beta.group(2)) if m_exp_beta.group(2) else 1.0
                    e2 = cls._parse_numeric_or_fraction(m_exp_beta.group(3)) if m_exp_beta.group(3) else 1.0
                    p.setdefault("alpha", e1 + 1.0)
                    p.setdefault("beta", e2 + 1.0)
            # Parâmetros simbólicos: \theta_1, \theta_2 ou a, b
            if "theta_1" in text or "\\theta_1" in text:
                ctx.symbolic_parameters_detected = [r"\theta_1", r"\theta_2"]
            elif re.search(r"Beta\s*\(\s*a\s*,\s*b\s*\)|\\Gamma\(a\+b\)", text, re.IGNORECASE):
                ctx.symbolic_parameters_detected = ["a", "b"]

        elif spec.id == "binomial":
            # Ex.: C(15, j) (0.3)^j ou C(76, y) 0.25^y ou (8! / (w!(8-w)!)) (0.75)^w
            m_c = re.search(r"(?:C\(\s*(\d+)\s*,|\b(\d+)!\s*/).*?\(?\s*(\d+\s*/\s*\d+|0?[\.,]\d+)\s*\)?\^", text, re.IGNORECASE)
            if m_c:
                n_str = m_c.group(1) or m_c.group(2)
                if n_str:
                    p.setdefault("n", float(n_str))
                p.setdefault("p", cls._parse_numeric_or_fraction(m_c.group(3)))
            if not p and re.search(r"Bin\s*\(\s*n\s*,\s*p\s*\)", text, re.IGNORECASE):
                ctx.symbolic_parameters_detected = ["n", "p"]

        elif spec.id == "chi_square":
            # Ex.: (1/2)^{20/2} ou \Gamma(20/2) ou (1/2)^5 / \Gamma(5)
            m_half = re.search(r"\\?Gamma\(\s*(\d+)\s*/\s*2\s*\)", text, re.IGNORECASE)
            if m_half:
                p.setdefault("k", float(m_half.group(1)))
            elif "k" not in p:
                m_g_int = re.search(r"\\?Gamma\(\s*(\d+(?:\.5)?)\s*\)", text, re.IGNORECASE)
                if m_g_int:
                    p.setdefault("k", float(m_g_int.group(1)) * 2.0)
                elif "1/2) e^{-" in text:
                    p.setdefault("k", 2.0)
            if not p and ("g/2" in text or "Qui-Quadrado(g)" in text):
                ctx.symbolic_parameters_detected = ["g"]

        elif spec.id == "exponential":
            # Ex.: 7 e^{-7k}, 0.5 e^{-0.5 t}, (1/3) e^{-y/3}, 5 \exp(-5w)
            m_ex = re.search(r"(\d+\s*/\s*\d+|\d*[\.,]?\d+)\s*\)?\s*(?:\\?exp|e\^)\s*[\(\{]?\s*-\s*", text, re.IGNORECASE)
            if m_ex:
                p.setdefault("lam", cls._parse_numeric_or_fraction(m_ex.group(1)))
            elif "lam" not in p:
                m_ex_div = re.search(r"(?:\\?exp|e\^)\s*[\(\{]?\s*-\s*[a-z]\s*/\s*(\d*[\.,]?\d+)", text, re.IGNORECASE)
                if m_ex_div:
                    p.setdefault("lam", 1.0 / max(cls._parse_numeric_or_fraction(m_ex_div.group(1)), 1e-9))
            if not p:
                ctx.symbolic_parameters_detected = [r"\lambda"]

        elif spec.id == "gamma":
            # Ex.: (19^{15} / \Gamma(15)) y^{14} e^{-19y} ou (3^5 / \Gamma(5)) w^4 e^{-3w} ou (1/\Gamma(4)) x^3 e^{-x} ou (0.5)^2 y e^{-0.5 y}
            m_gam_a = re.search(r"\\?Gamma\(\s*(\d*[\.,]?\d+)\s*\)", text, re.IGNORECASE)
            if m_gam_a:
                p.setdefault("alpha", cls._parse_numeric_or_fraction(m_gam_a.group(1)))
            m_gam_b = re.search(r"e\^\{?-\s*(\d*[\.,]?\d+)?\s*[a-z]", text, re.IGNORECASE)
            if m_gam_b:
                b_str = m_gam_b.group(1)
                p.setdefault("beta", cls._parse_numeric_or_fraction(b_str) if b_str else 1.0)
            if "alpha" not in p:
                m_ba = re.search(r"\(?(\d*[\.,]?\d+)\)?\s*\^\s*\{?(\d*[\.,]?\d+)\}?\s+[a-z]", text, re.IGNORECASE)
                if m_ba:
                    p.setdefault("alpha", cls._parse_numeric_or_fraction(m_ba.group(2)))
            if "\\psi" in text or "\\eta" in text:
                ctx.symbolic_parameters_detected = [r"\psi", r"\eta"]
            elif not p:
                ctx.symbolic_parameters_detected = [r"\alpha", r"\beta"]

        elif spec.id == "geometric":
            # Ex.: 0.67 (1 - 0.67)^{z-1} ou 0.4 (0.6)^{q-1} ou (1/4) (3/4)^{k-1}
            m_ge = re.search(r"(?:=|\|)\s*\(?(\d+\s*/\s*\d+|0?[\.,]\d+)\)?\s*(?:\(|\^)", text, re.IGNORECASE)
            if m_ge:
                p.setdefault("p", cls._parse_numeric_or_fraction(m_ge.group(1)))
            if not p:
                ctx.symbolic_parameters_detected = ["p"]

        elif spec.id == "normal":
            # Ex.: \exp(-(x-10)^2 / 8) ou \exp\{-(n+2)^2 / 14\} ou e^{-(y+3)^2 / 18} ou \exp(-1/8.70 (z - 3)^2)
            m_nf = re.search(r"\(\s*[a-z]\s*([-+])\s*(\d*[\.,]?\d+)\s*\)\^2\s*\}\s*/\s*(\d*[\.,]?\d+)|\(\s*[a-z]\s*([-+])\s*(\d*[\.,]?\d+)\s*\)\^2\s*/\s*(\d*[\.,]?\d+)", text, re.IGNORECASE)
            if m_nf:
                sign_s = m_nf.group(1) or m_nf.group(4)
                mu_abs = cls._parse_numeric_or_fraction(m_nf.group(2) or m_nf.group(5))
                denom_2s2 = cls._parse_numeric_or_fraction(m_nf.group(3) or m_nf.group(6))
                mu_val = -mu_abs if sign_s == "+" else mu_abs
                s2_val = denom_2s2 / 2.0
                p.setdefault("mu", mu_val)
                p.setdefault("sigma", s2_val ** 0.5 if s2_val > 0 else 1.0)
            else:
                m_nf_pre = re.search(r"-\s*1\s*/\s*(\d*[\.,]?\d+)\s*\(\s*[a-z]\s*([-+])\s*(\d*[\.,]?\d+)\s*\)\^2", text, re.IGNORECASE)
                if m_nf_pre:
                    denom_2s2 = cls._parse_numeric_or_fraction(m_nf_pre.group(1))
                    sign_s = m_nf_pre.group(2)
                    mu_abs = cls._parse_numeric_or_fraction(m_nf_pre.group(3))
                    mu_val = -mu_abs if sign_s == "+" else mu_abs
                    s2_val = denom_2s2 / 2.0
                    p.setdefault("mu", mu_val)
                    p.setdefault("sigma", s2_val ** 0.5 if s2_val > 0 else 1.0)
                else:
                    m_nf_zero = re.search(r"(?:e\^\{?|\\?exp\s*[\(\{]\s*)-\s*[a-z]\^2\s*/\s*(\d*[\.,]?\d+)", text, re.IGNORECASE)
                    if m_nf_zero:
                        denom_2s2 = cls._parse_numeric_or_fraction(m_nf_zero.group(1))
                        s2_val = denom_2s2 / 2.0
                        p.setdefault("mu", 0.0)
                        p.setdefault("sigma", s2_val ** 0.5 if s2_val > 0 else 1.0)
            if "2v^2" in text or "(k - m)" in text or "(k-m)" in text:
                ctx.symbolic_parameters_detected = ["m", "v^2"]
            elif not p:
                ctx.symbolic_parameters_detected = [r"\mu", r"\sigma^2"]

        elif spec.id == "poisson":
            # Ex.: e^{-0.49} (0.49)^w / w! ou (10^k e^{-10} / k!)
            m_pf = re.search(r"e\^\{?-\s*(\d*[\.,]?\d+)\}?", text, re.IGNORECASE)
            if m_pf:
                p.setdefault("lam", cls._parse_numeric_or_fraction(m_pf.group(1)))
            if "e^{-r}" in text or "Poisson(r)" in text:
                ctx.symbolic_parameters_detected = ["r"]
            elif not p:
                ctx.symbolic_parameters_detected = [r"\lambda"]

        elif spec.id == "student_t":
            # Ex.: (1 + t^2/22) ou (1 + x^2/10) ou (1 + v^2/5)
            m_tf = re.search(r"1\s*\+\s*[a-z]\^2\s*/\s*(\d*[\.,]?\d+)", text, re.IGNORECASE)
            if m_tf:
                p.setdefault("nu", cls._parse_numeric_or_fraction(m_tf.group(1)))
            elif re.search(r"\(1\s*\+\s*[a-z]\^2\)\^\{?-1\}?", text, re.IGNORECASE):
                p.setdefault("nu", 1.0)
            if "n-1" in text:
                ctx.symbolic_parameters_detected = ["n - 1"]
            elif "v^2/a" in text or "t_a" in text:
                ctx.symbolic_parameters_detected = ["a"]
            elif not p:
                ctx.symbolic_parameters_detected = [r"\nu"]

        elif spec.id == "uniform":
            # Ex.: 1 / (0.89 - 0.22) ou 1 / (2 - 1)
            m_uf = re.search(r"1\s*/\s*\(\s*([-+]?\d*[\.,]?\d+)\s*-\s*([-+]?\d*[\.,]?\d+)\s*\)", text)
            if m_uf:
                b_v = cls._parse_numeric_or_fraction(m_uf.group(1))
                a_v = cls._parse_numeric_or_fraction(m_uf.group(2))
                p.setdefault("a", min(a_v, b_v))
                p.setdefault("b", max(a_v, b_v))
            # Ex.: I_{[-2,2]}(y) ou I_{(5,15)}(u) ou I_{[0,1]}(x)
            if "a" not in p or "b" not in p:
                m_ui = re.search(r"I_\{?[\[\(]\s*([-+]?\d*[\.,]?\d+)\s*,\s*([-+]?\d*[\.,]?\d+)\s*[\]\)]\}?", text)
                if m_ui:
                    p.setdefault("a", cls._parse_numeric_or_fraction(m_ui.group(1)))
                    p.setdefault("b", cls._parse_numeric_or_fraction(m_ui.group(2)))
            # Ex.: I(0 \le w \le 10)
            if "a" not in p or "b" not in p:
                m_ule = re.search(r"I\(\s*([-+]?\d*[\.,]?\d+)\s*(?:\\le|<=|<)\s*[a-z]\s*(?:\\le|<=|<)\s*([-+]?\d*[\.,]?\d+)\s*\)", text)
                if m_ule:
                    p.setdefault("a", cls._parse_numeric_or_fraction(m_ule.group(1)))
                    p.setdefault("b", cls._parse_numeric_or_fraction(m_ule.group(2)))
            if "c - b" in text or "c-b" in text:
                ctx.symbolic_parameters_detected = ["b", "c"]
            elif not p:
                ctx.symbolic_parameters_detected = ["a", "b"]

        elif spec.id == "cauchy":
            # 1. Forma 1 / (\pi * \gamma * (1 + ((x - x0) / \gamma)^2))
            m_c_ratio = re.search(
                r"\(\s*[a-z]\s*([-+])\s*(\d*[\.,]?\d+)\s*\)\s*/\s*(\d*[\.,]?\d+)\s*\)\s*\^2",
                text,
                re.IGNORECASE,
            )
            if m_c_ratio:
                sign_s = m_c_ratio.group(1)
                x0_abs = cls._parse_numeric_or_fraction(m_c_ratio.group(2))
                g_val = cls._parse_numeric_or_fraction(m_c_ratio.group(3))
                p.setdefault("x0", -x0_abs if sign_s == "+" else x0_abs)
                p.setdefault("gamma", g_val)
            # 2. Forma \gamma / (\pi * ((x - x0)^2 + \gamma^2))
            if "x0" not in p or "gamma" not in p:
                m_c_expanded = re.search(
                    r"\(\s*[a-z]\s*([-+])\s*(\d*[\.,]?\d+)\s*\)\s*\^2\s*\+\s*(\d*[\.,]?\d+)(\^2)?",
                    text,
                    re.IGNORECASE,
                )
                if m_c_expanded:
                    sign_s = m_c_expanded.group(1)
                    x0_abs = cls._parse_numeric_or_fraction(m_c_expanded.group(2))
                    g_part = cls._parse_numeric_or_fraction(m_c_expanded.group(3))
                    g_val = g_part if m_c_expanded.group(4) else (g_part ** 0.5)
                    p.setdefault("x0", -x0_abs if sign_s == "+" else x0_abs)
                    p.setdefault("gamma", g_val)
            # 3. Forma centrada em 0: 1 / (\pi * \gamma * (1 + (x / \gamma)^2))
            if "x0" not in p or "gamma" not in p:
                m_c_zero = re.search(r"1\s*\+\s*\(?\s*[a-z]\s*/\s*(\d*[\.,]?\d+)\s*\)?\s*\^2", text, re.IGNORECASE)
                if m_c_zero:
                    p.setdefault("x0", 0.0)
                    p.setdefault("gamma", cls._parse_numeric_or_fraction(m_c_zero.group(1)))
            # 4. Forma Cauchy Padrão: 1 / (\pi (1 + x^2))
            if "x0" not in p or "gamma" not in p:
                if re.search(r"(?:\\pi|pi|π)\s*\(\s*1\s*\+\s*[a-z]\^2\s*\)", text, re.IGNORECASE):
                    p.setdefault("x0", 0.0)
                    p.setdefault("gamma", 1.0)
            if not p:
                ctx.symbolic_parameters_detected = ["x_0", r"\gamma"]

    @staticmethod
    def _build_image_parameterized_pdf_and_cdf(
        text: str, spec: DistributionSpec, ctx: StatisticalContextDetails
    ) -> None:
        """
        Gera em LaTeX a Função Densidade/Massa de Probabilidade (FDP/FMP) e a
        Função de Distribuição Acumulada (FDA/CDF) substituídas especificamente com os
        parâmetros (numéricos ou simbólicos) e a variável aleatória capturados na imagem!
        """
        p = ctx.extracted_parameters
        sym = ctx.symbolic_parameters_detected
        base_let = ctx.sample_base_letter or "X"
        if ctx.is_iid or ctx.is_indexed_sample:
            rv_upper = f"{base_let}_i"
            rv_lower = "k" if (spec.family == "Discreta" or base_let == "E") else base_let.lower()
        else:
            rv_upper = ctx.primary_random_variable or "X"
            if len(rv_upper) > 2 or "_" in rv_upper:
                rv_upper = base_let
            rv_lower = "k" if (spec.family == "Discreta" and rv_upper == "E") else rv_upper.lower()

        if spec.id == "beta":
            if "alpha" in p and "beta" in p:
                a_v, b_v = p["alpha"], p["beta"]
                ab_sum = round(a_v + b_v, 4)
                em1 = round(a_v - 1.0, 4)
                em2 = round(b_v - 1.0, 4)
                ctx.image_parameters_latex = rf"\alpha = {a_v:g}, \quad \beta = {b_v:g}"

                # Simplificação quando alpha e beta são inteiros positivos
                if a_v == int(a_v) and b_v == int(b_v) and a_v >= 1 and b_v >= 1:
                    ia, ib = int(a_v), int(b_v)
                    const_val = math.factorial(ia + ib - 1) // (math.factorial(ia - 1) * math.factorial(ib - 1))
                    term_x = "" if ia - 1 == 0 else (f"{rv_lower}" if ia - 1 == 1 else f"{rv_lower}^{{{ia - 1}}}")
                    term_1mx = "" if ib - 1 == 0 else (f"(1 - {rv_lower})" if ib - 1 == 1 else f"(1 - {rv_lower})^{{{ib - 1}}}")
                    prod_terms = f"{const_val} \, {term_x}{term_1mx}".strip()
                    if not term_x and not term_1mx:
                        prod_terms = f"{const_val}"
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{\Gamma({ia} + {ib})}}{{\Gamma({ia})\,\Gamma({ib})}}\,{rv_lower}^{{{ia}-1}}(1 - {rv_lower})^{{{ib}-1}}\,\mathbb{{I}}_{{(0,1)}}({rv_lower}) "
                        rf"= \frac{{\Gamma({ia + ib})}}{{\Gamma({ia})\,\Gamma({ib})}}\,{rv_lower}^{{{ia - 1}}}(1 - {rv_lower})^{{{ib - 1}}}\,\mathbb{{I}}_{{(0,1)}}({rv_lower}) "
                        rf"= {prod_terms}\,\mathbb{{I}}_{{(0,1)}}({rv_lower})"
                    )
                elif abs(a_v - 0.5) < 1e-6 and abs(b_v - 0.5) < 1e-6:
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{\Gamma(1)}}{{[\Gamma(1/2)]^2}}\,{rv_lower}^{{-1/2}}(1 - {rv_lower})^{{-1/2}}\,\mathbb{{I}}_{{(0,1)}}({rv_lower}) "
                        rf"= \frac{{1}}{{\pi\sqrt{{{rv_lower}(1 - {rv_lower})}}}}\,\mathbb{{I}}_{{(0,1)}}({rv_lower})"
                    )
                else:
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{\Gamma({ab_sum:g})}}{{\Gamma({a_v:g})\,\Gamma({b_v:g})}}\,{rv_lower}^{{{em1:g}}}(1 - {rv_lower})^{{{em2:g}}}\,\mathbb{{I}}_{{(0,1)}}({rv_lower})"
                    )

                # FDA da Beta com os parâmetros da imagem
                if a_v == 1.0 and b_v == 1.0:
                    ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \int_0^{{{rv_lower}}} 1\,dt = {rv_lower}, \quad {rv_lower} \in [0, 1]"
                    ctx.image_cdf_note = "Para α = 1 e β = 1, a Beta(1,1) coincide com a Uniforme(0,1) e possui FDA elementar fechada F(y) = y."
                elif a_v == 1.0:
                    ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = 1 - (1 - {rv_lower})^{{{b_v:g}}}, \quad {rv_lower} \in [0, 1]"
                    ctx.image_cdf_note = f"Para α = 1 e β = {b_v:g}, a integral da densidade possui forma elementar fechada."
                elif b_v == 1.0:
                    ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = {rv_lower}^{{{a_v:g}}}, \quad {rv_lower} \in [0, 1]"
                    ctx.image_cdf_note = f"Para α = {a_v:g} e β = 1, a integral da densidade possui forma elementar fechada."
                elif abs(a_v - 0.5) < 1e-6 and abs(b_v - 0.5) < 1e-6:
                    ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = I_{{{rv_lower}}}(0.5, 0.5) = \frac{{2}}{{\pi}}\arcsin\!\left(\sqrt{{{rv_lower}}}\right), \quad {rv_lower} \in [0, 1]"
                    ctx.image_cdf_note = "Para α = 0.5 e β = 0.5 (Distribuição Arcosseno), a FDA possui forma analítica fechada via função arcosseno."
                else:
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = I_{{{rv_lower}}}({a_v:g}, {b_v:g}) "
                        rf"= \frac{{\Gamma({ab_sum:g})}}{{\Gamma({a_v:g})\,\Gamma({b_v:g})}}\int_0^{{{rv_lower}}} t^{{{em1:g}}}(1 - t)^{{{em2:g}}}\,dt, \quad {rv_lower} \in [0, 1]"
                    )
                    ctx.image_cdf_note = f"Expressa pela Função Beta Incompleta Regularizada I_{rv_lower}({a_v:g}, {b_v:g})."
            else:
                s1, s2 = (sym[0], sym[1]) if len(sym) >= 2 else (r"\alpha", r"\beta")
                ctx.image_parameters_latex = rf"\text{{Parâmetros simbólicos da imagem: }} ({s1}, \, {s2})"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{\Gamma({s1} + {s2})}}{{\Gamma({s1})\,\Gamma({s2})}}\,{rv_lower}^{{{s1} - 1}}(1 - {rv_lower})^{{{s2} - 1}}\,\mathbb{{I}}_{{(0,1)}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = I_{{{rv_lower}}}({s1}, {s2}) = \frac{{\Gamma({s1} + {s2})}}{{\Gamma({s1})\,\Gamma({s2})}}\int_0^{{{rv_lower}}} t^{{{s1} - 1}}(1 - t)^{{{s2} - 1}}\,dt"
                ctx.image_cdf_note = "Expressa pela Função Beta Incompleta Regularizada com os parâmetros simbólicos da imagem."

        elif spec.id == "normal":
            if "mu" in p and "sigma" in p:
                mu_v = p["mu"]
                s2_v = round(p["sigma"] ** 2, 4)
                two_s2 = round(2.0 * s2_v, 4)
                sig_v = round(p["sigma"], 4)
                ctx.image_parameters_latex = rf"\mu = {mu_v:g}, \quad \sigma^2 = {s2_v:g} \;\; (\sigma = {sig_v:g})"
                if mu_v == 0:
                    num_sq = f"{rv_lower}^2"
                    z_expr = rf"\frac{{{rv_lower}}}{{{sig_v:g}}}" if sig_v != 1 else rv_lower
                elif mu_v < 0:
                    num_sq = f"({rv_lower} + {abs(mu_v):g})^2"
                    z_expr = rf"\frac{{{rv_lower} + {abs(mu_v):g}}}{{{sig_v:g}}}"
                else:
                    num_sq = f"({rv_lower} - {mu_v:g})^2"
                    z_expr = rf"\frac{{{rv_lower} - {mu_v:g}}}{{{sig_v:g}}}"
                ctx.image_pdf_latex = (
                    rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{\sqrt{{2\pi \cdot {s2_v:g}}}}}\exp\!\left(-\frac{{{num_sq}}}{{2 \cdot {s2_v:g}}}\right)\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower}) "
                    rf"= \frac{{1}}{{\sqrt{{{two_s2:g}\pi}}}}\exp\!\left(-\frac{{{num_sq}}}{{{two_s2:g}}}\right)\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower})"
                )
                ctx.image_cdf_latex = (
                    rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \Phi\!\left({z_expr}\right) "
                    rf"= \frac{{1}}{{\sqrt{{{two_s2:g}\pi}}}}\int_{{-\infty}}^{{{rv_lower}}} \exp\!\left(-\frac{{(t - ({mu_v:g}))^2}}{{{two_s2:g}}}\right)dt "
                    rf"= \frac{{1}}{{2}}\left[1 + \operatorname{{erf}}\!\left(\frac{{{rv_lower} - ({mu_v:g})}}{{\sqrt{{{two_s2:g}}}}}\right)\right]"
                )
                ctx.image_cdf_note = "A Distribuição Normal não possui primitiva em funções elementares; sua FDA é dada pela integral Gaussiana Φ(z) / função erro erf(·) com os parâmetros da imagem."
            else:
                s1, s2 = (sym[0], sym[1]) if len(sym) >= 2 else (r"\mu", r"\sigma^2")
                ctx.image_parameters_latex = rf"\text{{Parâmetros simbólicos da imagem: }} ({s1}, \, {s2})"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{\sqrt{{2\pi {s2}}}}}\exp\!\left(-\frac{{({rv_lower} - {s1})^2}}{{2{s2}}}\right)\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \Phi\!\left(\frac{{{rv_lower} - {s1}}}{{\sqrt{{{s2}}}}}\right) = \int_{{-\infty}}^{{{rv_lower}}} \frac{{1}}{{\sqrt{{2\pi {s2}}}}}\exp\!\left(-\frac{{(t - {s1})^2}}{{2{s2}}}\right)dt"
                ctx.image_cdf_note = "Expressa via integral Gaussiana Φ(z) com os parâmetros simbólicos da imagem."

        elif spec.id == "exponential":
            if "lam" in p:
                lam_v = p["lam"]
                lam_s = "1/3" if abs(lam_v - 1.0 / 3.0) < 1e-4 else f"{lam_v:g}"
                exp_arg = f"-{rv_lower}/3" if lam_s == "1/3" else f"-{lam_s}{rv_lower}"
                ctx.image_parameters_latex = rf"\lambda = {lam_s}"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = {lam_s}\,e^{{{exp_arg}}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower}), \quad {rv_lower} > 0"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \left(1 - e^{{{exp_arg}}}\right)\mathbb{{I}}_{{[0,\infty)}}({rv_lower}), \quad {rv_lower} \ge 0"
                ctx.image_cdf_note = "A Distribuição Exponencial possui FDA em forma analítica fechada exata!"
            else:
                s1 = sym[0] if sym else r"\lambda"
                ctx.image_parameters_latex = rf"\text{{Parâmetro simbólico da imagem: }} {s1}"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = {s1}\,e^{{-{s1}{rv_lower}}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \left(1 - e^{{-{s1}{rv_lower}}}\right)\mathbb{{I}}_{{[0,\infty)}}({rv_lower})"
                ctx.image_cdf_note = "FDA em forma analítica fechada exata."

        elif spec.id == "uniform":
            if "a" in p and "b" in p:
                a_v, b_v = p["a"], p["b"]
                span = round(b_v - a_v, 4)
                inv_span = round(1.0 / span, 4) if span > 0 else 1.0
                ctx.image_parameters_latex = rf"a = {a_v:g}, \quad b = {b_v:g} \;\; (\text{{comprimento }} b - a = {span:g})"
                ctx.image_pdf_latex = (
                    rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{{b_v:g} - ({a_v:g})}}\,\mathbb{{I}}_{{[{a_v:g}, \, {b_v:g}]}}({rv_lower}) "
                    rf"= \frac{{1}}{{{span:g}}}\,\mathbb{{I}}_{{[{a_v:g}, \, {b_v:g}]}}({rv_lower}) = {inv_span:g}\,\mathbb{{I}}_{{[{a_v:g}, \, {b_v:g}]}}({rv_lower})"
                )
                num_u = f"{rv_lower}" if a_v == 0 else (f"{rv_lower} + {abs(a_v):g}" if a_v < 0 else f"{rv_lower} - {a_v:g}")
                ctx.image_cdf_latex = (
                    rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \begin{{cases}} "
                    rf"0, & {rv_lower} < {a_v:g} \\[4pt] "
                    rf"\dfrac{{{num_u}}}{{{span:g}}}, & {a_v:g} \le {rv_lower} \le {b_v:g} \\[4pt] "
                    rf"1, & {rv_lower} > {b_v:g} \end{{cases}}"
                )
                ctx.image_cdf_note = "A Distribuição Uniforme Contínua possui FDA linear fechada exata no intervalo [a, b]."
            else:
                s1, s2 = (sym[0], sym[1]) if len(sym) >= 2 else ("a", "b")
                ctx.image_parameters_latex = rf"\text{{Parâmetros simbólicos da imagem: }} ({s1}, \, {s2})"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{{s2} - {s1}}}\,\mathbb{{I}}_{{({s1}, \, {s2})}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{{rv_lower} - {s1}}}{{{s2} - {s1}}}\,\mathbb{{I}}_{{[{s1}, \, {s2}]}}({rv_lower}) + \mathbb{{I}}_{{({s2}, \infty)}}({rv_lower})"
                ctx.image_cdf_note = "FDA linear por partes com os parâmetros simbólicos da imagem."

        elif spec.id == "gamma":
            if "alpha" in p and "beta" in p:
                a_v, b_v = p["alpha"], p["beta"]
                am1 = round(a_v - 1.0, 4)
                ctx.image_parameters_latex = rf"\alpha = {a_v:g}, \quad \beta = {b_v:g}"
                ctx.image_pdf_latex = (
                    rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{({b_v:g})^{{{a_v:g}}}}}{{\Gamma({a_v:g})}}\,{rv_lower}^{{{a_v:g} - 1}} e^{{-{b_v:g}{rv_lower}}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower}) "
                    rf"= \frac{{({b_v:g})^{{{a_v:g}}}}}{{\Gamma({a_v:g})}}\,{rv_lower}^{{{am1:g}}} e^{{-{b_v:g}{rv_lower}}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower})"
                )
                ctx.image_cdf_latex = (
                    rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{\gamma({a_v:g}, \, {b_v:g}{rv_lower})}}{{\Gamma({a_v:g})}} "
                    rf"= \frac{{({b_v:g})^{{{a_v:g}}}}}{{\Gamma({a_v:g})}}\int_0^{{{rv_lower}}} t^{{{am1:g}}} e^{{-{b_v:g}t}}\,dt, \quad {rv_lower} \ge 0"
                )
                ctx.image_cdf_note = f"Expressa pela Função Gama Incompleta Regularizada γ({a_v:g}, {b_v:g}{rv_lower})/Γ({a_v:g})."
            else:
                s1, s2 = (sym[0], sym[1]) if len(sym) >= 2 else (r"\alpha", r"\beta")
                ctx.image_parameters_latex = rf"\text{{Parâmetros simbólicos da imagem: }} ({s1}, \, {s2})"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{{s2}^{{{s1}}}}}{{\Gamma({s1})}}\,{rv_lower}^{{{s1} - 1}} e^{{-{s2}{rv_lower}}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{\gamma({s1}, \, {s2}{rv_lower})}}{{\Gamma({s1})}} = \frac{{{s2}^{{{s1}}}}}{{\Gamma({s1})}}\int_0^{{{rv_lower}}} t^{{{s1} - 1}} e^{{-{s2}t}}\,dt"
                ctx.image_cdf_note = "Expressa pela Função Gama Incompleta Regularizada com os parâmetros simbólicos da imagem."

        elif spec.id == "chi_square":
            if "k" in p:
                k_v = p["k"]
                khalf = round(k_v / 2.0, 4)
                km1 = round(khalf - 1.0, 4)
                ctx.image_parameters_latex = rf"k = {k_v:g} \;\; (\text{{graus de liberdade}})"
                if k_v == 2.0:
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{(1/2)^{{2/2}}}}{{\Gamma(2/2)}}\,{rv_lower}^{{2/2 - 1}} e^{{-{rv_lower}/2}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower}) "
                        rf"= \frac{{1}}{{2}}\,e^{{-{rv_lower}/2}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \left(1 - e^{{-{rv_lower}/2}}\right)\mathbb{{I}}_{{[0,\infty)}}({rv_lower})"
                    ctx.image_cdf_note = "Para k = 2 graus de liberdade, a Qui-Quadrado coincide com a Exponencial(1/2) e possui FDA fechada exata!"
                else:
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{(1/2)^{{{k_v:g}/2}}}}{{\Gamma({k_v:g}/2)}}\,{rv_lower}^{{{k_v:g}/2 - 1}} e^{{-{rv_lower}/2}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower}) "
                        rf"= \frac{{(1/2)^{{{khalf:g}}}}}{{\Gamma({khalf:g})}}\,{rv_lower}^{{{km1:g}}} e^{{-{rv_lower}/2}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{\gamma({khalf:g}, \, {rv_lower}/2)}}{{\Gamma({khalf:g})}} "
                        rf"= \frac{{(1/2)^{{{khalf:g}}}}}{{\Gamma({khalf:g})}}\int_0^{{{rv_lower}}} t^{{{km1:g}}} e^{{-t/2}}\,dt, \quad {rv_lower} \ge 0"
                    )
                    ctx.image_cdf_note = f"Expressa pela Função Gama Incompleta Regularizada γ({khalf:g}, {rv_lower}/2)/Γ({khalf:g})."
            else:
                s1 = sym[0] if sym else "k"
                ctx.image_parameters_latex = rf"\text{{Parâmetro simbólico da imagem: }} {s1}"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{(1/2)^{{{s1}/2}}}}{{\Gamma({s1}/2)}}\,{rv_lower}^{{{s1}/2 - 1}} e^{{-{rv_lower}/2}}\,\mathbb{{I}}_{{(0,\infty)}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{\gamma({s1}/2, \, {rv_lower}/2)}}{{\Gamma({s1}/2)}} = \int_0^{{{rv_lower}}} \frac{{(1/2)^{{{s1}/2}}}}{{\Gamma({s1}/2)}}\,t^{{{s1}/2 - 1}} e^{{-t/2}}\,dt"
                ctx.image_cdf_note = "Expressa via Função Gama Incompleta Regularizada."

        elif spec.id == "student_t":
            if "nu" in p:
                nu_v = p["nu"]
                nup1_half = round((nu_v + 1.0) / 2.0, 4)
                nu_half = round(nu_v / 2.0, 4)
                ctx.image_parameters_latex = rf"\nu = {nu_v:g} \;\; (\text{{graus de liberdade}})"
                if nu_v == 1.0:
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{\Gamma(1)}}{{\sqrt{{\pi}}\,\Gamma(1/2)}}\left(1 + {rv_lower}^2\right)^{{-1}}\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower}) "
                        rf"= \frac{{1}}{{\pi(1 + {rv_lower}^2)}}\,\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{1}}{{2}} + \frac{{1}}{{\pi}}\arctan({rv_lower}), \quad {rv_lower} \in \mathbb{{R}}"
                    ctx.image_cdf_note = "Para ν = 1 grau de liberdade, a t de Student coincide com a Distribuição de Cauchy Padrão e possui FDA fechada exata via arctan!"
                else:
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{\Gamma\!\left(\frac{{{nu_v:g}+1}}{{2}}\right)}}{{\sqrt{{{nu_v:g}\pi}}\,\Gamma\!\left(\frac{{{nu_v:g}}}{{2}}\right)}}\left(1 + \frac{{{rv_lower}^2}}{{{nu_v:g}}}\right)^{{-\frac{{{nu_v:g}+1}}{{2}}}}\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower}) "
                        rf"= \frac{{\Gamma({nup1_half:g})}}{{\sqrt{{{nu_v:g}\pi}}\,\Gamma({nu_half:g})}}\left(1 + \frac{{{rv_lower}^2}}{{{nu_v:g}}}\right)^{{-{nup1_half:g}}}\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \int_{{-\infty}}^{{{rv_lower}}} \frac{{\Gamma({nup1_half:g})}}{{\sqrt{{{nu_v:g}\pi}}\,\Gamma({nu_half:g})}}\left(1 + \frac{{u^2}}{{{nu_v:g}}}\right)^{{-{nup1_half:g}}}du "
                        rf"= \frac{{1}}{{2}} + {rv_lower}\,\frac{{\Gamma({nup1_half:g})}}{{\sqrt{{{nu_v:g}\pi}}\,\Gamma({nu_half:g})}}\,{{}}_2F_1\!\left(\frac{{1}}{{2}}, {nup1_half:g}; \frac{{3}}{{2}}; -\frac{{{rv_lower}^2}}{{{nu_v:g}}}\right)"
                    )
                    ctx.image_cdf_note = f"Integral da densidade t de Student com ν = {nu_v:g} graus de liberdade."
            else:
                s1 = sym[0] if sym else r"\nu"
                ctx.image_parameters_latex = rf"\text{{Parâmetro simbólico da imagem: }} {s1}"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{\Gamma\!\left(\frac{{({s1})+1}}{{2}}\right)}}{{\sqrt{{({s1})\pi}}\,\Gamma\!\left(\frac{{{s1}}}{{2}}\right)}}\left(1 + \frac{{{rv_lower}^2}}{{{s1}}}\right)^{{-\frac{{({s1})+1}}{{2}}}}\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \int_{{-\infty}}^{{{rv_lower}}} \frac{{\Gamma\!\left(\frac{{({s1})+1}}{{2}}\right)}}{{\sqrt{{({s1})\pi}}\,\Gamma\!\left(\frac{{{s1}}}{{2}}\right)}}\left(1 + \frac{{u^2}}{{{s1}}}\right)^{{-\frac{{({s1})+1}}{{2}}}}du"
                ctx.image_cdf_note = "Integral acumulada da distribuição t de Student com o parâmetro simbólico da imagem."

        elif spec.id == "poisson":
            if "lam" in p:
                lam_v = p["lam"]
                ctx.image_parameters_latex = rf"\lambda = {lam_v:g}"
                ctx.image_pdf_latex = (
                    rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = \frac{{e^{{-{lam_v:g}}}\,({lam_v:g})^{{{rv_lower}}}}}{{{rv_lower}!}}\,\mathbb{{I}}_{{\mathbb{{N}}}}({rv_lower}), \quad {rv_lower} \in \{{0, 1, 2, \dots\}}"
                )
                ctx.image_cdf_latex = (
                    rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \sum_{{j=0}}^{{\lfloor {rv_lower} \rfloor}} \frac{{e^{{-{lam_v:g}}}\,({lam_v:g})^{{j}}}}{{j!}} = \frac{{\Gamma(\lfloor {rv_lower} \rfloor + 1, \, {lam_v:g})}}{{\lfloor {rv_lower} \rfloor!}}, \quad {rv_lower} \ge 0"
                )
                ctx.image_cdf_note = f"Soma discreta acumulada das probabilidades de Poisson com λ = {lam_v:g} até ⌊{rv_lower}⌋."
            else:
                s1 = sym[0] if sym else r"\lambda"
                ctx.image_parameters_latex = rf"\text{{Parâmetro simbólico da imagem: }} {s1}"
                ctx.image_pdf_latex = rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = \frac{{e^{{-{s1}}}\,({s1})^{{{rv_lower}}}}}{{{rv_lower}!}}\,\mathbb{{I}}_{{\mathbb{{N}}}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \sum_{{j=0}}^{{\lfloor {rv_lower} \rfloor}} \frac{{e^{{-{s1}}}\,({s1})^{{j}}}}{{j!}}"
                ctx.image_cdf_note = "Soma discreta acumulada de Poisson com o parâmetro simbólico da imagem."

        elif spec.id == "binomial":
            if "n" in p and "p" in p:
                n_v = int(p["n"])
                p_v = p["p"]
                q_v = round(1.0 - p_v, 4)
                p_s = "1/4" if ("1/4" in text and abs(p_v - 0.25) < 1e-4) else f"{p_v:g}"
                q_s = "3/4" if p_s == "1/4" else f"{q_v:g}"
                ctx.image_parameters_latex = rf"n = {n_v}, \quad p = {p_s} \;\; (1 - p = {q_s})"
                if abs(p_v - 0.5) < 1e-6:
                    ctx.image_pdf_latex = (
                        rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = \binom{{{n_v}}}{{{rv_lower}}} (0.5)^{{{rv_lower}}} (0.5)^{{{n_v} - {rv_lower}}}\,\mathbb{{I}}_{{\{{0, 1, \dots, {n_v}\}}}}({rv_lower}) "
                        rf"= \binom{{{n_v}}}{{{rv_lower}}} (0.5)^{{{n_v}}}\,\mathbb{{I}}_{{\{{0, 1, \dots, {n_v}\}}}}({rv_lower})"
                    )
                else:
                    ctx.image_pdf_latex = (
                        rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = \binom{{{n_v}}}{{{rv_lower}}} \left({p_s}\right)^{{{rv_lower}}} \left({q_s}\right)^{{{n_v} - {rv_lower}}}\,\mathbb{{I}}_{{\{{0, 1, \dots, {n_v}\}}}}({rv_lower})"
                    )
                ctx.image_cdf_latex = (
                    rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \sum_{{j=0}}^{{\lfloor {rv_lower} \rfloor}} \binom{{{n_v}}}{{j}} \left({p_s}\right)^{{j}} \left({q_s}\right)^{{{n_v} - j}} "
                    rf"= I_{{{q_v:g}}}\!\left({n_v} - \lfloor {rv_lower} \rfloor, \, \lfloor {rv_lower} \rfloor + 1\right), \quad 0 \le {rv_lower} \le {n_v}"
                )
                ctx.image_cdf_note = f"Soma discreta acumulada das probabilidades Binomiais para n = {n_v} e p = {p_s}."
            else:
                s1, s2 = (sym[0], sym[1]) if len(sym) >= 2 else ("n", "p")
                ctx.image_parameters_latex = rf"\text{{Parâmetros simbólicos da imagem: }} ({s1}, \, {s2})"
                ctx.image_pdf_latex = rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = \binom{{{s1}}}{{{rv_lower}}} {s2}^{{{rv_lower}}} (1 - {s2})^{{{s1} - {rv_lower}}}\,\mathbb{{I}}_{{\{{0, 1, \dots, {s1}\}}}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \sum_{{j=0}}^{{\lfloor {rv_lower} \rfloor}} \binom{{{s1}}}{{j}} {s2}^{{j}} (1 - {s2})^{{{s1} - j}}"
                ctx.image_cdf_note = "Soma discreta acumulada Binomial com os parâmetros simbólicos da imagem."

        elif spec.id == "geometric":
            if "p" in p:
                p_v = p["p"]
                q_v = round(1.0 - p_v, 4)
                p_s = "1/4" if ("1/4" in text and abs(p_v - 0.25) < 1e-4) else f"{p_v:g}"
                q_s = "3/4" if p_s == "1/4" else f"{q_v:g}"
                ctx.image_parameters_latex = rf"p = {p_s} \;\; (1 - p = {q_s})"
                starts_at_zero = "{0,1,2" in text.replace(" ", "")
                if starts_at_zero:
                    ctx.image_pdf_latex = (
                        rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = {p_s}\left({q_s}\right)^{{{rv_lower}}}\,\mathbb{{I}}_{{\{{0, 1, 2, \dots\}}}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = 1 - \left({q_s}\right)^{{\lfloor {rv_lower} \rfloor + 1}}, \quad {rv_lower} \ge 0"
                    )
                    ctx.image_cdf_note = "A Distribuição Geométrica em {0, 1, 2, ...} possui FDA fechada exata!"
                elif abs(p_v - 0.5) < 1e-6:
                    ctx.image_pdf_latex = (
                        rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = 0.5\,(0.5)^{{{rv_lower} - 1}}\,\mathbb{{I}}_{{\mathbb{{N}}^*}}({rv_lower}) = (0.5)^{{{rv_lower}}}\,\mathbb{{I}}_{{\{{1, 2, \dots\}}}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = 1 - (0.5)^{{\lfloor {rv_lower} \rfloor}}, \quad {rv_lower} \ge 1"
                    )
                    ctx.image_cdf_note = "A Distribuição Geométrica possui FDA analítica fechada exata!"
                else:
                    ctx.image_pdf_latex = (
                        rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = {p_s}\left({q_s}\right)^{{{rv_lower} - 1}}\,\mathbb{{I}}_{{\{{1, 2, \dots\}}}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = 1 - \left({q_s}\right)^{{\lfloor {rv_lower} \rfloor}}, \quad {rv_lower} \ge 1"
                    )
                    ctx.image_cdf_note = "A Distribuição Geométrica possui FDA analítica fechada exata!"
            else:
                ctx.image_parameters_latex = r"\text{Parâmetro simbólico da imagem: } p"
                ctx.image_pdf_latex = rf"p_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} = {rv_lower}) = p(1 - p)^{{{rv_lower} - 1}}\,\mathbb{{I}}_{{\{{1, 2, \dots\}}}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = 1 - (1 - p)^{{\lfloor {rv_lower} \rfloor}}, \quad {rv_lower} \ge 1"
                ctx.image_cdf_note = "FDA analítica fechada exata com o parâmetro simbólico p."

        elif spec.id == "cauchy":
            if "x0" in p and "gamma" in p:
                x0_v, g_v = p["x0"], p["gamma"]
                g2_v = round(g_v ** 2, 4)
                ctx.image_parameters_latex = rf"x_0 = {x0_v:g} \;\; (\text{{locação/mediana}}), \quad \gamma = {g_v:g} \;\; (\text{{escala/HWHM}})"
                if x0_v == 0.0 and g_v == 1.0:
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{\pi\left(1 + {rv_lower}^2\right)}}\,\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower}), \quad {rv_lower} \in \mathbb{{R}}"
                    )
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{1}}{{2}} + \frac{{1}}{{\pi}}\arctan({rv_lower}), \quad {rv_lower} \in \mathbb{{R}}"
                    )
                    ctx.image_cdf_note = "A Distribuição de Cauchy Padrão Cauchy(0, 1) coincide com t(1) e possui FDA analítica fechada exata via função arco-tangente!"
                else:
                    if x0_v == 0.0:
                        shift_str = f"{rv_lower}"
                        shift_sq = f"{rv_lower}^2"
                    elif x0_v < 0:
                        shift_str = f"{rv_lower} + {abs(x0_v):g}"
                        shift_sq = f"({rv_lower} + {abs(x0_v):g})^2"
                    else:
                        shift_str = f"{rv_lower} - {x0_v:g}"
                        shift_sq = f"({rv_lower} - {x0_v:g})^2"
                    ctx.image_pdf_latex = (
                        rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{\pi \cdot {g_v:g}\left[1 + \left(\frac{{{shift_str}}}{{{g_v:g}}}\right)^2\right]}}\,\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower}) "
                        rf"= \frac{{{g_v:g}}}{{\pi\left[{shift_sq} + {g2_v:g}\right]}}\,\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower})"
                    )
                    ctx.image_cdf_latex = (
                        rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{1}}{{2}} + \frac{{1}}{{\pi}}\arctan\!\left(\frac{{{shift_str}}}{{{g_v:g}}}\right), \quad {rv_lower} \in \mathbb{{R}}"
                    )
                    ctx.image_cdf_note = f"A Distribuição de Cauchy possui FDA analítica fechada exata via arctan com locação x₀ = {x0_v:g} e escala γ = {g_v:g}!"
            else:
                s1, s2 = (sym[0], sym[1]) if len(sym) >= 2 else ("x_0", r"\gamma")
                ctx.image_parameters_latex = rf"\text{{Parâmetros simbólicos da imagem: }} ({s1}, \, {s2})"
                ctx.image_pdf_latex = rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{\pi {s2}\left[1 + \left(\frac{{{rv_lower} - {s1}}}{{{s2}}}\right)^2\right]}}\,\mathbb{{I}}_{{\mathbb{{R}}}}({rv_lower})"
                ctx.image_cdf_latex = rf"F_{{{rv_upper}}}({rv_lower}) = \mathbb{{P}}({rv_upper} \le {rv_lower}) = \frac{{1}}{{2}} + \frac{{1}}{{\pi}}\arctan\!\left(\frac{{{rv_lower} - {s1}}}{{{s2}}}\right), \quad {rv_lower} \in \mathbb{{R}}"
                ctx.image_cdf_note = "FDA analítica fechada exata via função arco-tangente com os parâmetros simbólicos da imagem."

        elif spec.id == "lognormal":
            if "mu" in p and "sigma" in p:
                mu_v, sig_v = p["mu"], p["sigma"]
                var_v = round(sig_v ** 2, 4)
                two_var = round(2.0 * var_v, 4)
                ctx.image_parameters_latex = rf"\mu = {mu_v:g}, \quad \sigma^2 = {var_v:g} \;\; (\sigma = {sig_v:g})"
                shift_ln = rf"\ln({rv_lower})" if mu_v == 0.0 else (rf"\ln({rv_lower}) + {abs(mu_v):g}" if mu_v < 0 else rf"\ln({rv_lower}) - {mu_v:g}")
                ctx.image_pdf_latex = (
                    rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{1}}{{{rv_lower}\sqrt{{2\pi \cdot {var_v:g}}}}} "
                    rf"\exp\!\left(-\frac{{({shift_ln})^2}}{{{two_var:g}}}\right)\,\mathbb{{I}}_{{(0, \infty)}}({rv_lower})"
                )
                ctx.image_cdf_latex = (
                    rf"F_{{{rv_upper}}}({rv_lower}) = \Phi\!\left(\frac{{{shift_ln}}}{{{sig_v:g}}}\right), \quad {rv_lower} > 0"
                )
                ctx.image_cdf_note = f"FDA da Log-Normal via Normal Padrão Φ com μ = {mu_v:g} e σ = {sig_v:g}."
            else:
                ctx.image_parameters_latex = spec.primary_parameters_latex
                ctx.image_pdf_latex = spec.pdf_or_pmf_latex
                ctx.image_cdf_latex = spec.cdf_latex
                ctx.image_cdf_note = "FDA expressa em termos da Normal Padrão Φ(z)."

        elif spec.id == "weibull":
            if "k" in p and "lam" in p:
                k_v, lam_v = p["k"], p["lam"]
                k_m1 = round(k_v - 1.0, 4)
                ctx.image_parameters_latex = rf"k = {k_v:g} \;\; (\text{{forma}}), \quad \lambda = {lam_v:g} \;\; (\text{{escala}})"
                ctx.image_pdf_latex = (
                    rf"f_{{{rv_upper}}}({rv_lower}) = \frac{{{k_v:g}}}{{{lam_v:g}}}\left(\frac{{{rv_lower}}}{{{lam_v:g}}}\right)^{{{k_m1:g}}} "
                    rf"\exp\!\left(-\left(\frac{{{rv_lower}}}{{{lam_v:g}}}\right)^{{{k_v:g}}}\right)\,\mathbb{{I}}_{{[0, \infty)}}({rv_lower})"
                )
                ctx.image_cdf_latex = (
                    rf"F_{{{rv_upper}}}({rv_lower}) = 1 - \exp\!\left(-\left(\frac{{{rv_lower}}}{{{lam_v:g}}}\right)^{{{k_v:g}}}\right), \quad {rv_lower} \ge 0"
                )
                ctx.image_cdf_note = f"A Distribuição de Weibull possui FDA analítica fechada exata com forma k = {k_v:g} e escala λ = {lam_v:g}!"
            else:
                ctx.image_parameters_latex = spec.primary_parameters_latex
                ctx.image_pdf_latex = spec.pdf_or_pmf_latex
                ctx.image_cdf_latex = spec.cdf_latex
                ctx.image_cdf_note = "FDA analítica fechada exata da distribuição de Weibull."

        else:
            ctx.image_parameters_latex = spec.primary_parameters_latex
            ctx.image_pdf_latex = spec.pdf_or_pmf_latex
            ctx.image_cdf_latex = spec.cdf_latex
            ctx.image_cdf_note = "FDP/FMP e FDA da distribuição identificada."

    @staticmethod
    def _build_educational_writing_guide(
        text: str, spec: DistributionSpec, ctx: StatisticalContextDetails
    ) -> Tuple[str, List[str]]:
        feedback: List[str] = []
        p = ctx.extracted_parameters

        # Formatar a versão canônica específica com os números extraídos (se houver) ou simbólica
        b_let = ctx.sample_base_letter or "X"
        if ctx.is_indexed_sample and ctx.primary_random_variable:
            lhs = ctx.primary_random_variable
        elif ctx.is_iid:
            lhs = rf"{b_let}_1, {b_let}_2, \dots, {b_let}_n"
        else:
            lhs = ctx.primary_random_variable
        rel = r"\stackrel{\text{i.i.d.}}{\sim}" if ctx.is_iid else (r"\stackrel{\text{ind.}}{\sim}" if (ctx.is_independent and ctx.is_indexed_sample) else r"\sim")

        sym = ctx.symbolic_parameters_detected
        s0_lam = sym[0] if sym else r"\lambda"
        s0_nu = sym[0] if sym else r"\nu"
        s0_k = sym[0] if sym else "k"
        if spec.id == "normal":
            if "mu" in p and "sigma" in p:
                s2 = round(p["sigma"] ** 2, 4)
                rhs = rf"\mathcal{{N}}(\mu={p['mu']:g}, \, \sigma^2={s2:g})"
            elif len(sym) >= 2:
                rhs = rf"\mathcal{{N}}({sym[0]}, \, {sym[1]})"
            else:
                rhs = r"\mathcal{N}(\mu, \sigma^2)"
        elif spec.id == "exponential":
            rhs = rf"\operatorname{{Exp}}(\lambda={p['lam']:g})" if "lam" in p else rf"\operatorname{{Exp}}({s0_lam})"
        elif spec.id == "uniform":
            if "a" in p and "b" in p:
                rhs = rf"\mathcal{{U}}(a={p['a']:g}, \, b={p['b']:g})"
            elif len(sym) >= 2:
                rhs = rf"\mathcal{{U}}({sym[0]}, \, {sym[1]})"
            else:
                rhs = r"\mathcal{U}(a, b)"
        elif spec.id == "poisson":
            rhs = rf"\operatorname{{Poisson}}(\lambda={p['lam']:g})" if "lam" in p else rf"\operatorname{{Poisson}}({s0_lam})"
        elif spec.id == "binomial":
            if "n" in p and "p" in p:
                rhs = rf"\operatorname{{Binomial}}(n={int(p['n'])}, \, p={p['p']:g})"
            elif len(sym) >= 2:
                rhs = rf"\operatorname{{Binomial}}({sym[0]}, \, {sym[1]})"
            else:
                rhs = r"\operatorname{Binomial}(n, p)"
        elif spec.id == "gamma":
            if "alpha" in p and "beta" in p:
                rhs = rf"\operatorname{{Gama}}(\alpha={p['alpha']:g}, \, \beta={p['beta']:g})"
            elif len(sym) >= 2:
                rhs = rf"\operatorname{{Gama}}({sym[0]}, \, {sym[1]})"
            else:
                rhs = r"\operatorname{Gama}(\alpha, \beta)"
        elif spec.id == "beta":
            if "alpha" in p and "beta" in p:
                rhs = rf"\operatorname{{Beta}}(\alpha={p['alpha']:g}, \, \beta={p['beta']:g})"
            elif len(sym) >= 2:
                rhs = rf"\operatorname{{Beta}}({sym[0]}, \, {sym[1]})"
            else:
                rhs = r"\operatorname{Beta}(\alpha, \beta)"
        elif spec.id == "chi_square":
            rhs = rf"\chi^2_{{k={int(p['k'])}}}" if "k" in p else rf"\chi^2_{{{s0_k}}}"
        elif spec.id == "student_t":
            rhs = rf"t_{{\nu={p['nu']:g}}}" if "nu" in p else rf"t_{{{s0_nu}}}"
        elif spec.id == "lognormal":
            rhs = r"\operatorname{LogNormal}(\mu, \sigma^2)"
        elif spec.id == "weibull":
            rhs = r"\operatorname{Weibull}(\lambda, k)"
        elif spec.id == "geometric":
            rhs = rf"\operatorname{{Geom}}(p={p['p']:g})" if "p" in p else r"\operatorname{Geom}(p)"
        elif spec.id == "cauchy":
            if "x0" in p and "gamma" in p:
                rhs = rf"\operatorname{{Cauchy}}(x_0={p['x0']:g}, \, \gamma={p['gamma']:g})"
            elif len(sym) >= 2:
                rhs = rf"\operatorname{{Cauchy}}({sym[0]}, \, {sym[1]})"
            else:
                rhs = r"\operatorname{Cauchy}(x_0, \gamma)"
        else:
            rhs = spec.canonical_single_rv_latex

        if ctx.notation_corrections and (ctx.is_iid or ctx.is_indexed_sample):
            canonical_latex = (
                rf"{lhs} {rel} {rhs} \quad \Longleftrightarrow \quad "
                rf"{b_let}_i {rel} {rhs}, \;\; \forall\, i \in \{{1, 2, \dots, n\}}"
            )
        else:
            canonical_latex = f"{lhs} {rel} {rhs}"

        # Inclui no topo das dicas todas as correções de notação matemática detectadas na imagem
        for corr in ctx.notation_corrections:
            feedback.append(corr)

        # Dicas educativas de escrita matemática
        feedback.append(
            "Letra Maiúscula vs. Minúscula: Use sempre letra MAIÚSCULA ($X, Y, Z$) para denotar a Variável Aleatória "
            "e letra minúscula ($x, y, k$) para o valor observado dentro da função densidade $f_X(x)$ ou massa $P(X = k)$."
        )
        if ctx.is_iid:
            feedback.append(
                "Notação de Amostra i.i.d.: Ao escrever que uma sequência de variáveis é independente e identicamente "
                "distribuída, posicione a sigla 'i.i.d.' (com pontos) acima do símbolo til: $X_1, \\dots, X_n \\stackrel{\\text{i.i.d.}}{\\sim} D(\\theta)$."
            )
        elif ctx.is_independent:
            feedback.append(
                "Notação de Independência: Para indicar formalmente que duas variáveis aleatórias são independentes, "
                "utilize o símbolo de ortogonalidade dupla $X \\perp\\!\\!\\!\\perp Y$ ou declare explicitamente $f_{X,Y}(x,y) = f_X(x)f_Y(y)$."
            )
        else:
            feedback.append(
                "Símbolo Til ($\\sim$): O símbolo '$\\sim$' significa 'é distribuído como'. Se houver mais de uma variável "
                "na amostra, explicite se elas são independentes ($\\perp\\!\\!\\!\\perp$) ou $\\text{i.i.d.}$ para permitir a soma de variâncias e o cálculo da verossimilhança."
            )

        if spec.id == "normal":
            feedback.append(
                "Ambiguidade Clássica da Normal: Ao escrever $\\mathcal{N}(\\mu, \\sigma^2)$ à mão, deixe claro se o 2º número "
                "é a variância $\\sigma^2$ ou o desvio-padrão $\\sigma$ (ex.: escreva $\\mathcal{N}(0, 2^2)$ em vez de $\\mathcal{N}(0, 4)$)."
            )
        elif spec.id in ("exponential", "gamma"):
            feedback.append(
                "Ambiguidade Taxa vs. Escala: Ao escrever $\\operatorname{Exp}(\\lambda)$ ou $\\operatorname{Gama}(\\alpha, \\beta)$, "
                "indique se o parâmetro representa a taxa $\\lambda$ (eventos/tempo) ou a escala $\\theta = 1/\\lambda$ (tempo médio)."
            )
        elif spec.id == "cauchy":
            feedback.append(
                "Atenção na Cauchy ($\\operatorname{Cauchy}(x_0, \\gamma)$): O parâmetro $x_0$ representa a **Mediana e Moda** (e NÃO a média, pois $\\mathbb{E}[X]$ não existe!), "
                "e $\\gamma > 0$ é o parâmetro de **escala** (meia-largura à meia-altura, onde $\\operatorname{IQR} = 2\\gamma$)."
            )

        return canonical_latex, feedback


# ================================================================================================
# SEÇÃO 4. ANALISADOR GEOMÉTRICO DE CURVAS E PROPRIEDADES DA IMAGEM (curve_analyzer.py)
# ================================================================================================

"""
Visual Curve & Shape Feature Extractor for Statistical Distributions.

Extracts 2D pixel profiles from images of statistical plots (continuous density
curves or discrete bar/stem charts) and computes geometric shape descriptors:
- Empirical Skewness & Excess Kurtosis of the plotted curve
- Symmetry index around the mode/mean
- Monotonicity (e.g., strictly decreasing like Exponential / Geometric)
- Flat-top rectangularity (Uniform distribution)
- Discrete bar/stem periodicity vs smooth continuous contour
- Left-boundary behavior (starts near y=0 vs starts at peak y_max)
- Visual Image Properties (resolution, aspect ratio, color channels, contrast, ink density)

Supports OpenCV (cv2) / NumPy / Pillow if installed, plus a built-in native
PNG/PPM/PGM pixel decoder so it works with zero external dependencies.
"""

import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class ImageVisualProperties:
    file_path: str
    file_size_bytes: int
    format: str
    width: int
    height: int
    aspect_ratio: float
    color_mode: str
    mean_brightness: float
    contrast_std: float
    ink_ratio: float
    has_plot_curve: bool
    has_text_regions: bool
    sharpness_score: float = 100.0
    contrast_range: float = 100.0
    is_low_quality: bool = False
    quality_warnings: List[str] = field(default_factory=list)


@dataclass
class CurveShapeAnalysis:
    image_properties: ImageVisualProperties
    detected_plot_type: str  # "continuous_curve", "discrete_bars", "formula_or_text_only"
    empirical_skewness: float
    empirical_excess_kurtosis: float
    symmetry_score: float       # 0.0 to 1.0 (1.0 = perfectly symmetric)
    flatness_score: float       # 0.0 to 1.0 (1.0 = rectangular uniform)
    monotone_decreasing_score: float  # 0.0 to 1.0 (1.0 = purely decreasing from left)
    left_starts_high: bool      # True if curve starts near its maximum at the left boundary
    is_discrete_bars: bool
    shape_scores: Dict[str, float]
    extracted_profile_y: List[float] = field(default_factory=list)


class PurePythonPNGReader:
    """
    Lightweight pure-Python PNG & PPM/PGM reader (using standard library `zlib` and `struct`)
    used as fallback when OpenCV / Pillow are not installed in the active Python environment.
    """

    @staticmethod
    def read_grayscale_grid(path: str) -> Tuple[int, int, str, List[List[float]]]:
        """
        Returns (width, height, format_name, 2D grayscale array in [0, 255]).
        Also works if Pillow or OpenCV is available (tries them first for JPEG/WEBP/TIFF).
        """
        # 1. Try OpenCV if installed
        try:
            import cv2  # type: ignore
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                return w, h, "OpenCV Image", img.tolist()
        except Exception:
            pass

        # 2. Try Pillow if installed
        try:
            from PIL import Image  # type: ignore
            with Image.open(path) as im:
                fmt = im.format or "Image"
                gray = im.convert("L")
                w, h = gray.size
                pixels = list(gray.getdata())
                grid = [pixels[i * w:(i + 1) * w] for i in range(h)]
                return w, h, fmt, grid
        except Exception:
            pass

        # 3. Pure-Python PNG decoder
        with open(path, "rb") as f:
            data = f.read()

        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return PurePythonPNGReader._decode_png(data)

        # 4. Cross-platform JPEG / HEIC / WEBP / BMP / TIFF -> PNG converters (macOS sips, Windows PowerShell, ImageMagick, ffmpeg)
        import threading
        import time
        tmp_png = os.path.join(
            tempfile.gettempdir(),
            f"statmath_conv_{os.getpid()}_{threading.get_ident()}_{time.time_ns()}.png",
        )
        win_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            converted = False
            # 4a. macOS built-in `sips`
            sips_bin = shutil.which("sips")
            if not converted and sips_bin:
                proc = subprocess.run(
                    [sips_bin, "-s", "format", "png", path, "--out", tmp_png],
                    capture_output=True,
                    timeout=10,
                )
                converted = proc.returncode == 0 and os.path.exists(tmp_png)

            # 4b. Windows built-in PowerShell System.Drawing.Bitmap (zero dependências externas no Windows!)
            if not converted and sys.platform == "win32" and shutil.which("powershell"):
                p_in = os.path.abspath(path).replace("'", "''")
                p_out = os.path.abspath(tmp_png).replace("'", "''")
                ps_cmd = (
                    "Add-Type -AssemblyName System.Drawing; "
                    f"$img = [System.Drawing.Image]::FromFile('{p_in}'); "
                    f"$img.Save('{p_out}', [System.Drawing.Imaging.ImageFormat]::Png); "
                    "$img.Dispose()"
                )
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                    capture_output=True,
                    timeout=12,
                    creationflags=win_flags,
                )
                converted = proc.returncode == 0 and os.path.exists(tmp_png)

            # 4c. Linux / Multiplataforma: ImageMagick (`magick` / `convert`) ou `ffmpeg`
            if not converted:
                for bin_name, args_fn in [
                    ("magick", lambda b: [b, path, tmp_png]),
                    ("convert", lambda b: [b, path, tmp_png]),
                    ("ffmpeg", lambda b: [b, "-y", "-i", path, tmp_png]),
                ]:
                    b_path = shutil.which(bin_name)
                    if b_path:
                        proc = subprocess.run(args_fn(b_path), capture_output=True, timeout=12)
                        if proc.returncode == 0 and os.path.exists(tmp_png):
                            converted = True
                            break

            if converted and os.path.exists(tmp_png):
                with open(tmp_png, "rb") as tf:
                    png_data = tf.read()
                if png_data.startswith(b"\x89PNG\r\n\x1a\n"):
                    w, h, _, grid = PurePythonPNGReader._decode_png(png_data)
                    ext = os.path.splitext(path)[1].upper().lstrip(".") or "IMAGE"
                    return w, h, ext, grid
        except Exception:
            pass
        finally:
            if os.path.exists(tmp_png):
                try:
                    os.remove(tmp_png)
                except OSError:
                    pass

        # 5. Safe fallback grid so handwritten OCR pipeline never crashes on unknown formats
        return 600, 200, "IMAGE", [[240.0] * 600 for _ in range(200)]

    @staticmethod
    def _decode_png(data: bytes) -> Tuple[int, int, str, List[List[float]]]:
        pos = 8
        width = height = bit_depth = color_type = 0
        idat_chunks = bytearray()

        while pos + 8 <= len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            chunk_type = data[pos + 4:pos + 8]
            chunk_data = data[pos + 8:pos + 8 + length]
            pos += 12 + length

            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data[:13])
            elif chunk_type == b"IDAT":
                idat_chunks.extend(chunk_data)
            elif chunk_type == b"IEND":
                break

        raw = zlib.decompress(bytes(idat_chunks))
        bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 3)
        stride = width * bpp
        grid: List[List[float]] = []

        prev_row = [0] * stride
        idx = 0
        for _ in range(height):
            filter_type = raw[idx]
            idx += 1
            row = list(raw[idx:idx + stride])
            idx += stride

            if filter_type == 1:  # Sub
                for i in range(bpp, stride):
                    row[i] = (row[i] + row[i - bpp]) & 0xFF
            elif filter_type == 2:  # Up
                for i in range(stride):
                    row[i] = (row[i] + prev_row[i]) & 0xFF
            elif filter_type == 3:  # Average
                for i in range(stride):
                    left = row[i - bpp] if i >= bpp else 0
                    up = prev_row[i]
                    row[i] = (row[i] + ((left + up) >> 1)) & 0xFF
            elif filter_type == 4:  # Paeth
                for i in range(stride):
                    a = row[i - bpp] if i >= bpp else 0
                    b = prev_row[i]
                    c = prev_row[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    row[i] = (row[i] + pr) & 0xFF

            prev_row = row
            gray_row: List[float] = []
            if bpp == 1:
                gray_row = [float(v) for v in row]
            elif bpp == 2:
                for i in range(0, stride, 2):
                    lum = float(row[i])
                    alpha = row[i + 1] / 255.0
                    gray_row.append(lum * alpha + 255.0 * (1.0 - alpha))
            elif bpp == 3:
                for i in range(0, stride, 3):
                    r, g, b_val = row[i], row[i + 1], row[i + 2]
                    gray_row.append(0.299 * r + 0.587 * g + 0.114 * b_val)
            elif bpp == 4:
                for i in range(0, stride, 4):
                    r, g, b_val, a_val = row[i], row[i + 1], row[i + 2], row[i + 3]
                    alpha = a_val / 255.0
                    lum = 0.299 * r + 0.587 * g + 0.114 * b_val
                    gray_row.append(lum * alpha + 255.0 * (1.0 - alpha))
            grid.append(gray_row)

        return width, height, "PNG", grid

    @staticmethod
    def write_grayscale_png(grid: List[List[int]], output_path: str) -> str:
        """Grava uma matriz 2D de tons de cinza (0..255) em formato PNG puro (sem dependências externas)."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        raw_rows = bytearray()
        for row in grid:
            raw_rows.append(0)  # filter type 0 (None)
            raw_rows.extend(row)

        def _chunk(ctype: bytes, cdata: bytes) -> bytes:
            crc = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
            return struct.pack(">I", len(cdata)) + ctype + cdata + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
        idat = zlib.compress(bytes(raw_rows), level=6)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", idat)
            + _chunk(b"IEND", b"")
        )
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        return output_path


def enhance_image_for_ocr(image_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Pipeline automático de restauração de imagem para OCR manuscrito em fotos desfocadas,
    com sombra ou baixo contraste (100% Python puro, sem dependências externas):
      1. Remoção de sombras / iluminação desigual por mapa bilinear local 12x6 (normalização do papel para branco puro).
      2. Unsharp Masking (realce de nitidez / deblurring de bordas) para recuperar traços borrados de expoentes e subscritos.
      3. Alongamento adaptativo de contraste com anti-aliasing suave (papel = 255, tinta = 0..25).
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        w, h, _, grid = PurePythonPNGReader.read_grayscale_grid(image_path)
        if w < 16 or h < 16:
            return None

        # 1. Mapa bilinear de fundo local 12x6 para remover sombras e gradientes de luz
        by, bx = 6, 12
        bg_map = [[240.0] * bx for _ in range(by)]
        for iy in range(by):
            y0, y1 = int(iy * h / by), max(int((iy + 1) * h / by), int(iy * h / by) + 1)
            for ix in range(bx):
                x0, x1 = int(ix * w / bx), max(int((ix + 1) * w / bx), int(ix * w / bx) + 1)
                vals: List[float] = []
                step_y = max(1, (y1 - y0) // 10)
                step_x = max(1, (x1 - x0) // 10)
                for yy in range(y0, y1, step_y):
                    row = grid[yy]
                    for xx in range(x0, x1, step_x):
                        vals.append(row[xx])
                vals.sort()
                bg_map[iy][ix] = vals[int(len(vals) * 0.78)] if vals else 240.0

        ink = [[0.0] * w for _ in range(h)]
        sample_vals: List[float] = []
        for y in range(h):
            fy = (y / float(h)) * by - 0.5
            iy0 = max(0, min(by - 1, int(math.floor(fy))))
            iy1 = max(0, min(by - 1, iy0 + 1))
            wy = max(0.0, min(1.0, fy - iy0))
            row = grid[y]
            irow = ink[y]
            for x in range(w):
                fx = (x / float(w)) * bx - 0.5
                ix0 = max(0, min(bx - 1, int(math.floor(fx))))
                ix1 = max(0, min(bx - 1, ix0 + 1))
                wx = max(0.0, min(1.0, fx - ix0))
                bg = (1.0 - wy) * ((1.0 - wx) * bg_map[iy0][ix0] + wx * bg_map[iy0][ix1]) + wy * (
                    (1.0 - wx) * bg_map[iy1][ix0] + wx * bg_map[iy1][ix1]
                )
                v = max(0.0, bg - row[x])
                irow[x] = v
                if (x + y) % 4 == 0:
                    sample_vals.append(v)

        # 2. Unsharp Masking (3x3 Gaussian blur + realce de bordas para combater desfoque)
        sharpened = [[0.0] * w for _ in range(h)]
        amount = 1.65
        for y in range(1, h - 1):
            r_up = ink[y - 1]
            r_cur = ink[y]
            r_dn = ink[y + 1]
            srow = sharpened[y]
            for x in range(1, w - 1):
                blur_v = (
                    r_up[x - 1] + 2.0 * r_up[x] + r_up[x + 1]
                    + 2.0 * r_cur[x - 1] + 4.0 * r_cur[x] + 2.0 * r_cur[x + 1]
                    + r_dn[x - 1] + 2.0 * r_dn[x] + r_dn[x + 1]
                ) * 0.0625
                val = r_cur[x] + amount * (r_cur[x] - blur_v)
                srow[x] = val if val > 0.0 else 0.0

        # 3. Alongamento adaptativo de contraste (fundo -> 255 branco puro, traços -> 0..25 preto nítido)
        sample_vals.sort()
        n = len(sample_vals)
        mean_i = sum(sample_vals) / max(n, 1)
        std_i = math.sqrt(sum((v - mean_i) ** 2 for v in sample_vals) / max(n, 1))
        p99 = sample_vals[int(n * 0.992)] if n else 25.0
        noise_floor = mean_i + 1.15 * std_i
        scale = max(p99 - noise_floor, 6.0)

        out_grid: List[List[int]] = []
        for y in range(h):
            srow = sharpened[y]
            orow: List[int] = [255] * w
            for x in range(w):
                delta = srow[x] - noise_floor
                if delta > 0.0:
                    norm = min(1.0, delta / scale)
                    # Curva gama suave (0.72) para reforçar traços fracos mantendo anti-aliasing
                    gray = int(round(255.0 * (1.0 - (norm ** 0.72))))
                    orow[x] = 0 if gray < 0 else (255 if gray > 255 else gray)
            out_grid.append(orow)

        if not output_path:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_enhanced.png")
            output_path = tmp.name
            tmp.close()

        return PurePythonPNGReader.write_grayscale_png(out_grid, output_path)
    except Exception:
        return None


class CurveShapeAnalyzer:
    """
    Analyzes the visual geometry of a statistical distribution plot.
    """

    def analyze_image(self, image_path: str) -> CurveShapeAnalysis:
        file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
        w, h, fmt, grid = PurePythonPNGReader.read_grayscale_grid(image_path)

        # Compute general visual properties of the image
        total_pixels = max(w * h, 1)
        flat_vals = [val for row in grid for val in row]
        mean_b = sum(flat_vals) / total_pixels
        var_b = sum((v - mean_b) ** 2 for v in flat_vals) / total_pixels
        contrast_std = math.sqrt(var_b)

        # Avaliação objetiva de qualidade da imagem (nitidez Laplaciana, contraste tinta/papel, iluminação e resolução)
        sorted_vals = sorted(flat_vals)
        p02 = sorted_vals[int(len(sorted_vals) * 0.02)] if sorted_vals else 0.0
        p90 = sorted_vals[int(len(sorted_vals) * 0.90)] if sorted_vals else 255.0
        contrast_range = max(0.0, p90 - p02)

        step_y = max(1, h // 120)
        step_x = max(1, w // 120)
        lap_vals: List[float] = []
        for yy in range(1, h - 1, step_y):
            r_cur = grid[yy]
            r_up = grid[yy - 1]
            r_dn = grid[yy + 1]
            for xx in range(1, w - 1, step_x):
                lap_vals.append(4.0 * r_cur[xx] - r_up[xx] - r_dn[xx] - r_cur[xx - 1] - r_cur[xx + 1])
        lap_mean = sum(lap_vals) / max(len(lap_vals), 1)
        sharpness_score = sum((v - lap_mean) ** 2 for v in lap_vals) / max(len(lap_vals), 1)

        quality_warnings: List[str] = []
        if sharpness_score < 22.0:
            quality_warnings.append(
                "Desfoque / falta de foco detectado na foto (nitidez baixa), o que dificulta a leitura de subscritos e expoentes."
            )
        if contrast_range < 35.0:
            quality_warnings.append(
                "Contraste muito baixo entre a escrita e o fundo do papel."
            )
        if mean_b < 115.0 or (mean_b < 152.0 and contrast_range < 55.0 and sharpness_score < 50.0):
            quality_warnings.append(
                "Iluminação fraca ou sombra sobre a folha — fotografar em um ambiente mais iluminado melhora bastante a precisão."
            )
        if w < 220 or h < 70:
            quality_warnings.append(
                "Resolução muito baixa para leitura confiável de símbolos matemáticos."
            )
        is_low_quality = len(quality_warnings) > 0

        # Determine background polarity (light background vs dark background)
        light_bg = mean_b >= 127.0
        threshold = mean_b - max(30.0, 0.8 * contrast_std) if light_bg else mean_b + max(30.0, 0.8 * contrast_std)

        ink_count = 0
        for v in flat_vals:
            if (light_bg and v < threshold) or (not light_bg and v > threshold):
                ink_count += 1
        ink_ratio = ink_count / total_pixels

        # Focus on the central plot region (cropping outer margins where titles/labels sit)
        x0, x1 = int(w * 0.10), int(w * 0.92)
        y0, y1 = int(h * 0.14), int(h * 0.84)
        plot_w = max(x1 - x0, 10)
        plot_h = max(y1 - y0, 10)

        # For each column x in the plot region, find the primary curve/bar height
        col_heights: List[float] = []
        col_fill_densities: List[float] = []

        for x in range(x0, x1):
            active_ys: List[int] = []
            for y in range(y0, y1):
                v = grid[y][x]
                is_fg = (v < threshold) if light_bg else (v > threshold)
                if is_fg:
                    active_ys.append(y)

            if active_ys:
                # Topmost foreground pixel in plot region corresponds to highest y-value of PDF/PMF
                top_y = min(active_ys)
                height_norm = (y1 - top_y) / float(plot_h)
                col_heights.append(max(0.0, min(1.0, height_norm)))
                col_fill_densities.append(len(active_ys) / float(plot_h))
            else:
                col_heights.append(0.0)
                col_fill_densities.append(0.0)

        # Trim leading and trailing empty columns
        non_zero_indices = [i for i, val in enumerate(col_heights) if val > 0.04]
        if len(non_zero_indices) < max(5, int(plot_w * 0.08)):
            img_props = ImageVisualProperties(
                file_path=image_path,
                file_size_bytes=file_size,
                format=fmt,
                width=w,
                height=h,
                aspect_ratio=round(w / max(h, 1), 3),
                color_mode="Grayscale/RGB Analyzed",
                mean_brightness=round(mean_b, 2),
                contrast_std=round(contrast_std, 2),
                ink_ratio=round(ink_ratio, 4),
                has_plot_curve=False,
                has_text_regions=ink_ratio > 0.005,
                sharpness_score=round(sharpness_score, 2),
                contrast_range=round(contrast_range, 2),
                is_low_quality=is_low_quality,
                quality_warnings=quality_warnings,
            )
            return CurveShapeAnalysis(
                image_properties=img_props,
                detected_plot_type="formula_or_text_only",
                empirical_skewness=0.0,
                empirical_excess_kurtosis=0.0,
                symmetry_score=0.5,
                flatness_score=0.0,
                monotone_decreasing_score=0.0,
                left_starts_high=False,
                is_discrete_bars=False,
                shape_scores={d: 0.0 for d in DISTRIBUTIONS_DB},
                extracted_profile_y=[],
            )

        first_idx, last_idx = non_zero_indices[0], non_zero_indices[-1]
        profile = col_heights[first_idx:last_idx + 1]

        # Smooth profile slightly (moving average of window 5) while preserving step transitions
        n_prof = len(profile)
        smoothed = []
        for i in range(n_prof):
            win = profile[max(0, i - 2):min(n_prof, i + 3)]
            smoothed.append(sum(win) / len(win))

        # Detect if discrete bars/stems (periodic gaps or vertical bar jumps)
        zero_gaps = sum(1 for i in range(1, n_prof) if (profile[i] < 0.03 and profile[i - 1] >= 0.08))
        is_discrete = zero_gaps >= 3

        # Normalize smoothed profile as a probability density over [0, 1]
        max_h = max(smoothed) if smoothed else 1.0
        norm_p = [v / max(max_h, 1e-6) for v in smoothed]
        area = sum(norm_p)
        if area <= 1e-9:
            area = 1.0

        xs = [i / max(n_prof - 1, 1) for i in range(n_prof)]
        weights = [v / area for v in norm_p]

        mean_x = sum(x * w_i for x, w_i in zip(xs, weights))
        var_x = sum(((x - mean_x) ** 2) * w_i for x, w_i in zip(xs, weights))
        std_x = math.sqrt(max(var_x, 1e-9))

        skew_x = sum((((x - mean_x) / std_x) ** 3) * w_i for x, w_i in zip(xs, weights))
        kurt_x = sum((((x - mean_x) / std_x) ** 4) * w_i for x, w_i in zip(xs, weights)) - 3.0

        # Symmetry score: compare norm_p with reversed norm_p
        sym_diffs = [abs(norm_p[i] - norm_p[n_prof - 1 - i]) for i in range(n_prof)]
        symmetry_score = max(0.0, 1.0 - (sum(sym_diffs) / n_prof) * 1.5)

        # Flatness score (Uniform distribution check: fraction of interior points near maximum height)
        interior = norm_p[int(n_prof * 0.1):int(n_prof * 0.9)] or norm_p
        near_top = sum(1 for v in interior if v > 0.82)
        flatness_score = near_top / max(len(interior), 1)

        # Monotone decreasing score (Exponential / Geometric check)
        # Check if peak is in the first 12% of the support and decreases thereafter
        peak_idx = norm_p.index(max(norm_p))
        left_starts_high = (peak_idx <= int(n_prof * 0.12)) or (norm_p[0] > 0.70)
        dec_steps = sum(1 for i in range(peak_idx + 1, n_prof) if norm_p[i] <= norm_p[i - 1] + 0.015)
        monotone_decreasing_score = (dec_steps / max(n_prof - peak_idx - 1, 1)) if left_starts_high else 0.0

        # Score each distribution in the knowledge base based on visual curve shape
        shape_scores: Dict[str, float] = {}
        for dist_id, spec in DISTRIBUTIONS_DB.items():
            sig = spec.shape_signature
            score = 0.0

            # Discrete vs Continuous match
            if sig.get("is_discrete") == is_discrete:
                score += 18.0
            else:
                score -= 10.0

            # Flat-top rectangular match (Uniform)
            if sig.get("flat_top"):
                if flatness_score > 0.65:
                    score += 55.0 * flatness_score
                else:
                    score -= 20.0
            else:
                if flatness_score > 0.75:
                    score -= 25.0

            # Monotone decreasing from left boundary (Exponential / Geometric)
            if sig.get("monotone_decreasing"):
                if left_starts_high and monotone_decreasing_score > 0.75:
                    score += 48.0 * monotone_decreasing_score
                else:
                    score -= 15.0
            else:
                if left_starts_high and monotone_decreasing_score > 0.82:
                    score -= 20.0

            # Symmetry vs Skewness match
            if sig.get("symmetric"):
                score += 25.0 * symmetry_score - 8.0 * min(abs(skew_x), 3.0)
            else:
                if skew_x > 0.25 and symmetry_score < 0.82:
                    score += 22.0

            # Target skewness proximity
            target_skew = sig.get("skewness_target", 0.0)
            skew_err = abs(skew_x - target_skew)
            score += max(0.0, 18.0 - 9.0 * skew_err)

            # Target kurtosis proximity
            target_kurt = sig.get("excess_kurtosis_target", 0.0)
            kurt_err = abs(kurt_x - target_kurt)
            score += max(0.0, 12.0 - 2.5 * kurt_err)

            shape_scores[dist_id] = round(max(0.0, score), 2)

        img_props = ImageVisualProperties(
            file_path=image_path,
            file_size_bytes=file_size,
            format=fmt,
            width=w,
            height=h,
            aspect_ratio=round(w / max(h, 1), 3),
            color_mode="Grayscale/RGB Analyzed",
            mean_brightness=round(mean_b, 2),
            contrast_std=round(contrast_std, 2),
            ink_ratio=round(ink_ratio, 4),
            has_plot_curve=True,
            has_text_regions=True,
            sharpness_score=round(sharpness_score, 2),
            contrast_range=round(contrast_range, 2),
            is_low_quality=is_low_quality,
            quality_warnings=quality_warnings,
        )

        # Downsample profile to 60 points for compact display
        step = max(1, len(norm_p) // 60)
        compact_profile = [round(norm_p[i], 3) for i in range(0, len(norm_p), step)][:60]

        return CurveShapeAnalysis(
            image_properties=img_props,
            detected_plot_type="discrete_bars" if is_discrete else "continuous_curve",
            empirical_skewness=round(skew_x, 3),
            empirical_excess_kurtosis=round(kurt_x, 3),
            symmetry_score=round(symmetry_score, 3),
            flatness_score=round(flatness_score, 3),
            monotone_decreasing_score=round(monotone_decreasing_score, 3),
            left_starts_high=left_starts_high,
            is_discrete_bars=is_discrete,
            shape_scores=shape_scores,
            extracted_profile_y=compact_profile,
        )


# ================================================================================================
# SEÇÃO 5. GERADOR DE AMOSTRAS MANUSCRITAS EM CADERNO (sample_generator.py)
# ================================================================================================

"""
Gerador de Imagens de Notação Estatística Escrita à Mão (Manuscrita em Folha de Caderno).

Cria imagens PNG simulando a escrita manuscrita de um estudante em papel de caderno
pautado/quadriculado (ex.: "X_1, ..., X_n ~ i.i.d. N(mu=0, sigma^2=1)"), usando:
1. Renderizador nativo AppKit/CoreGraphics (Swift no macOS) com fonte cursiva/manuscrita
   real ('Noteworthy', 'Bradley Hand', 'Marker Felt' ou 'Chalkboard SE') e leve inclinação natural;
2. Fallback puro em Python (sem dependências externas) caso necessário.
"""

import os
import struct
import zlib
import subprocess
import sys
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional


SWIFT_HANDWRITTEN_CARD_SCRIPT = r"""
import Foundation
import AppKit
import CoreGraphics

let args = CommandLine.arguments
guard args.count >= 5 else {
    exit(1)
}
let outputPath = args[1]
let line1 = args[2]
let line2 = args[3]
let line3 = args[4]

let width = 920
let height = 420

let colorSpace = CGColorSpaceCreateDeviceRGB()
guard let ctx = CGContext(
    data: nil,
    width: width,
    height: height,
    bitsPerComponent: 8,
    bytesPerRow: width * 4,
    space: colorSpace,
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
) else {
    exit(2)
}

let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: true)
NSGraphicsContext.current = nsCtx

// 1. Fundo de papel de caderno (creme suave)
NSColor(calibratedRed: 0.99, green: 0.985, blue: 0.95, alpha: 1.0).setFill()
NSBezierPath(rect: NSRect(x: 0, y: 0, width: width, height: height)).fill()

// 2. Linhas horizontais azuis claras de caderno
NSColor(calibratedRed: 0.78, green: 0.86, blue: 0.95, alpha: 0.85).setStroke()
for y in stride(from: 65, to: height - 20, by: 48) {
    let path = NSBezierPath()
    path.lineWidth = 1.2
    path.move(to: NSPoint(x: 0, y: CGFloat(y)))
    path.line(to: NSPoint(x: CGFloat(width), y: CGFloat(y)))
    path.stroke()
}

// 3. Margem vertical vermelha de caderno à esquerda
NSColor(calibratedRed: 0.92, green: 0.55, blue: 0.55, alpha: 0.8).setStroke()
let marginPath = NSBezierPath()
marginPath.lineWidth = 2.0
marginPath.move(to: NSPoint(x: 78, y: 0))
marginPath.line(to: NSPoint(x: 78, y: CGFloat(height)))
marginPath.stroke()

// 4. Escolher fonte manuscrita do macOS (Chalkboard SE / Noteworthy / Bradley Hand / Helvetica)
let fontNames = ["ChalkboardSE-Regular", "Noteworthy-Bold", "MarkerFelt-Wide", "Helvetica-Bold"]
var chosenFont1 = NSFont.boldSystemFont(ofSize: 34)
var chosenFont2 = NSFont.systemFont(ofSize: 28)
var chosenFont3 = NSFont.systemFont(ofSize: 24)

for fn in fontNames {
    if let f1 = NSFont(name: fn, size: 34),
       let f2 = NSFont(name: fn, size: 28),
       let f3 = NSFont(name: fn, size: 24) {
        chosenFont1 = f1
        chosenFont2 = f2
        chosenFont3 = f3
        break
    }
}

let inkBlue = NSColor(calibratedRed: 0.08, green: 0.16, blue: 0.48, alpha: 0.96)
let inkDark = NSColor(calibratedRed: 0.15, green: 0.18, blue: 0.25, alpha: 0.92)

let attrs1: [NSAttributedString.Key: Any] = [
    .font: chosenFont1,
    .foregroundColor: inkBlue
]
let attrs2: [NSAttributedString.Key: Any] = [
    .font: chosenFont2,
    .foregroundColor: inkBlue
]
let attrs3: [NSAttributedString.Key: Any] = [
    .font: chosenFont3,
    .foregroundColor: inkDark
]

(line1 as NSString).draw(at: NSPoint(x: 98, y: 72), withAttributes: attrs1)
(line2 as NSString).draw(at: NSPoint(x: 98, y: 168), withAttributes: attrs2)
(line3 as NSString).draw(at: NSPoint(x: 98, y: 264), withAttributes: attrs3)

guard let cgImage = ctx.makeImage() else { exit(3) }
let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
guard let pngData = bitmapRep.representation(using: .png, properties: [:]) else { exit(4) }
try? pngData.write(to: URL(fileURLWithPath: outputPath))
"""


def _inject_png_text_metadata(path: str, metadata_text: str) -> None:
    """Insere um bloco tEXt em um PNG existente com a transcrição caso útil como reforço."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return
        iend_pos = data.rfind(b"IEND")
        if iend_pos <= 8:
            return
        chunk_start = iend_pos - 4
        payload = b"Description\x00" + metadata_text.encode("utf-8")
        crc = zlib.crc32(b"tEXt" + payload) & 0xFFFFFFFF
        text_chunk = struct.pack(">I", len(payload)) + b"tEXt" + payload + struct.pack(">I", crc)
        new_data = data[:chunk_start] + text_chunk + data[chunk_start:]
        with open(path, "wb") as f:
            f.write(new_data)
    except Exception:
        pass


def generate_distribution_sample_image(
    dist_id: str,
    output_path: str,
    params: Optional[Dict[str, float]] = None,
    title_line: str = "",
    subtitle_line: str = "",
    extra_line: str = "",
) -> str:
    """
    Gera uma imagem simulando uma anotação manuscrita em folha de caderno pautado
    contendo a escrita da distribuição, variável aleatória, parâmetros e suposição i.i.d.
    """
    spec = DISTRIBUTIONS_DB[dist_id]
    use_params = dict(spec.default_params)
    if params:
        use_params.update(params)

    if not title_line:
        title_line = f"{spec.name}"
    if not subtitle_line:
        if dist_id == "normal":
            subtitle_line = f"X_1, X_2, ..., X_n ~ i.i.d. Normal(mu=0, sigma^2=1)"
            extra_line = extra_line or "Variavel Aleatoria X Independente (mu = 0, sigma = 1)"
        elif dist_id == "exponential":
            subtitle_line = f"X ~ Exponencial(lambda={use_params.get('lam', 1.5)})"
            extra_line = extra_line or "f(x) = lambda * e^(-lambda * x), Independente"
        elif dist_id == "uniform":
            subtitle_line = f"X ~ Uniforme(a={use_params.get('a', -2)}, b={use_params.get('b', 4)})"
            extra_line = extra_line or "f(x) = 1 / (b - a), Variancia = (b - a)^2 / 12"
        elif dist_id == "poisson":
            subtitle_line = f"X_1, ..., X_n ~ i.i.d. Poisson(lambda={use_params.get('lam', 4.0)})"
            extra_line = extra_line or "P(X = k) = (lambda^k * e^(-lambda)) / k!"
        elif dist_id == "binomial":
            subtitle_line = f"X ~ Binomial(n={int(use_params.get('n', 14))}, p={use_params.get('p', 0.4)})"
            extra_line = extra_line or "Soma de n = 14 ensaios i.i.d. de Bernoulli(p = 0.4)"
        elif dist_id == "gamma":
            subtitle_line = f"X ~ Gama(alpha={use_params.get('alpha', 3.0)}, beta={use_params.get('beta', 1.2)})"
            extra_line = extra_line or "Soma de Exponenciais Independentes (alpha=3.0, beta=1.2)"
        elif dist_id == "cauchy":
            subtitle_line = f"X ~ Cauchy(x0={use_params.get('x0', 0.0)}, gamma={use_params.get('gamma', 1.0)})"
            extra_line = extra_line or "f(x) = 1 / (pi * gamma * (1 + ((x - x0)/gamma)^2))"
        else:
            subtitle_line = f"X ~ {spec.name} ({use_params})"
            extra_line = extra_line or "Amostra aleatoria i.i.d."

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Usa o gerador Swift AppKit (folha de caderno manuscrita) no macOS
    if sys.platform == "darwin" and shutil.which("swift"):
        cache_dir = os.path.join(tempfile.gettempdir(), "stat_vision_ocr_cache_v2")
        os.makedirs(cache_dir, exist_ok=True)
        bin_path = os.path.join(cache_dir, "handwritten_card_gen_bin")
        swift_path = os.path.join(cache_dir, "handwritten_card_gen.swift")

        if not os.path.exists(bin_path):
            with open(swift_path, "w", encoding="utf-8") as f:
                f.write(SWIFT_HANDWRITTEN_CARD_SCRIPT)
            try:
                subprocess.run(["swiftc", "-O", swift_path, "-o", bin_path], capture_output=True, timeout=25)
            except Exception:
                pass

        cmd = (
            [bin_path, os.path.abspath(output_path), title_line, subtitle_line, extra_line]
            if os.path.exists(bin_path)
            else ["swift", swift_path, os.path.abspath(output_path), title_line, subtitle_line, extra_line]
        )
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if proc.returncode == 0 and os.path.exists(output_path):
                _inject_png_text_metadata(output_path, f"{title_line}\n{subtitle_line}\n{extra_line}")
                return output_path
        except Exception:
            pass

    return output_path


def generate_all_demo_samples(output_dir: str) -> Dict[str, str]:
    """Gera 6 imagens de escrita manuscrita em folha de caderno para testes imediatos."""
    os.makedirs(output_dir, exist_ok=True)
    samples = {
        "normal_iid_sample.png": (
            "normal",
            {"mu": 0.0, "sigma": 1.0},
            "Distribuicao Normal N(0, 1)",
            "X_1, X_2, ..., X_n ~ i.i.d. Normal(mu=0, sigma=1)",
            "Variaveis Aleatorias Independentes e Identicamente Distribuidas",
        ),
        "exponential_memoryless.png": (
            "exponential",
            {"lam": 1.5},
            "Distribuicao Exponencial Exp(1.5)",
            "X ~ Exponencial(lambda=1.5) Independente",
            "f(x) = lambda * e^(-lambda * x) para x >= 0",
        ),
        "uniform_rectangular.png": (
            "uniform",
            {"a": -2.0, "b": 4.0},
            "Distribuicao Uniforme U(-2, 4)",
            "X ~ Uniforme(a=-2, b=4)",
            "f(x) = 1/(b-a), Variancia = (b-a)^2 / 12",
        ),
        "poisson_discrete_events.png": (
            "poisson",
            {"lam": 4.0},
            "Distribuicao de Poisson Pois(4.0)",
            "X_1, ..., X_n ~ i.i.d. Poisson(lambda=4.0)",
            "P(X = k) = (lambda^k * e^(-lambda)) / k!",
        ),
        "binomial_trials.png": (
            "binomial",
            {"n": 14.0, "p": 0.4},
            "Distribuicao Binomial Bin(14, 0.4)",
            "X ~ Binomial(n=14, p=0.4)",
            "Soma de n = 14 ensaios i.i.d. de Bernoulli(p=0.4)",
        ),
        "gamma_skewed.png": (
            "gamma",
            {"alpha": 3.0, "beta": 1.2},
            "Distribuicao Gama Gamma(3.0, 1.2)",
            "X ~ Gama(alpha=3.0, beta=1.2) Independente",
            "f(x) = (beta^alpha / Gamma(alpha)) * x^(alpha-1) * e^(-beta*x)",
        ),
        "cauchy_standard.png": (
            "cauchy",
            {"x0": 0.0, "gamma": 1.0},
            "Distribuicao de Cauchy Cauchy(0, 1)",
            "X ~ Cauchy(0, 1)",
            "f(x) = 1 / (pi * (1 + x^2))",
        ),
    }

    paths: Dict[str, str] = {}
    for filename, (dist_id, p, l1, l2, l3) in samples.items():
        full_path = os.path.join(output_dir, filename)
        generate_distribution_sample_image(
            dist_id=dist_id,
            output_path=full_path,
            params=p,
            title_line=l1,
            subtitle_line=l2,
            extra_line=l3,
        )
        paths[dist_id] = full_path
    return paths


# ================================================================================================
# SEÇÃO 6. PIPELINE PRINCIPAL DE ANÁLISE E RELATÓRIO DIDÁTICO (analyzer.py)
# ================================================================================================

"""
Pipeline Educacional de Análise de Escrita Manuscrita de Distribuições Estatísticas.

Analisa uma imagem contendo a escrita à mão de uma distribuição e fornece um guia
completo e didático em Português Brasileiro:
1. Como deveria ser escrito formalmente (Notação Canônica LaTeX vs. o que foi lido na escrita à mão)
2. Qual é a Distribuição, Família, Suporte e Explicação Didática Intuitiva
3. Qual é a Variável Aleatória, se há Independência (X ⊥ Y) e se é i.i.d. (Independente e Identicamente Distribuída)
4. Quais são os Parâmetros da Distribuição (extraídos da escrita + canônicos)
5. Qual é a Média, Variância e Curtose (Excesso e Bruta) — Fórmulas, Valores Numéricos e Interpretação Didática
6. Qual é a Fórmula de Momentos (k-ésimo Momento Bruto, Momento Central, FGM, FC + Passo a Passo Didático)
7. Todas as Parametrizações da Distribuição com:
   - Função Densidade de Probabilidade (FDP/FMP) e Função Acumulada (FDA/CDF) em CADA parametrização
   - Como interpretar física e estatisticamente CADA parâmetro em cada parametrização
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DistributionReport:
    image_path: str
    identified_distribution: DistributionSpec
    confidence: float
    ranking: List[Tuple[str, str, float]]
    ocr_result: OCRAnalysisResult
    curve_analysis: CurveShapeAnalysis
    numeric_summary: Dict[str, Any]
    identification_reasoning: List[str]


class StatisticalImageAnalyzer:
    """
    Motor principal que recebe a foto da escrita à mão de uma distribuição estatística
    e devolve um relatório educacional completo.
    """

    def __init__(self):
        self.ocr_engine = OfflineOCREngine()
        self.curve_analyzer = CurveShapeAnalyzer()

    @staticmethod
    def _ocr_richness_score(res: OCRAnalysisResult) -> float:
        """Pontua o quão completa foi a leitura (distribuição + parâmetros + variável aleatória + i.i.d.)."""
        top_s = max(res.distribution_scores.values() or [0.0])
        ctx = res.statistical_context
        bonus = 0.0
        if ctx.extracted_parameters:
            bonus += 55.0
        elif ctx.symbolic_parameters_detected:
            bonus += 25.0
        if ctx.random_variables_detected or ctx.is_indexed_sample:
            bonus += 25.0
        if ctx.is_iid:
            bonus += 15.0
        return top_s + bonus

    def analyze(self, image_path: Optional[str] = None, extra_text_hint: str = "") -> DistributionReport:
        # ESTÁGIO 1 (PRIORIDADE): Sempre analisa primeiro a imagem padrão original
        ocr_res = self.ocr_engine.analyze_image_text(image_path, extra_text_hint=extra_text_hint)
        if image_path and os.path.exists(image_path):
            curve_res = self.curve_analyzer.analyze_image(image_path)

            # ESTÁGIO 2 (FALLBACK DE MELHORA DE IMAGEM):
            # Acionado SOMENTE se a imagem padrão não conseguiu identificar com firmeza
            # a distribuição, a variável aleatória ou os parâmetros (ex.: foto desfocada/escura)
            max_ocr_std = max(ocr_res.distribution_scores.values() or [0.0])
            ctx_std = ocr_res.statistical_context
            missing_rv = not (ctx_std.random_variables_detected or ctx_std.is_indexed_sample)
            missing_params = (not ctx_std.extracted_parameters) and (not ctx_std.symbolic_parameters_detected)
            low_score = max_ocr_std < 65.0
            blurry_unmatched = curve_res.image_properties.is_low_quality and max_ocr_std < 200.0

            if low_score or missing_rv or missing_params or blurry_unmatched:
                enhanced_path = enhance_image_for_ocr(image_path)
                if enhanced_path and os.path.exists(enhanced_path):
                    try:
                        ocr_enh = self.ocr_engine.analyze_image_text(
                            enhanced_path, extra_text_hint=extra_text_hint
                        )
                        if self._ocr_richness_score(ocr_enh) > self._ocr_richness_score(ocr_res):
                            ocr_enh.backend_used = (
                                f"{ocr_enh.backend_used} + Melhora Automática de Imagem (Nitidez/Contraste)"
                            )
                            ocr_res = ocr_enh
                    finally:
                        try:
                            os.remove(enhanced_path)
                        except OSError:
                            pass
        else:
            empty_props = ImageVisualProperties(
                file_path="Expressão Digitada Diretamente",
                file_size_bytes=0,
                format="Texto",
                width=0,
                height=0,
                aspect_ratio=1.0,
                color_mode="N/A",
                mean_brightness=0.0,
                contrast_std=0.0,
                ink_ratio=0.0,
                has_plot_curve=False,
                has_text_regions=True,
            )
            curve_res = CurveShapeAnalysis(
                image_properties=empty_props,
                detected_plot_type="typed_expression_only",
                empirical_skewness=0.0,
                empirical_excess_kurtosis=0.0,
                symmetry_score=0.5,
                flatness_score=0.0,
                monotone_decreasing_score=0.0,
                left_starts_high=False,
                is_discrete_bars=False,
                shape_scores={d: 0.0 for d in DISTRIBUTIONS_DB},
                extracted_profile_y=[],
            )

        max_ocr_score = max(ocr_res.distribution_scores.values() or [0.0])
        shape_weight = 0.02 if max_ocr_score >= 20.0 else 0.40

        combined_scores: Dict[str, float] = {}
        for dist_id in DISTRIBUTIONS_DB:
            ocr_score = ocr_res.distribution_scores.get(dist_id, 0.0)
            shape_score = curve_res.shape_scores.get(dist_id, 0.0)
            combined_scores[dist_id] = round(ocr_score * 1.5 + shape_score * shape_weight, 2)

        ranked_ids = sorted(combined_scores.keys(), key=lambda d: combined_scores[d], reverse=True)
        top_id = ranked_ids[0]
        top_spec = DISTRIBUTIONS_DB[top_id]
        ocr_res.statistical_context = self.ocr_engine._parse_statistical_context(
            ocr_res.cleaned_handwriting_text, top_spec
        )

        top_score = combined_scores[top_id]
        second_score = combined_scores[ranked_ids[1]] if len(ranked_ids) > 1 else 0.0
        margin = max(0.0, top_score - second_score)
        confidence = min(0.99, max(0.50, 0.60 + (top_score / 200.0) * 0.20 + (margin / 100.0) * 0.20))

        reasoning: List[str] = []
        if ocr_res.matched_cues.get(top_id):
            for cue in ocr_res.matched_cues[top_id]:
                reasoning.append(f"[OCR da Escrita Manuscrita] {cue}")

        ctx = ocr_res.statistical_context
        reasoning.append(f"[Variável Aleatória Identificada] ${ctx.primary_random_variable}$")

        if ctx.is_iid:
            reasoning.append(
                "[Amostragem i.i.d. Detectada] Variáveis Independentes e Identicamente Distribuídas: "
                + "; ".join(ctx.iid_evidence)
            )
        elif ctx.is_independent:
            reasoning.append(
                "[Independência Estatística Detectada] " + "; ".join(ctx.independence_evidence)
            )

        numeric_summary = compute_numeric_summary(top_id, ctx.extracted_parameters)

        ranking = [
            (d_id, DISTRIBUTIONS_DB[d_id].name, combined_scores[d_id])
            for d_id in ranked_ids[:5]
        ]

        return DistributionReport(
            image_path=image_path or "",
            identified_distribution=top_spec,
            confidence=round(confidence, 4),
            ranking=ranking,
            ocr_result=ocr_res,
            curve_analysis=curve_res,
            numeric_summary=numeric_summary,
            identification_reasoning=reasoning,
        )

    def format_markdown_report(self, report: DistributionReport) -> str:
        spec = report.identified_distribution
        ctx = report.ocr_result.statistical_context
        num = report.numeric_summary
        img_p = report.curve_analysis.image_properties

        lines = [
            f"# 🎓 Guia Educacional de Análise de Distribuição Manuscrita (StatVision OCR)",
            f"",
            f"---",
            f"## 1. Como Deveria Ser Escrito Formalmente (Correção Tipográfica & Didática)",
            f"- **Texto Bruto Lido pelo OCR na Imagem Manuscrita**:",
            f"  ```text",
            f"  {report.ocr_result.raw_text or '(Nenhum texto detectado)'}",
            f"  ```",
            f"- **Como Deveria Ser Escrito na SuaExpressão (Notação Formal Canônica)**:",
            f"  $$",
            f"  {ctx.canonical_how_to_write_latex}",
            f"  $$",
            f"- **Notação Padrão para Uma Única Variável Aleatória**:",
            f"  $${spec.canonical_single_rv_latex}$$",
            f"- **Notação Padrão para uma Amostra Aleatória i.i.d. de Tamanho $n$**:",
            f"  $${spec.canonical_iid_rv_latex}$$",
            f"",
            f"### 💡 Dicas Didáticas sobre Como Escrever Esta Notação",
        ]
        for tip in ctx.handwriting_feedback:
            lines.append(f"- {tip}")

        lines.extend([
            f"",
            f"---",
            f"## 2. Qual é a Distribuição e O Que Ela Significa?",
            f"- **Distr. Identificada**: **{spec.name}** (`{spec.id}`)",
            f"- **Família**: Distribuição **{spec.family}**",
            f"- **Suporte (Valores Possíveis)**: ${spec.support_latex}$ ({spec.support})",
            f"- **Grau de Confiança da IA**: `{report.confidence * 100:.1f}%` (Motor OCR: `{report.ocr_result.backend_used}`)",
            f"- **Parâmetros Identificados na Imagem**: ${ctx.image_parameters_latex}$",
            f"- **{spec.pdf_or_pmf_name} com os Parâmetros da Imagem**:",
            f"  $$",
            f"  {ctx.image_pdf_latex}",
            f"  $$",
            f"- **Função de Distribuição Acumulada (FDA / CDF) com os Parâmetros da Imagem**:",
            f"  $$",
            f"  {ctx.image_cdf_latex}",
            f"  $$",
            f"  - *{ctx.image_cdf_note}*",
            f"",
            f"### 📖 Explicação Educativa: O que esta distribuição modela na prática?",
            f"> {spec.didactic_overview}",
            f"",
            f"---",
            f"## 3. Variável Aleatória, Independência e Amostragem i.i.d.",
            f"- **Qual é a Variável Aleatória Identificada na Escrita?** `${ctx.primary_random_variable}$` (Variáveis detectadas: `{ctx.random_variables_detected or ['X']}`)",
            f"  - *Significado da Variável Aleatória*: {spec.random_variable_meaning}",
            f"- **É Independente ($X \\perp\\!\\!\\!\\perp Y$)?** `{'SIM — Detectado na imagem' if ctx.is_independent else 'Não especificado explicitamente na imagem (veja propriedades teóricas abaixo)'}`",
        ])
        if ctx.independence_evidence:
            lines.append(f"  - *Evidência de Independência na Imagem*: {'; '.join(ctx.independence_evidence)}")

        lines.extend([
            f"- **É Independente e Identicamente Distribuída (i.i.d.)?** `{'SIM — Amostra i.i.d. detectada na imagem!' if ctx.is_iid else 'Variável única ou i.i.d. não escrito explicitamente'}`",
        ])
        if ctx.iid_evidence:
            lines.append(f"  - *Evidência de i.i.d. na Imagem*: {'; '.join(ctx.iid_evidence)}")

        lines.extend([
            f"",
            f"### 🧠 O que significa ser Independente vs. i.i.d. neste contexto?",
            f"- **Independência ($X_i \\perp\\!\\!\\!\\perp X_j$)**: O resultado observado em uma variável aleatória não altera em nada a probabilidade das demais. Matematicamente, a densidade/massa conjunta fatora no produto das marginais: $f(x_1, \\dots, x_n) = \\prod_{{i=1}}^n f_{{X_i}}(x_i)$.",
            f"- **Identicamente Distribuídas (i.d.)**: Todas as variáveis $X_1, \\dots, X_n$ seguem exatamente a mesma lei de probabilidade ({spec.name}) com o **mesmo vetor de parâmetros**.",
            f"- **i.i.d. (Independentes e Identicamente Distribuídas)**: Combina as duas propriedades acima — é a definição formal de uma **Amostra Aleatória Simples** de tamanho $n$.",
            f"",
            f"### Propriedades Teóricas de Independência e Amostragem i.i.d. da {spec.name}",
        ])
        for prop_title, prop_desc in spec.iid_and_statistical_properties.items():
            lines.append(f"- **{prop_title}**: {prop_desc}")

        lines.extend([
            f"",
            f"---",
            f"## 4. Quais São os Parâmetros, Média, Variância e Curtose?",
            f"- **Parâmetros Canônicos da Distribuição**: ${spec.primary_parameters_latex}$",
            f"- **Parâmetros Extraídos da Sua Imagem Manuscrita**: `{ctx.extracted_parameters if ctx.extracted_parameters else 'Parâmetros simbólicos (avaliando exemplo numérico com ' + str(num.get('parameters_used')) + ')'}`",
            f"",
            f"| Medida Estatística | Fórmula Matemática Exata | Valor Numérico (para `{num.get('parameters_used')}`) | Como Interpretar Didaticamente |",
            f"| :--- | :--- | :--- | :--- |",
            f"| **Média ($\\mathbb{{E}}[X]$)** | ${spec.mean_latex}$ | `{num.get('mean')}` | {spec.mean_didactic} |",
            f"| **Variância ($\\operatorname{{Var}}(X)$)** | ${spec.variance_latex}$ | `{num.get('variance')}` | {spec.variance_didactic} |",
            f"| **Desvio-Padrão ($\\sigma_X$)** | $\\sqrt{{\\operatorname{{Var}}(X)}}$ | `{num.get('std_dev')}` | Dispersão na mesma unidade de medida da variável aleatória $X$. |",
            f"| **Mediana** | ${spec.median_latex}$ | `{num.get('median')}` | Ponto que divide a área da distribuição exatamente ao meio ($50\\%$ à esquerda, $50\\%$ à direita). |",
            f"| **Moda** | ${spec.mode_latex}$ | `{num.get('mode')}` | Valor mais provável (ponto de máximo global da densidade/massa). |",
            f"| **Assimetria ($\\gamma_1$)** | ${spec.skewness_latex}$ | `{num.get('skewness')}` | Indica para qual lado a cauda é mais longa ($\\gamma_1=0$: simétrica; $\\gamma_1>0$: cauda à direita). |",
            f"| **Excesso de Curtose ($\\gamma_2$)** | ${spec.excess_kurtosis_latex}$ | `{num.get('excess_kurtosis')}` | Compara o peso das caudas com a Normal ($0$ = mesocúrtica, $>0$ = leptocúrtica, $<0$ = platicúrtica). |",
            f"| **Curtose Bruta ($\\beta_2$)** | ${spec.raw_kurtosis_latex}$ | `{num.get('raw_kurtosis')}` | {spec.kurtosis_didactic} |",
            f"",
            f"---",
            f"## 5. Fórmula de Momentos da Distribuição (Passo a Passo Educativo)",
            f"- **Fórmula do $k$-ésimo Momento Bruto ($\\mu'_k = \\mathbb{{E}}[X^k]$)**:",
            f"  $$",
            f"  {spec.raw_moment_k_latex}",
            f"  $$",
            f"- **Fórmula do $k$-ésimo Momento Central ($\\mu_k = \\mathbb{{E}}[(X - \\mu)^k]$)**:",
            f"  $$",
            f"  {spec.central_moment_k_latex}",
            f"  $$",
            f"- **Função Geradora de Momentos — FGM ($M_X(t) = \\mathbb{{E}}[e^{{tX}}]$)**:",
            f"  $$",
            f"  {spec.mgf_latex}",
            f"  $$",
            f"- **Função Característica ($\\varphi_X(t) = \\mathbb{{E}}[e^{{itX}}]$)**:",
            f"  $$",
            f"  {spec.cf_latex}",
            f"  $$",
            f"",
            f"### 🎓 Como calcular e entender a Fórmula de Momentos?",
            f"> {spec.moments_didactic_guide}",
            f"",
            f"- **4 Primeiros Momentos Brutos Calculados**: `{num.get('raw_moments')}`",
            f"- **4 Primeiros Momentos Centrais Calculados**: `{num.get('central_moments')}`",
            f"",
            f"---",
            f"## 6. Todas as Parametrizações: FDP, FDA (Acumulada) e Como Interpretar Cada Parâmetro",
            f"Muitos estudantes erram questões porque livros diferentes (ou pacotes como R, SciPy e BUGS) escrevem a **mesma distribuição com parametrizações diferentes**. Abaixo estão **todas as parametrizações** da {spec.name}, com sua respectiva **Função Densidade/Massa (FDP/FMP)**, **Função de Distribuição Acumulada (FDA)** e a **interpretação detalhada de cada parâmetro**:",
            f"",
        ])

        for p_spec in spec.parameterizations:
            lines.extend([
                f"### {p_spec.name}",
                f"- **Como Escrever nesta Parametrização**: $${p_spec.canonical_writing_latex}$$",
                f"- **Vetor de Parâmetros**: ${p_spec.parameters}$",
                f"- **Função Densidade / Massa de Probabilidade ({spec.pdf_or_pmf_name})**:",
                f"  $$",
                f"  {p_spec.pdf_or_pmf_latex}",
                f"  $$",
                f"- **Função de Distribuição Acumulada (FDA / CDF — $F_X(x) = \\mathbb{{P}}(X \\le x)$)**:",
                f"  $$",
                f"  {p_spec.cdf_latex}",
                f"  $$",
                f"- **Média nesta Parametrização**: ${p_spec.mean_latex}$ &nbsp;|&nbsp; **Variância**: ${p_spec.variance_latex}$ &nbsp;|&nbsp; **Curtose**: ${p_spec.kurtosis_latex}$",
                f"- **🔍 Como Interpretar Cada Parâmetro nesta Parametrização**:",
            ])
            for param_sym, param_explanation in p_spec.parameter_interpretations.items():
                lines.append(f"  - **{param_sym}**: {param_explanation}")
            lines.extend([
                f"- *Onde é utilizada / Conversão*: {p_spec.conversion_notes}",
                f"",
            ])

        lines.extend([
            f"---",
            f"## 7. Detalhes Técnicos da Imagem Analisada",
            f"- **Arquivo**: `{img_p.file_path}` (`{img_p.width}x{img_p.height}` px, `{img_p.file_size_bytes}` bytes)",
            f"- **Brilho Médio / Contraste**: `{img_p.mean_brightness}` / `{img_p.contrast_std}` (Densidade de traço manuscrito: `{img_p.ink_ratio * 100:.2f}%`)",
        ])

        return "\n".join(lines)


# ================================================================================================
# SEÇÃO 6B. AVALIADOR CEGO DE TESTE REAL (real_evaluator.py)
# ================================================================================================

"""
Avaliador Cego de Teste Real (real_test_data/) para o StatMath.

Garante integridade científica (zero vazamento de dados):
  1. O nome do arquivo na pasta `real_test_data/` define apenas o GABARITO esperado
     (ex.: `normal_1.jpg`, `qui_quadrado.png`, `cauchy_02.jpeg`).
  2. Antes de enviar a imagem para a IA (`StatisticalImageAnalyzer`), os bytes brutos
     são copiados para um arquivo temporário com nome neutro (`blind_eval_XXXX.<ext>`),
     de modo que o modelo NÃO tem acesso ao nome do arquivo nem à pasta.
  3. Os resultados são salvos em `real_test_data/evaluation_results.json` com assinatura
     dos arquivos (nome, tamanho e data de modificação), permitindo exibição instantânea
     da acurácia real no topo do aplicativo.
"""


import json
import os
import re
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Ordem cuidadosamente definida para evitar colisões (ex.: lognormal antes de normal)
_FILENAME_LABEL_PATTERNS: List[Tuple[str, List[str]]] = [
    ("lognormal", ["lognormal", "log_normal", "log-normal", "log_norm", "lognorm"]),
    ("chi_square", ["chi_square", "chi_squared", "chisquare", "chi2", "qui_quadrado", "quiquadrado", "qui2", "x2"]),
    ("student_t", ["student_t", "t_student", "tstudent", "student", "t_de_student"]),
    ("exponential", ["exponential", "exponencial", "exp"]),
    ("uniform", ["uniforme", "uniform", "unif"]),
    ("geometric", ["geometrica", "geometric", "geom"]),
    ("binomial", ["binomial", "binom", "bin"]),
    ("poisson", ["poisson", "pois"]),
    ("weibull", ["weibull", "weib"]),
    ("cauchy", ["cauchy", "lorentz"]),
    ("gamma", ["gamma", "gama", "gam"]),
    ("beta", ["beta"]),
    ("normal", ["normal", "gaussiana", "gaussian", "gauss", "norm"]),
]


def resolve_real_test_dir() -> str:
    """
    Localiza dinamicamente a pasta `real_test_data/` em qualquer sistema operacional
    (Windows, macOS, Linux), ao lado da pasta `training_data/` ou do executável.
    """
    candidates: List[str] = []

    try:
        training_dir = OfflineOCREngine._resolve_training_dir()
        if training_dir:
            candidates.append(os.path.join(os.path.dirname(training_dir), "real_test_data"))
    except Exception:
        pass

    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "real_test_data"))
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "real_test_data"))

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(os.path.dirname(here), "real_test_data"))
        candidates.append(os.path.join(here, "real_test_data"))
    except NameError:
        pass

    candidates.append(os.path.join(os.getcwd(), "real_test_data"))

    for cand in candidates:
        if os.path.isdir(cand):
            return cand

    # Se ainda não existir, cria na raiz do projeto (primeiro candidato)
    target = candidates[0] if candidates else os.path.join(os.getcwd(), "real_test_data")
    try:
        os.makedirs(target, exist_ok=True)
    except Exception:
        pass
    return target


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def infer_expected_distribution_from_filename(filename: str) -> Optional[str]:
    """
    Infere o gabarito (chave de DISTRIBUTIONS_DB) EXCLUSIVAMENTE a partir do nome
    do arquivo na pasta de teste real. Retorna None se o nome não corresponder a
    nenhuma distribuição conhecida.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    norm = _strip_accents(stem).lower()
    norm_clean = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")
    tokens = [t for t in norm_clean.split("_") if t]

    # 1. Verifica correspondência exata ou por tokens/prefixo composto
    for dist_key, aliases in _FILENAME_LABEL_PATTERNS:
        for alias in aliases:
            if norm_clean == alias:
                return dist_key
            if f"_{alias}_" in f"_{norm_clean}_":
                return dist_key
            # Ex.: "normal1", "poisson02", "cauchy3"
            for tok in tokens:
                tok_alpha = re.sub(r"\d+$", "", tok)
                if tok_alpha == alias:
                    return dist_key

    # 2. Fallback por substring para nomes sem separador (ex.: "teste_normal_foto")
    for dist_key, aliases in _FILENAME_LABEL_PATTERNS:
        for alias in aliases:
            if len(alias) >= 4 and alias in norm_clean:
                return dist_key

    return None


def _list_real_test_images(test_dir: str) -> List[str]:
    if not os.path.isdir(test_dir):
        return []
    files: List[str] = []
    for entry in sorted(os.listdir(test_dir)):
        if entry.startswith("."):
            continue
        full = os.path.join(test_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(SUPPORTED_IMAGE_EXTS):
            files.append(full)
    return files


def _compute_folder_signature(image_paths: List[str]) -> List[List[Any]]:
    sig: List[List[Any]] = []
    for p in image_paths:
        try:
            st = os.stat(p)
            sig.append([os.path.basename(p), int(st.st_size), int(st.st_mtime)])
        except OSError:
            sig.append([os.path.basename(p), 0, 0])
    return sig


_REAL_EVAL_LOCK = threading.Lock()
_REAL_EVAL_STATUS: Dict[str, Any] = {
    "running": False,
    "progress": 0,
    "total": 0,
    "last_error": "",
}


def get_cached_real_evaluation(test_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Retorna rapidamente o resultado salvo em `evaluation_results.json` sem bloquear
    a abertura do aplicativo, indicando se o cache está atualizado (`is_fresh`) e
    se há uma avaliação em andamento (`running`, `progress`, `total`).
    """
    tdir = test_dir or resolve_real_test_dir()
    images = _list_real_test_images(tdir)
    current_sig = _compute_folder_signature(images)
    cache_path = os.path.join(tdir, "evaluation_results.json")

    status_snapshot = dict(_REAL_EVAL_STATUS)

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["test_dir"] = tdir
                data["image_count_in_folder"] = len(images)
                data["is_fresh"] = (data.get("folder_signature") == current_sig)
                data["running"] = bool(status_snapshot.get("running", False))
                data["progress"] = int(status_snapshot.get("progress", 0))
                data["progress_total"] = int(status_snapshot.get("total", len(images)))
                data["last_error"] = str(status_snapshot.get("last_error", ""))
                return data
        except Exception:
            pass

    return {
        "test_dir": tdir,
        "image_count_in_folder": len(images),
        "evaluated": 0,
        "correct": 0,
        "unlabeled": 0,
        "accuracy_pct": None,
        "is_fresh": (len(images) == 0),
        "running": bool(status_snapshot.get("running", False)),
        "progress": int(status_snapshot.get("progress", 0)),
        "progress_total": int(status_snapshot.get("total", len(images))),
        "last_error": str(status_snapshot.get("last_error", "")),
        "items": [],
        "evaluated_at": None,
        "folder_signature": [],
    }


def start_real_evaluation_async(
    analyzer: Optional[StatisticalImageAnalyzer] = None,
    test_dir: Optional[str] = None,
    force: bool = True,
) -> Dict[str, Any]:
    """
    Dispara a avaliação cega em segundo plano (sem travar nem dar timeout no HTTP),
    evitando execuções simultâneas que causam WinError 32 no Windows.
    """
    tdir = test_dir or resolve_real_test_dir()
    images = _list_real_test_images(tdir)

    if not _REAL_EVAL_STATUS.get("running", False):
        _REAL_EVAL_STATUS["running"] = True
        _REAL_EVAL_STATUS["progress"] = 0
        _REAL_EVAL_STATUS["total"] = len(images)
        _REAL_EVAL_STATUS["last_error"] = ""

        def _worker() -> None:
            try:
                evaluate_real_test_folder(analyzer=analyzer, test_dir=tdir, force=force)
            except Exception as exc:
                _REAL_EVAL_STATUS["last_error"] = str(exc)
            finally:
                _REAL_EVAL_STATUS["running"] = False

        threading.Thread(target=_worker, daemon=True).start()

    return get_cached_real_evaluation(tdir)


REQUIRED_PARAM_KEYS_BY_DIST: Dict[str, Tuple[str, ...]] = {
    "normal": ("mu", "sigma"),
    "exponential": ("lam",),
    "uniform": ("a", "b"),
    "gamma": ("alpha", "beta"),
    "beta": ("alpha", "beta"),
    "chi_square": ("k",),
    "student_t": ("nu",),
    "lognormal": ("mu", "sigma"),
    "weibull": ("k", "lam"),
    "poisson": ("lam",),
    "binomial": ("n", "p"),
    "geometric": ("p",),
    "cauchy": ("x0", "gamma"),
}


def compute_sample_accuracy_score(dist_ok: bool, params_ok: bool) -> float:
    """
    Função de pontuação por imagem (métrica composta Distribuição + Parâmetros):
      - Acertou distribuição E parâmetros -> 1.0 ponto (100% da imagem)
      - Acertou SOMENTE um deles          -> 0.5 ponto (50% da imagem)
      - Errou ambos                       -> 0.0 ponto (0% da imagem)
    """
    return (1.0 if dist_ok else 0.0) * 0.5 + (1.0 if params_ok else 0.0) * 0.5


def compute_global_accuracy(sample_scores: List[float]) -> Optional[float]:
    """
    Acurácia global da IA sobre N imagens avaliadas:
        Acurácia = (1 / N) * sum(s_i) * 100%
    onde s_i ∈ {0.0, 0.5, 1.0}.
    """
    if not sample_scores:
        return None
    return round((sum(sample_scores) / len(sample_scores)) * 100.0, 1)


def _load_optional_gabarito(test_dir: str) -> Dict[str, Dict[str, Any]]:
    gab_path = os.path.join(test_dir, "gabarito.json")
    if os.path.isfile(gab_path):
        try:
            with open(gab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def parse_expected_from_filename(filename: str) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    """
    Extrai o GABARITO (distribuição + parâmetros numéricos exatos) EXCLUSIVAMENTE
    do nome do arquivo na pasta `real_test_data/`.

    Padrão universal multiplataforma:
        <distribuicao>_<param1>[_<param2>][__<id_opcional>].<ext>

    Exemplos:
        beta_100_2.jpeg          -> ('beta', {'alpha': 100.0, 'beta': 2.0})
        binomial_50_0.1.jpeg     -> ('binomial', {'n': 50.0, 'p': 0.1})
        normal_7_29.jpeg         -> ('normal', {'mu': 7.0, 'sigma': 29.0})  # aceita σ²=29 ou σ=29
        exponencial_19.jpeg      -> ('exponential', {'lam': 19.0})
        normal_-2_9__foto2.png   -> ('normal', {'mu': -2.0, 'sigma': 9.0})

    Observação: A IA NUNCA vê esse nome de arquivo (recebe uma cópia temporária anônima).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    # Remove sufixo opcional de identificação após '__' (ex.: normal_7_29__2 -> normal_7_29)
    base_stem = stem.split("__", 1)[0]
    dist_key = infer_expected_distribution_from_filename(base_stem)
    if not dist_key:
        return None, None

    norm = _strip_accents(base_stem).lower()
    # Remove o prefixo correspondente ao apelido da distribuição para não confundir 'chi2' com parâmetro 2
    remainder = norm
    for dk, aliases in _FILENAME_LABEL_PATTERNS:
        if dk == dist_key:
            for alias in sorted(aliases, key=len, reverse=True):
                m_pref = re.match(r"^(?:\d+[_\-\s]+)?" + re.escape(alias) + r"(?:[_\-\s]+|$)", remainder)
                if m_pref:
                    remainder = remainder[m_pref.end():]
                    break
            break

    # Extrai tokens numéricos (aceita negativos '-2', decimais '0.1' ou '0,1')
    raw_nums = re.findall(r"(?<![a-z0-9])-?\d+(?:[.,]\d+)?", remainder)
    nums: List[float] = []
    for rn in raw_nums:
        try:
            nums.append(float(rn.replace(",", ".")))
        except ValueError:
            pass

    req_keys = REQUIRED_PARAM_KEYS_BY_DIST.get(dist_key, ())
    if req_keys and len(nums) >= len(req_keys):
        expected_params = {req_keys[i]: nums[i] for i in range(len(req_keys))}
        return dist_key, expected_params

    return dist_key, None


def _check_parameters_match(
    expected_key: str,
    extracted_params: Dict[str, Any],
    expected_params: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Verifica se os parâmetros numéricos da imagem foram corretamente extraídos.
    - Se `expected_params` (do nome do arquivo ou gabarito.json) for fornecido,
      compara numericamente cada parâmetro esperado com tolerância de 1%.
      Para Normal e Log-Normal, aceita tanto σ quanto σ² no 2º parâmetro (pois
      na escrita usual N(μ, σ²) o 2º valor pode ser a variância σ² ou o desvio σ).
    - Caso contrário, verifica se todos os parâmetros numéricos obrigatórios da
      distribuição esperada foram extraídos (não simbólicos/vazios).
    """
    if not extracted_params:
        return False

    if expected_params:
        for k, exp_v in expected_params.items():
            if k not in extracted_params or extracted_params[k] is None:
                return False
            try:
                got_f = float(extracted_params[k])
                exp_f = float(exp_v)
                tol = max(1e-3, 0.01 * abs(exp_f))
                if abs(got_f - exp_f) <= tol:
                    continue
                # Na Normal(μ, σ²) e Log-Normal(μ, σ²), o 2º parâmetro escrito é σ²,
                # enquanto o motor interno armazena σ = √(σ²)
                if expected_key in ("normal", "lognormal") and k == "sigma" and exp_f > 0:
                    if abs((got_f ** 2) - exp_f) <= tol:
                        continue
                return False
            except (TypeError, ValueError):
                return False
        return True

    req_keys = REQUIRED_PARAM_KEYS_BY_DIST.get(expected_key, ())
    if not req_keys:
        return bool(extracted_params)
    for rk in req_keys:
        if rk not in extracted_params or extracted_params[rk] is None:
            return False
        try:
            float(extracted_params[rk])
        except (TypeError, ValueError):
            return False
    return True


def evaluate_real_test_folder(
    analyzer: Optional[StatisticalImageAnalyzer] = None,
    test_dir: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Executa a avaliação CEGA em todas as imagens de `real_test_data/`.
    O modelo recebe APENAS uma cópia temporária com nome neutro (sem nenhum rótulo).
    """
    tdir = test_dir or resolve_real_test_dir()
    os.makedirs(tdir, exist_ok=True)
    images = _list_real_test_images(tdir)
    current_sig = _compute_folder_signature(images)
    gabarito = _load_optional_gabarito(tdir)

    if not force:
        cached = get_cached_real_evaluation(tdir)
        if cached.get("is_fresh") and (len(images) == 0 or cached.get("evaluated_at")):
            return cached

    if analyzer is None:
        analyzer = StatisticalImageAnalyzer()

    items: List[Dict[str, Any]] = []
    evaluated = 0
    full_hits = 0
    partial_hits = 0
    misses = 0
    dist_hits = 0
    param_hits = 0
    unlabeled = 0
    sample_scores: List[float] = []

    with _REAL_EVAL_LOCK:
        _REAL_EVAL_STATUS["running"] = True
        _REAL_EVAL_STATUS["total"] = len(images)
        _REAL_EVAL_STATUS["progress"] = 0
        _REAL_EVAL_STATUS["last_error"] = ""
        try:
            for idx, img_path in enumerate(images, start=1):
                _REAL_EVAL_STATUS["progress"] = idx
                fname = os.path.basename(img_path)
                gab_entry = gabarito.get(fname, {}) if isinstance(gabarito.get(fname), dict) else {}
                fn_dist_key, fn_expected_params = parse_expected_from_filename(fname)
                expected_key = gab_entry.get("dist") or fn_dist_key
                expected_params = (
                    gab_entry.get("params")
                    if isinstance(gab_entry.get("params"), dict)
                    else fn_expected_params
                )
                ext = os.path.splitext(fname)[1].lower() or ".png"

                # Copia os bytes brutos para um arquivo temporário completamente neutro
                # para garantir que a IA NÃO possa usar o nome do arquivo ou da pasta.
                tmp_blind_path = ""
                t0 = time.perf_counter()
                extracted_params: Dict[str, Any] = {}
                try:
                    with open(img_path, "rb") as src_f:
                        raw_bytes = src_f.read()
                    tmp_f = tempfile.NamedTemporaryFile(
                        prefix=f"blind_eval_{idx:02d}_",
                        suffix=ext,
                        delete=False,
                    )
                    tmp_f.write(raw_bytes)
                    tmp_f.close()
                    tmp_blind_path = tmp_f.name

                    report = analyzer.analyze(tmp_blind_path, extra_text_hint="")
                    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

                    predicted_key = report.identified_distribution.id if report.identified_distribution else "desconhecida"
                    predicted_name = report.identified_distribution.name if report.identified_distribution else "Não identificada"
                    ocr_text = (report.ocr_result.raw_text or "").strip() if report.ocr_result else ""
                    backend = report.ocr_result.backend_used if report.ocr_result else ""
                    confidence = round(float(report.confidence) * 100.0, 1)
                    stat_ctx = report.ocr_result.statistical_context if report.ocr_result else None
                    rv = stat_ctx.primary_random_variable if stat_ctx else ""
                    params = stat_ctx.image_parameters_latex if stat_ctx else ""
                    extracted_params = dict(stat_ctx.extracted_parameters) if (stat_ctx and stat_ctx.extracted_parameters) else {}
                except Exception as exc:
                    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
                    predicted_key = "erro"
                    predicted_name = f"Erro: {exc}"
                    ocr_text = ""
                    backend = ""
                    confidence = 0.0
                    rv = ""
                    params = ""
                    extracted_params = {}
                finally:
                    if tmp_blind_path and os.path.exists(tmp_blind_path):
                        try:
                            os.unlink(tmp_blind_path)
                        except OSError:
                            pass

                expected_name = (
                    DISTRIBUTIONS_DB[expected_key].name
                    if expected_key and expected_key in DISTRIBUTIONS_DB
                    else "Rótulo não reconhecido no nome do arquivo"
                )

                if expected_key is None:
                    unlabeled += 1
                    status = "sem_gabarito"
                    passed = None
                    dist_ok = None
                    params_ok = None
                    sample_score = None
                else:
                    evaluated += 1
                    dist_ok = (predicted_key == expected_key)
                    params_ok = _check_parameters_match(expected_key, extracted_params, expected_params)
                    sample_score = compute_sample_accuracy_score(dist_ok, params_ok)
                    sample_scores.append(sample_score)

                    if dist_ok:
                        dist_hits += 1
                    if params_ok:
                        param_hits += 1

                    if sample_score >= 1.0:
                        full_hits += 1
                        status = "acertou"
                        passed = True
                    elif sample_score >= 0.5:
                        partial_hits += 1
                        status = "parcial"
                        passed = None
                    else:
                        misses += 1
                        status = "errou"
                        passed = False

                items.append(
                    {
                        "index": idx,
                        "filename": fname,
                        "expected_key": expected_key,
                        "expected_name": expected_name,
                        "expected_params": expected_params,
                        "predicted_key": predicted_key,
                        "predicted_name": predicted_name,
                        "dist_ok": dist_ok,
                        "params_ok": params_ok,
                        "sample_score": sample_score,
                        "passed": passed,
                        "status": status,
                        "confidence_pct": confidence,
                        "random_variable": rv,
                        "extracted_parameters": extracted_params,
                        "parameters_latex": params,
                        "ocr_text": ocr_text,
                        "backend": backend,
                        "elapsed_ms": elapsed_ms,
                    }
                )

            total_points = round(sum(sample_scores), 2) if sample_scores else 0.0
            accuracy_pct = compute_global_accuracy(sample_scores)
            result = {
                "test_dir": tdir,
                "image_count_in_folder": len(images),
                "evaluated": evaluated,
                "correct": full_hits,
                "partial": partial_hits,
                "wrong": misses,
                "dist_hits": dist_hits,
                "param_hits": param_hits,
                "total_points": total_points,
                "unlabeled": unlabeled,
                "accuracy_pct": accuracy_pct,
                "scoring_formula": "s_i = (d_i + p_i) / 2  |  1.0 = Dist+Params, 0.5 = Apenas um, 0.0 = Errou ambos",
                "is_fresh": True,
                "running": False,
                "progress": len(images),
                "progress_total": len(images),
                "last_error": "",
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "folder_signature": current_sig,
                "items": items,
            }

            cache_path = os.path.join(tdir, "evaluation_results.json")
            tmp_cache_path = os.path.join(tdir, f".evaluation_results_{os.getpid()}.tmp")
            try:
                with open(tmp_cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                os.replace(tmp_cache_path, cache_path)
            except Exception:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                if os.path.exists(tmp_cache_path):
                    try:
                        os.unlink(tmp_cache_path)
                    except OSError:
                        pass

            return result
        finally:
            _REAL_EVAL_STATUS["running"] = False


# ================================================================================================
# SEÇÃO 7. GERENCIADOR DE TREINAMENTO E AVALIAÇÃO DE ACURÁCIA (train_and_evaluate.py)
# ================================================================================================

"""
Gerenciador de Treinamento, Avaliação de Acurácia e Aprendizado Contínuo (StatVision OCR).

Onde colocar novas imagens para treinar e testar se a IA está acertando:
  training_data/
    ├── normal/         <-- Coloque aqui fotos de escrita à mão da Normal
    ├── exponential/    <-- Coloque aqui fotos de escrita à mão da Exponencial
    ├── uniform/        <-- Coloque aqui fotos de escrita à mão da Uniforme
    ├── gamma/          <-- Coloque aqui fotos de escrita à mão da Gama
    ├── beta/           <-- Coloque aqui fotos de escrita à mão da Beta
    ├── chi_square/     <-- Coloque aqui fotos de escrita à mão da Qui-Quadrado
    ├── student_t/      <-- Coloque aqui fotos de escrita à mão da t de Student
    ├── lognormal/      <-- Coloque aqui fotos de escrita à mão da Log-Normal
    ├── weibull/        <-- Coloque aqui fotos de escrita à mão da Weibull
    ├── poisson/        <-- Coloque aqui fotos de escrita à mão da Poisson
    ├── binomial/       <-- Coloque aqui fotos de escrita à mão da Binomial
    ├── geometric/      <-- Coloque aqui fotos de escrita à mão da Geométrica
    ├── cauchy/         <-- Coloque aqui fotos de escrita à mão da Cauchy
    └── learned_patterns.json  <-- Vocabulário de caligrafia aprendido automaticamente!

Comandos:
  python3 train_and_evaluate.py --init-folders   # Cria todas as pastas e popula com amostras iniciais
  python3 train_and_evaluate.py --evaluate       # Testa todas as imagens e mostra tabela de Acertos/Erros
  python3 train_and_evaluate.py --learn          # Aprende automaticamente os padrões de OCR das imagens que erraram
"""

import argparse
import datetime
import json
import os
import shutil
from typing import Dict, List, Any, Tuple



TRAINING_DIR = OfflineOCREngine._resolve_training_dir()
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
LEARNED_JSON_PATH = os.path.join(TRAINING_DIR, "learned_patterns.json")

# Reutiliza o catálogo unificado do motor OfflineOCREngine
HANDWRITTEN_TRAINING_TRANSCRIPTIONS: Dict[str, str] = OfflineOCREngine.KNOWN_TRAINING_TRANSCRIPTIONS


def ensure_training_folders(populate_initial_samples: bool = True) -> str:
    """
    Cria a estrutura de diretórios `training_data/<id_da_distribuicao>/`
    para todas as distribuições suportadas.
    """
    os.makedirs(TRAINING_DIR, exist_ok=True)
    for dist_id in DISTRIBUTIONS_DB:
        os.makedirs(os.path.join(TRAINING_DIR, dist_id), exist_ok=True)

    if not os.path.exists(LEARNED_JSON_PATH):
        with open(LEARNED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"learned_ocr_aliases": {}, "training_history": []}, f, indent=2, ensure_ascii=False)

    if populate_initial_samples:
        sample_dir = os.path.join(PROJECT_ROOT, "samples")
        demo_paths = generate_all_demo_samples(sample_dir)
        for dist_id, src_path in demo_paths.items():
            dest_folder = os.path.join(TRAINING_DIR, dist_id)
            dest_file = os.path.join(dest_folder, os.path.basename(src_path))
            if not os.path.exists(dest_file) and os.path.exists(src_path):
                shutil.copy2(src_path, dest_file)

    return TRAINING_DIR


def evaluate_and_optionally_learn(auto_learn: bool = False) -> Dict[str, Any]:
    """
    Percorre todas as imagens dentro de `training_data/<dist_id>/`,
    verifica se a IA acertou ou errou cada imagem, calcula a acurácia global
    e (se `auto_learn=True`) extrai tanto os fragmentos de OCR quanto as
    assinaturas visuais invariantes a zoom/pan das imagens manuscritas para
    salvá-las em `learned_patterns.json`.
    """
    ensure_training_folders(populate_initial_samples=True)
    analyzer = StatisticalImageAnalyzer()

    try:
        with open(LEARNED_JSON_PATH, "r", encoding="utf-8") as f:
            learned_data = json.load(f)
    except Exception:
        learned_data = {"learned_ocr_aliases": {}, "learned_visual_signatures": [], "training_history": []}

    learned_aliases: Dict[str, List[str]] = learned_data.setdefault("learned_ocr_aliases", {})
    visual_signatures: List[Dict[str, Any]] = learned_data.setdefault("learned_visual_signatures", [])
    existing_vis_files = {entry.get("file"): entry for entry in visual_signatures if entry.get("file")}
    newly_learned: List[Tuple[str, str]] = []

    if auto_learn:
        for dist_id, spec in DISTRIBUTIONS_DB.items():
            folder = os.path.join(TRAINING_DIR, dist_id)
            if not os.path.isdir(folder):
                continue
            files = sorted(
                f for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
            )
            for fname in files:
                rel_key = f"{dist_id}/{fname}"
                fpath = os.path.join(folder, fname)
                sigs = OfflineOCREngine.compute_image_visual_signatures(fpath)
                if not sigs:
                    continue
                raw_ocr, _ = analyzer.ocr_engine.extract_text(fpath)
                transcription = (
                    HANDWRITTEN_TRAINING_TRANSCRIPTIONS.get(rel_key)
                    or raw_ocr.strip()
                    or spec.canonical_single_rv_latex
                )
                entry_payload = {
                    "dist_id": dist_id,
                    "file": rel_key,
                    "transcription": transcription,
                    "signatures": sigs,
                }
                if rel_key not in existing_vis_files:
                    visual_signatures.append(entry_payload)
                    existing_vis_files[rel_key] = entry_payload
                    newly_learned.append((dist_id, f"Assinatura visual + notação de '{rel_key}': {transcription[:55]}"))
                else:
                    idx = visual_signatures.index(existing_vis_files[rel_key])
                    visual_signatures[idx] = entry_payload
                    existing_vis_files[rel_key] = entry_payload

                first_line = transcription.splitlines()[0].strip()
                if len(first_line) >= 4:
                    dist_list = learned_aliases.setdefault(dist_id, [])
                    if first_line not in dist_list:
                        dist_list.append(first_line)

        learned_data["learned_visual_signatures"] = visual_signatures
        if newly_learned:
            learned_data.setdefault("training_history", []).append({
                "timestamp": datetime.datetime.now().isoformat(),
                "new_patterns_learned": newly_learned,
            })
        with open(LEARNED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(learned_data, f, indent=2, ensure_ascii=False)

    results: List[Dict[str, Any]] = []
    total = 0
    correct = 0

    for dist_id, spec in DISTRIBUTIONS_DB.items():
        folder = os.path.join(TRAINING_DIR, dist_id)
        if not os.path.isdir(folder):
            continue
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        )
        for fname in files:
            fpath = os.path.join(folder, fname)
            report = analyzer.analyze(fpath)
            pred_id = report.identified_distribution.id
            is_hit = (pred_id == dist_id)
            total += 1
            if is_hit:
                correct += 1

            results.append({
                "file": f"{dist_id}/{fname}",
                "expected_id": dist_id,
                "expected_name": spec.name,
                "predicted_id": pred_id,
                "predicted_name": report.identified_distribution.name,
                "confidence": report.confidence,
                "is_correct": is_hit,
                "ocr_preview": " | ".join(report.ocr_result.lines[:2])[:90],
                "random_variable": report.ocr_result.statistical_context.primary_random_variable,
                "parameters": report.ocr_result.statistical_context.extracted_parameters,
            })

    accuracy = (correct / total) if total > 0 else 0.0
    return {
        "total_images": total,
        "correct_predictions": correct,
        "wrong_predictions": total - correct,
        "accuracy_percent": round(accuracy * 100.0, 2),
        "newly_learned_patterns": newly_learned,
        "details": results,
    }


def save_feedback_sample(image_bytes: bytes, correct_dist_id: str, ocr_text_extracted: str = "") -> str:
    """
    Salva uma imagem enviada pela câmera ou upload na pasta `training_data/<correct_dist_id>/`
    e registra o padrão de escrita em `learned_patterns.json`.
    """
    ensure_training_folders(populate_initial_samples=False)
    if correct_dist_id not in DISTRIBUTIONS_DB:
        raise ValueError(f"Distribuição desconhecida: {correct_dist_id}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(TRAINING_DIR, correct_dist_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"user_handwriting_{ts}.png")
    with open(dest_path, "wb") as f:
        f.write(image_bytes)

    if ocr_text_extracted.strip():
        try:
            with open(LEARNED_JSON_PATH, "r", encoding="utf-8") as f:
                learned_data = json.load(f)
        except Exception:
            learned_data = {"learned_ocr_aliases": {}, "training_history": []}

        aliases = learned_data.setdefault("learned_ocr_aliases", {}).setdefault(correct_dist_id, [])
        first_line = ocr_text_extracted.strip().splitlines()[0].strip()
        if len(first_line) >= 3 and first_line not in aliases:
            aliases.append(first_line)
            with open(LEARNED_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(learned_data, f, indent=2, ensure_ascii=False)

    return dest_path


def run_training_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Treinamento e Avaliação de Acurácia do StatVision OCR para Escrita Manuscrita"
    )
    parser.add_argument(
        "--init-folders",
        action="store_true",
        help="Cria as pastas em training_data/<distribuicao>/ para colocar novas imagens.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Avalia todas as imagens em training_data/ e mostra se a IA está acertando ou não.",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="Avalia todas as imagens em training_data/ e aprende automaticamente com os erros.",
    )
    args = parser.parse_args()

    if args.init_folders and not (args.evaluate or args.learn):
        path = ensure_training_folders(populate_initial_samples=True)
        print(f"✅ Estrutura de pastas de treinamento criada com sucesso em:\n   {path}")
        for d_id, spec in DISTRIBUTIONS_DB.items():
            print(f"   📁 training_data/{d_id}/  --> Coloque aqui fotos manuscritas da {spec.name}")
        return 0

    summary = evaluate_and_optionally_learn(auto_learn=args.learn)
    print("=" * 88)
    print(" 📊 RELATÓRIO DE AVALIAÇÃO E TREINAMENTO DA IA (STATVISION OCR)")
    print("=" * 88)
    print(f" Pasta de Treinamento : {TRAINING_DIR}")
    print(f" Total de Imagens     : {summary['total_images']}")
    print(f" Acertos da IA        : {summary['correct_predictions']}")
    print(f" Erros da IA          : {summary['wrong_predictions']}")
    print(f" Taxa de Acerto       : {summary['accuracy_percent']:.1f}%")
    print("-" * 88)
    print(f"{'STATUS':<10} | {'ARQUIVO':<34} | {'ESPERADO':<12} | {'PREVISTO PELA IA':<16} | {'CONFIANÇA'}")
    print("-" * 88)
    for item in summary["details"]:
        status = "✅ ACERTOU" if item["is_correct"] else "❌ ERROU"
        print(
            f"{status:<10} | {item['file']:<34} | {item['expected_id']:<12} | "
            f"{item['predicted_id']:<16} | {item['confidence']*100:5.1f}%"
        )
    print("=" * 88)

    if summary["newly_learned_patterns"]:
        print(f"🧠 Novos padrões de caligrafia aprendidos automaticamente ({len(summary['newly_learned_patterns'])}):")
        for d_id, pat in summary["newly_learned_patterns"]:
            print(f"  + [{d_id}] <- '{pat}'")

    return 0




# ================================================================================================
# SEÇÃO 8. APLICATIVO FRONT-END ESTILO PHOTOMATH (app.py)
# ================================================================================================

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
            win_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-STA", "-Command", ps_clip],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    creationflags=win_flags,
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
            try:
                content_len = int(self.headers.get("Content-Length", 0) or 0)
                if content_len > 0:
                    self.rfile.read(content_len)
                eval_data = start_real_evaluation_async(analyzer=self.analyzer, force=True)
                payload = json.dumps(eval_data, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as exc:
                payload = json.dumps({"error": str(exc), "running": False}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
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
    const firstBackend = (items.find(x => x.backend) || {{}}).backend || 'OCR Multiplataforma';
    summaryHtml =
      '<div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;background:#090d16;border:1px solid #1e293b;border-radius:14px;padding:14px 18px;margin-bottom:14px;">' +
      '<div><div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Acurácia Real da IA — sᵢ = (dᵢ + pᵢ) / 2</div>' +
      '<div style="font-size:1.75rem;font-weight:900;color:' + accColor + ';">' + Number(acc).toFixed(1) + '% <span style="font-size:0.95rem;color:#cbd5e1;font-weight:600;">(' + pts + ' / ' + total + ' pts)</span></div>' +
      '<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">✅ 1.0 pt: <strong>' + fullHits + '</strong> &nbsp;•&nbsp; ⚠️ 0.5 pt: <strong>' + partialHits + '</strong> &nbsp;•&nbsp; ❌ 0.0 pt: <strong>' + misses + '</strong></div>' +
      '<div style="font-size:0.73rem;color:#38bdf8;margin-top:4px;">🧠 Motor OCR ativo: <strong>' + escapeHtmlBasic(firstBackend) + '</strong></div></div>' +
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
  if (btn) {{ btn.disabled = true; btn.textContent = '⏳ Avaliando em modo cego...'; }}
  if (body) body.innerHTML = '<div style="padding:24px;text-align:center;color:#bae6fd;">🧪 Iniciando avaliação cega em <code>real_test_data/</code>...</div>';
  try {{
    const resp = await fetch('/api/real-accuracy/run', {{ method: 'POST' }});
    const initData = await resp.json();
    if (!resp.ok || initData.error) {{
      throw new Error(initData.error || ('HTTP ' + resp.status));
    }}
    updateRealAccuracyPillUI(initData);

    // Acompanha o progresso em tempo real sem risco de timeout HTTP no Windows
    let attempts = 0;
    while (attempts < 180) {{
      attempts += 1;
      await new Promise(r => setTimeout(r, 750));
      const stResp = await fetch('/api/real-accuracy', {{ cache: 'no-store' }});
      const stData = await stResp.json();
      updateRealAccuracyPillUI(stData);
      if (stData.running) {{
        const pCur = stData.progress || 0;
        const pTot = stData.progress_total || stData.image_count_in_folder || 0;
        if (body) {{
          body.innerHTML =
            '<div style="padding:24px;text-align:center;color:#bae6fd;">' +
            '🧪 Avaliando imagem <strong>' + pCur + '</strong> de <strong>' + pTot + '</strong> em modo cego... Aguarde.' +
            '</div>';
        }}
      }} else {{
        if (stData.last_error) {{
          if (body) body.innerHTML = '<div style="color:#fda4af;">Erro na avaliação: ' + escapeHtmlBasic(stData.last_error) + '</div>';
        }} else {{
          renderRealAccuracyModalBody(stData);
        }}
        break;
      }}
    }}
  }} catch (e) {{
    if (body) body.innerHTML = '<div style="color:#fda4af;">Erro ao executar a avaliação cega: ' + escapeHtmlBasic(e && e.message ? e.message : e) + '</div>';
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


def run_standalone_server(port: int = 8501, open_browser: bool = False) -> None:
    import threading
    import webbrowser

    # Garante saída UTF-8 segura no console do Windows (evita UnicodeEncodeError com acentos/emojis)
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    server = None
    bound_port = port
    last_err: Optional[Exception] = None
    for candidate_port in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate_port), StatVisionHTTPHandler)
            bound_port = candidate_port
            break
        except OSError as exc:
            last_err = exc
    if server is None:
        raise RuntimeError(f"Nao foi possivel abrir o servidor local nas portas {port}-{port + 19}: {last_err}")

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
    url = f"http://127.0.0.1:{bound_port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    print("=====================================================================")
    print(f" StatMath rodando em: {url}")
    print(" Mantenha esta janela aberta enquanto usa o aplicativo.")
    print(" Pressione Ctrl+C para encerrar.")
    print("=====================================================================")
    server.serve_forever()




# ================================================================================================
# SEÇÃO 9. PONTO DE ENTRADA UNIFICADO E PORTÁTIL (AUTO-ABERTURA DE NAVEGADOR + CLI)
# ================================================================================================

def main_unified() -> int:
    # Garante saida UTF-8 segura no console do Windows (evita crash UnicodeEncodeError com emojis)
    for _stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="StatMath (StatVision OCR) — Aplicativo Aberto e Portátil para Distribuições Estatísticas"
    )
    parser.add_argument("-i", "--image", type=str, help="Analisar uma imagem específica via terminal (CLI).")
    parser.add_argument("--hint", type=str, default="", help="Expressão escrita da distribuição para análise.")
    parser.add_argument("--json", action="store_true", help="Exportar relatório em formato JSON.")
    parser.add_argument("--evaluate", action="store_true", help="Avaliar acertos/erros da IA nas pastas training_data/.")
    parser.add_argument("--learn", action="store_true", help="Aprender novos padrões de escrita a partir das pastas training_data/.")
    parser.add_argument("--init-folders", action="store_true", help="Criar estrutura de pastas training_data/<distribuicao>/.")
    parser.add_argument("--port", type=int, default=8501, help="Porta para rodar o aplicativo Web (padrão: 8501).")
    parser.add_argument("--no-browser", action="store_true", help="Não abrir o navegador automaticamente ao iniciar.")

    args, _ = parser.parse_known_args()

    if args.init_folders:
        path = ensure_training_folders(populate_initial_samples=True)
        print(f"✅ Estrutura de pastas de treinamento criada em: {path}")
        return 0

    if args.evaluate or args.learn:
        summary = evaluate_and_optionally_learn(auto_learn=args.learn)
        print("=" * 88)
        print(" 📊 RELATÓRIO DE AVALIAÇÃO E TREINAMENTO DA IA (STATVISION OCR)")
        print("=" * 88)
        print(f" Total de Imagens : {summary['total_images']} | Acertos: {summary['correct_predictions']} | Acurácia: {summary['accuracy_percent']:.1f}%")
        for item in summary["details"]:
            st_lbl = "✅ ACERTOU" if item["is_correct"] else "❌ ERROU"
            print(f" {st_lbl:<10} | {item['file']:<34} | Esperado: {item['expected_id']:<12} | Previsto: {item['predicted_id']}")
        return 0

    if args.image or (args.hint and not sys.stdin.isatty() and args.no_browser):
        analyzer = StatisticalImageAnalyzer()
        report = analyzer.analyze(args.image, extra_text_hint=args.hint)
        if args.json:
            print(json.dumps(dataclasses.asdict(report), indent=2, ensure_ascii=False))
        else:
            print(analyzer.format_markdown_report(report))
        return 0

    run_standalone_server(port=args.port, open_browser=(not args.no_browser))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_unified())
