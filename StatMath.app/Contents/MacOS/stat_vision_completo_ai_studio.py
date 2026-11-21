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
import html
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
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
            r"N\s*\(\s*[^,]+,\s*[^)]+\)",
            r"\mathcal\{N\}",
            r"Normal\s*\(",
            r"Gauss",
            r"Z\s*~\s*N\s*\(\s*0\s*,\s*1\s*\)",
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
            r"U\s*\(\s*[^,]+,\s*[^)]+\)",
            r"Unif\s*\(",
            r"Uniforme?\s*\(",
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
        aliases=["student t", "t-distribution", "t de student", "student", "cauchy"],
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
        mean = math.exp(mu + 0.5 * s2)
        var = (math.exp(s2) - 1.0) * math.exp(2.0 * mu + s2)
        skew = (math.exp(s2) + 2.0) * math.sqrt(math.exp(s2) - 1.0)
        ex_kurt = math.exp(4 * s2) + 2 * math.exp(3 * s2) + 3 * math.exp(2 * s2) - 6.0
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": math.exp(mu),
            "mode": math.exp(mu - s2),
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": ex_kurt + 3.0,
            "raw_moments": {
                "E[X^1]": math.exp(mu + 0.5 * s2),
                "E[X^2]": math.exp(2 * mu + 2.0 * s2),
                "E[X^3]": math.exp(3 * mu + 4.5 * s2),
                "E[X^4]": math.exp(4 * mu + 8.0 * s2),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": skew * (var ** 1.5),
                "mu_4": (ex_kurt + 3.0) * (var ** 2),
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
        xs = [x_min + (x_max - x_min) * i / (num_points - 1) for i in range(num_points)]
        ys = [(1.0 / L) if (a <= x <= b) else 0.0 for x in xs]
        return xs, ys, False

    elif dist_id == "gamma":
        alpha = max(p.get("alpha", 3.0), 0.5)
        beta = max(p.get("beta", 1.0), 1e-6)
        x_max = (alpha + 4.0 * math.sqrt(alpha)) / beta
        xs = [max(1e-4, x_max * i / (num_points - 1)) for i in range(num_points)]
        norm_const = (beta ** alpha) / math.gamma(alpha)
        ys = [norm_const * (x ** (alpha - 1.0)) * math.exp(-beta * x) for x in xs]
        return xs, ys, False

    elif dist_id == "beta":
        alpha = max(p.get("alpha", 2.0), 0.5)
        beta = max(p.get("beta", 5.0), 0.5)
        xs = [0.001 + 0.998 * i / (num_points - 1) for i in range(num_points)]
        b_func = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
        ys = [((x ** (alpha - 1.0)) * ((1.0 - x) ** (beta - 1.0))) / b_func for x in xs]
        return xs, ys, False

    elif dist_id == "chi_square":
        k = max(p.get("k", 4.0), 1.0)
        x_max = k + 4.5 * math.sqrt(2.0 * k)
        xs = [max(0.01, x_max * i / (num_points - 1)) for i in range(num_points)]
        norm_const = 1.0 / ((2.0 ** (0.5 * k)) * math.gamma(0.5 * k))
        ys = [norm_const * (x ** (0.5 * k - 1.0)) * math.exp(-0.5 * x) for x in xs]
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
        x_max = math.exp(mu + 2.8 * sigma)
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
        lam = max(p.get("lam", 4.0), 0.5)
        k_max = max(12, int(lam + 4 * math.sqrt(lam)))
        xs = [float(k) for k in range(k_max + 1)]
        ys = [((lam ** k) * math.exp(-lam)) / math.factorial(int(k)) for k in xs]
        return xs, ys, True

    elif dist_id == "binomial":
        n = max(int(round(p.get("n", 10.0))), 1)
        prob = min(max(p.get("p", 0.5), 1e-6), 1.0 - 1e-6)
        xs = [float(k) for k in range(n + 1)]
        ys = [math.comb(n, int(k)) * (prob ** k) * ((1.0 - prob) ** (n - k)) for k in xs]
        return xs, ys, True

    elif dist_id == "geometric":
        prob = min(max(p.get("p", 0.35), 1e-6), 1.0 - 1e-6)
        k_max = min(25, max(10, int(5.0 / prob)))
        xs = [float(k) for k in range(1, k_max + 1)]
        ys = [((1.0 - prob) ** (k - 1)) * prob for k in xs]
        return xs, ys, True

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

import json
import os
import re
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StatisticalContextDetails:
    is_iid: bool = False
    iid_evidence: List[str] = field(default_factory=list)
    is_independent: bool = False
    independence_evidence: List[str] = field(default_factory=list)
    identically_distributed: bool = False
    primary_random_variable: str = "X"
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
guard let nsImage = NSImage(contentsOfFile: imagePath),
      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    exit(2)
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
    """

    def extract_text(self, image_path: str) -> Tuple[str, str]:
        if not os.path.exists(image_path):
            return "", "arquivo_nao_encontrado"

        sidecar_text = self._read_png_text_chunks(image_path)

        # 1. Tenta EasyOCR se instalado
        try:
            import easyocr  # type: ignore
            reader = easyocr.Reader(["pt", "en"], gpu=False, verbose=False)
            results = reader.readtext(image_path, detail=0)
            if results:
                combined = "\n".join(results)
                if sidecar_text and sidecar_text not in combined:
                    combined = combined + "\n" + sidecar_text
                return combined, "EasyOCR (Rede Neural Local para Manuscritos)"
        except Exception:
            pass

        # 2. Tenta Tesseract CLI se instalado
        tesseract_bin = shutil.which("tesseract")
        if tesseract_bin:
            try:
                proc = subprocess.run(
                    [tesseract_bin, image_path, "stdout", "--psm", "6"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    combined = proc.stdout.strip()
                    if sidecar_text and sidecar_text not in combined:
                        combined = combined + "\n" + sidecar_text
                    return combined, "Tesseract OCR (Motor Local CLI)"
            except Exception:
                pass

        # 3. Tenta Apple Vision Framework nativo do macOS (excelente para escrita à mão!)
        if os.uname().sysname == "Darwin" and shutil.which("swift"):
            try:
                text = self._run_macos_vision_ocr(image_path)
                if text:
                    if sidecar_text and sidecar_text not in text:
                        text = text + "\n" + sidecar_text
                    return text, "Apple Vision Framework (OCR Manuscrito Nativo macOS)"
            except Exception:
                pass

        if sidecar_text:
            return sidecar_text, "Camada de Metadados + OCR Geométrico"

        return "", "Sem camada de texto detectada"

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
    def _load_learned_patterns() -> Dict[str, List[str]]:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        learned_file = os.path.join(project_root, "training_data", "learned_patterns.json")
        if not os.path.exists(learned_file):
            return {}
        try:
            with open(learned_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("learned_ocr_aliases", {})
        except Exception:
            return {}

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
        Corrige confusões comuns de OCR quando o usuário escreve notação matemática à mão:
        - '1.1.d' ou 'l.l.d' -> 'i.i.d.'
        - '∼' ou '≈' -> '~'
        - 'N[' ou 'N{' -> 'N('
        """
        t = raw_text
        t = t.replace("∼", "~").replace("≈", "~").replace("˜", "~")
        t = re.sub(r"\b[1lI]\.[1lI]\.d\.?\b", "i.i.d.", t, flags=re.IGNORECASE)
        t = re.sub(r"\biid\b", "i.i.d.", t, flags=re.IGNORECASE)
        return t

    def analyze_image_text(self, image_path: Optional[str] = None, extra_text_hint: str = "") -> OCRAnalysisResult:
        if image_path and os.path.exists(image_path):
            raw_text, backend = self.extract_text(image_path)
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

        normalized = cleaned_text.lower()
        learned_patterns = self._load_learned_patterns()

        for dist_id, spec in DISTRIBUTIONS_DB.items():
            # 0. Padrões aprendidos do treinamento com imagens reais do usuário
            user_cues = learned_patterns.get(dist_id, [])
            for u_cue in user_cues:
                if u_cue and u_cue.lower() in normalized:
                    scores[dist_id] += 60.0
                    matched_cues[dist_id].append(f"Padrão treinado pelo usuário: '{u_cue}'")

            # 1. Nome e apelidos da distribuição
            for alias in spec.aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, normalized):
                    scores[dist_id] += 45.0
                    matched_cues[dist_id].append(f"Nome/Apelido reconhecido: '{alias}'")
                if lines and re.search(pattern, lines[0].lower()):
                    scores[dist_id] += 40.0
                    matched_cues[dist_id].append(f"Destaque principal na escrita: '{alias}'")

            # 2. Padrões de notação matemática (ex.: N(μ, σ²), Pois(λ), Bin(n, p))
            for pat in spec.notation_patterns:
                try:
                    m = re.search(pat, cleaned_text, flags=re.IGNORECASE)
                    if m:
                        scores[dist_id] += 45.0
                        matched_cues[dist_id].append(f"Notação manuscrita reconhecida: '{m.group(0)}'")
                except re.error:
                    pass

            # 3. Indícios de fórmula matemática (FDP/FMP escrita à mão)
            for cue in spec.formula_cues:
                if cue.lower() in normalized:
                    scores[dist_id] += 35.0
                    matched_cues[dist_id].append(f"Fragmento de fórmula reconhecido: '{cue}'")

        # Heurísticas adicionais para escrita manuscrita
        if re.search(r"sqrt\s*\(\s*2\s*[\*⋅]?\s*(pi|π)", normalized) or "2π" in normalized or "2\\pi" in normalized:
            if "ln" in normalized or "log" in normalized:
                scores["lognormal"] += 35.0
                matched_cues["lognormal"].append("Fórmula manuscrita: 'ln(x)' com constante '2π'")
            else:
                scores["normal"] += 35.0
                matched_cues["normal"].append("Fórmula manuscrita: constante Gaussiana '2π'")

        if re.search(r"e\s*\^\s*\{\s*-\s*\\?lambda", normalized) or "k!" in normalized:
            if "lambda^k" in normalized or "λ^k" in normalized or "k!" in normalized:
                scores["poisson"] += 35.0
                matched_cues["poisson"].append("Fórmula manuscrita: 'λ^k e^{-λ} / k!'")

        if "12" in normalized and ("b-a" in normalized or "b - a" in normalized):
            scores["uniform"] += 35.0
            matched_cues["uniform"].append("Fórmula manuscrita: '(b - a)' e denominador '12'")

        # Reconhecimento de atalhos rápidos digitados pelo usuário (ex.: N(0,1), U(0,1), Exp(2), Pois(3), Bin(10,0.5), t(5), chi2(4))
        if re.search(r"\b(?:n|norm|normal)\s*\(", normalized):
            scores["normal"] += 65.0
            matched_cues["normal"].append("Notação digitada: Normal N(...)")
        if re.search(r"\b(?:exp|exponencial|exponential)\s*\(", normalized):
            scores["exponential"] += 65.0
            matched_cues["exponential"].append("Notação digitada: Exponencial Exp(...)")
        if re.search(r"\b(?:u|unif|uniforme|uniform)\s*\(", normalized):
            scores["uniform"] += 65.0
            matched_cues["uniform"].append("Notação digitada: Uniforme U(...)")
        if re.search(r"\b(?:gama|gamma|erlang)\s*\(", normalized):
            scores["gamma"] += 65.0
            matched_cues["gamma"].append("Notação digitada: Gama(...)")
        if re.search(r"\bbeta\s*\(", normalized):
            scores["beta"] += 65.0
            matched_cues["beta"].append("Notação digitada: Beta(...)")
        if re.search(r"\b(?:chi2|chisq|qui-?quadrado|chi-?square)|\\chi\^2", normalized):
            scores["chi_square"] += 65.0
            matched_cues["chi_square"].append("Notação digitada: Qui-Quadrado")
        if re.search(r"\b(?:student|t_\d+|t\s*\(\s*\d+)", normalized):
            scores["student_t"] += 65.0
            matched_cues["student_t"].append("Notação digitada: t de Student")
        if re.search(r"\b(?:lognormal|log-?normal|lnorm)\s*\(", normalized):
            scores["lognormal"] += 65.0
            matched_cues["lognormal"].append("Notação digitada: Log-Normal")
        if re.search(r"\bweibull\s*\(", normalized):
            scores["weibull"] += 65.0
            matched_cues["weibull"].append("Notação digitada: Weibull")
        if re.search(r"\b(?:pois|poisson)\s*\(", normalized):
            scores["poisson"] += 65.0
            matched_cues["poisson"].append("Notação digitada: Poisson")
        if re.search(r"\b(?:bin|binom|binomial|bern|bernoulli)\s*\(", normalized):
            scores["binomial"] += 65.0
            matched_cues["binomial"].append("Notação digitada: Binomial / Bernoulli")
        if re.search(r"\b(?:geom|geometrica|geometric)\s*\(", normalized):
            scores["geometric"] += 65.0
            matched_cues["geometric"].append("Notação digitada: Geométrica")

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

        # 3. Identificar Variáveis Aleatórias e Amostras Indexadas (X_1, ..., X_n)
        rv_matches = re.findall(r"\b([XYZUVWST]|X_\d+|X_i|X_n|Y_\d+|Y_i|Y_n)\b", text)
        for rv in rv_matches:
            if rv not in ctx.random_variables_detected:
                ctx.random_variables_detected.append(rv)

        if any(tok in text for tok in ["X_1", "X_2", "X_n", "X_i", "Y_1", "Y_n"]) or re.search(r"[XYZ]_1\s*,?\s*\.\.\.", text):
            ctx.is_indexed_sample = True
            ctx.identically_distributed = True
            base_letter = "Y" if "Y_1" in text else "X"
            ctx.primary_random_variable = f"{base_letter}_1, {base_letter}_2, \\dots, {base_letter}_n"
            if ctx.is_independent and not ctx.is_iid:
                ctx.is_iid = True
                ctx.iid_evidence.append("Sequência amostral indexada combinada com independência")
        elif ctx.random_variables_detected:
            ctx.primary_random_variable = ctx.random_variables_detected[0]
        else:
            ctx.primary_random_variable = "X"

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

        # Argumentos posicionais em N(0, 1), U(a, b), Bin(n, p), Pois(λ), Exp(λ), Gama(α, β), Beta(α, β)
        m_norm = re.search(r"(?:N|Normal)\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_norm:
            ctx.extracted_parameters.setdefault("mu", float(m_norm.group(1)))
            s2 = float(m_norm.group(2))
            ctx.extracted_parameters.setdefault("sigma", s2 ** 0.5 if s2 > 0 else 1.0)
            ctx.parameterization_hint = f"Notação posicional N(μ={m_norm.group(1)}, σ²={m_norm.group(2)})"

        m_unif = re.search(r"(?:U|Unif|Uniforme?)\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_unif:
            ctx.extracted_parameters.setdefault("a", float(m_unif.group(1)))
            ctx.extracted_parameters.setdefault("b", float(m_unif.group(2)))
            ctx.parameterization_hint = f"Intervalo Uniforme U(a={m_unif.group(1)}, b={m_unif.group(2)})"

        m_bin = re.search(r"(?:Bin|Binom|Binomial)\s*\(\s*(\d+)\s*,\s*(0?\.\d+)\s*\)", text, re.IGNORECASE)
        if m_bin:
            ctx.extracted_parameters.setdefault("n", float(m_bin.group(1)))
            ctx.extracted_parameters.setdefault("p", float(m_bin.group(2)))
            ctx.parameterization_hint = f"Binomial(n={m_bin.group(1)}, p={m_bin.group(2)})"

        m_pois = re.search(r"(?:Pois|Poisson)\s*\(\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_pois:
            ctx.extracted_parameters.setdefault("lam", float(m_pois.group(1)))

        m_exp = re.search(r"(?:Exp|Exponencial|Exponential)\s*\(\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_exp:
            ctx.extracted_parameters.setdefault("lam", float(m_exp.group(1)))

        m_beta = re.search(r"Beta\s*\(\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_beta:
            ctx.extracted_parameters.setdefault("alpha", float(m_beta.group(1)))
            ctx.extracted_parameters.setdefault("beta", float(m_beta.group(2)))

        m_gamma = re.search(r"(?:Gamma|Gama|\\Gamma)\s*\(\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_gamma:
            ctx.extracted_parameters.setdefault("alpha", float(m_gamma.group(1)))
            ctx.extracted_parameters.setdefault("beta", float(m_gamma.group(2)))

        m_chisq = re.search(r"(?:chi2|chisq|qui-?quadrado|chi-?square|\\chi\^2)_?\{\s*(\d*\.?\d+)\s*\}|(?:chi2|chisq|qui-?quadrado|chi-?square)\s*\(\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_chisq:
            val_k = m_chisq.group(1) or m_chisq.group(2)
            if val_k:
                ctx.extracted_parameters.setdefault("k", float(val_k))

        m_stud = re.search(r"(?:student|t)\s*\(\s*(\d*\.?\d+)\s*\)|t_\{?\s*(\d*\.?\d+)\s*\}?", text, re.IGNORECASE)
        if m_stud:
            val_nu = m_stud.group(1) or m_stud.group(2)
            if val_nu:
                ctx.extracted_parameters.setdefault("nu", float(val_nu))

        m_weib = re.search(r"weibull\s*\(\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_weib:
            ctx.extracted_parameters.setdefault("lam", float(m_weib.group(1)))
            ctx.extracted_parameters.setdefault("k", float(m_weib.group(2)))

        m_lnorm = re.search(r"(?:lognormal|log-?normal|lnorm)\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\)", text, re.IGNORECASE)
        if m_lnorm:
            ctx.extracted_parameters.setdefault("mu", float(m_lnorm.group(1)))
            ctx.extracted_parameters.setdefault("sigma", float(m_lnorm.group(2)))

        m_geom = re.search(r"(?:geom|geometrica|geometric)\s*\(\s*(0?\.\d+)\s*\)", text, re.IGNORECASE)
        if m_geom:
            ctx.extracted_parameters.setdefault("p", float(m_geom.group(1)))

        if "n" in ctx.extracted_parameters:
            ctx.sample_size_n = int(ctx.extracted_parameters["n"])

        # Construir a recomendação educacional "Como deveria ser escrito formalmente"
        ctx.canonical_how_to_write_latex, ctx.handwriting_feedback = self._build_educational_writing_guide(
            text, top_spec, ctx
        )

        return ctx

    @staticmethod
    def _build_educational_writing_guide(
        text: str, spec: DistributionSpec, ctx: StatisticalContextDetails
    ) -> Tuple[str, List[str]]:
        feedback: List[str] = []
        p = ctx.extracted_parameters

        # Formatar a versão canônica específica com os números extraídos (se houver) ou simbólica
        lhs = r"X_1, X_2, \dots, X_n" if (ctx.is_iid or ctx.is_indexed_sample) else ctx.primary_random_variable
        rel = r"\stackrel{\text{i.i.d.}}{\sim}" if ctx.is_iid else (r"\stackrel{\text{ind.}}{\sim}" if (ctx.is_independent and ctx.is_indexed_sample) else r"\sim")

        if spec.id == "normal":
            if "mu" in p and "sigma" in p:
                s2 = round(p["sigma"] ** 2, 4)
                rhs = rf"\mathcal{{N}}(\mu={p['mu']:g}, \, \sigma^2={s2:g})"
            else:
                rhs = r"\mathcal{N}(\mu, \sigma^2)"
        elif spec.id == "exponential":
            rhs = rf"\operatorname{{Exp}}(\lambda={p['lam']:g})" if "lam" in p else r"\operatorname{Exp}(\lambda)"
        elif spec.id == "uniform":
            rhs = rf"\mathcal{{U}}(a={p['a']:g}, \, b={p['b']:g})" if ("a" in p and "b" in p) else r"\mathcal{U}(a, b)"
        elif spec.id == "poisson":
            rhs = rf"\operatorname{{Poisson}}(\lambda={p['lam']:g})" if "lam" in p else r"\operatorname{Poisson}(\lambda)"
        elif spec.id == "binomial":
            rhs = rf"\operatorname{{Binomial}}(n={int(p['n'])}, \, p={p['p']:g})" if ("n" in p and "p" in p) else r"\operatorname{Binomial}(n, p)"
        elif spec.id == "gamma":
            rhs = rf"\operatorname{{Gama}}(\alpha={p['alpha']:g}, \, \beta={p['beta']:g})" if ("alpha" in p and "beta" in p) else r"\operatorname{Gama}(\alpha, \beta)"
        elif spec.id == "beta":
            rhs = rf"\operatorname{{Beta}}(\alpha={p['alpha']:g}, \, \beta={p['beta']:g})" if ("alpha" in p and "beta" in p) else r"\operatorname{Beta}(\alpha, \beta)"
        elif spec.id == "chi_square":
            rhs = rf"\chi^2_{{k={int(p['k'])}}}" if "k" in p else r"\chi^2_k"
        elif spec.id == "student_t":
            rhs = rf"t_{{\nu={p['nu']:g}}}" if "nu" in p else r"t_\nu"
        elif spec.id == "lognormal":
            rhs = r"\operatorname{LogNormal}(\mu, \sigma^2)"
        elif spec.id == "weibull":
            rhs = r"\operatorname{Weibull}(\lambda, k)"
        elif spec.id == "geometric":
            rhs = rf"\operatorname{{Geom}}(p={p['p']:g})" if "p" in p else r"\operatorname{Geom}(p)"
        else:
            rhs = spec.canonical_single_rv_latex

        canonical_latex = f"{lhs} {rel} {rhs}"

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
import struct
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

        raise ValueError(f"Unsupported image format without Pillow/OpenCV: {path}")

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
                gray_row = [float(row[i]) for i in range(0, stride, 2)]
            elif bpp in (3, 4):
                for i in range(0, stride, bpp):
                    r, g, b_val = row[i], row[i + 1], row[i + 2]
                    gray_row.append(0.299 * r + 0.587 * g + 0.114 * b_val)
            grid.append(gray_row)

        return width, height, "PNG", grid


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
        else:
            subtitle_line = f"X ~ {spec.name} ({use_params})"
            extra_line = extra_line or "Amostra aleatoria i.i.d."

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Usa o gerador Swift AppKit (folha de caderno manuscrita) no macOS
    if os.uname().sysname == "Darwin" and shutil.which("swift"):
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

    def analyze(self, image_path: Optional[str] = None, extra_text_hint: str = "") -> DistributionReport:
        ocr_res = self.ocr_engine.analyze_image_text(image_path, extra_text_hint=extra_text_hint)
        if image_path and os.path.exists(image_path):
            curve_res = self.curve_analyzer.analyze_image(image_path)
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

        combined_scores: Dict[str, float] = {}
        for dist_id in DISTRIBUTIONS_DB:
            ocr_score = ocr_res.distribution_scores.get(dist_id, 0.0)
            shape_score = curve_res.shape_scores.get(dist_id, 0.0)
            # Para imagens de escrita à mão, o OCR domina a pontuação
            combined_scores[dist_id] = round(ocr_score * 1.5 + shape_score * 0.4, 2)

        ranked_ids = sorted(combined_scores.keys(), key=lambda d: combined_scores[d], reverse=True)
        top_id = ranked_ids[0]
        top_spec = DISTRIBUTIONS_DB[top_id]

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
            f"- **Distribuição Identificada**: **{spec.name}** (`{spec.id}`)",
            f"- **Família**: Distribuição **{spec.family}**",
            f"- **Suporte (Valores Possíveis)**: ${spec.support_latex}$ ({spec.support})",
            f"- **Grau de Confiança da IA**: `{report.confidence * 100:.1f}%` (Motor OCR: `{report.ocr_result.backend_used}`)",
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



PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training_data")
LEARNED_JSON_PATH = os.path.join(TRAINING_DIR, "learned_patterns.json")


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
    e (se `auto_learn=True`) extrai fragmentos de OCR das imagens que a IA errou
    para salvá-los em `learned_patterns.json`, fazendo a IA acertar nas próximas vezes!
    """
    ensure_training_folders(populate_initial_samples=True)
    analyzer = StatisticalImageAnalyzer()

    results: List[Dict[str, Any]] = []
    total = 0
    correct = 0
    newly_learned: List[Tuple[str, str]] = []

    try:
        with open(LEARNED_JSON_PATH, "r", encoding="utf-8") as f:
            learned_data = json.load(f)
    except Exception:
        learned_data = {"learned_ocr_aliases": {}, "training_history": []}

    learned_aliases: Dict[str, List[str]] = learned_data.setdefault("learned_ocr_aliases", {})

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
            elif auto_learn:
                # Extrair linha significativa do OCR da escrita à mão para ensinar à IA
                for line in report.ocr_result.lines:
                    clean_line = line.strip()
                    if len(clean_line) >= 4:
                        dist_list = learned_aliases.setdefault(dist_id, [])
                        if clean_line not in dist_list:
                            dist_list.append(clean_line)
                            newly_learned.append((dist_id, clean_line))
                        break

            results.append({
                "file": f"{dist_id}/{fname}",
                "expected_id": dist_id,
                "expected_name": spec.name,
                "predicted_id": pred_id,
                "predicted_name": report.identified_distribution.name,
                "confidence": report.confidence,
                "is_correct": is_hit,
                "ocr_preview": " | ".join(report.ocr_result.lines[:2])[:90],
            })

    if auto_learn and newly_learned:
        learned_data["training_history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "new_patterns_learned": newly_learned,
        })
        with open(LEARNED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(learned_data, f, indent=2, ensure_ascii=False)

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
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional



def _running_inside_streamlit() -> bool:
    try:
        import streamlit as st  # type: ignore
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _render_svg_curve(dist_id: str, params: Dict[str, float]) -> str:
    xs, ys, is_discrete = evaluate_curve_points(dist_id, params, num_points=140)
    if not xs or not ys:
        return ""
    w, h = 540, 190
    pad_x, pad_y = 36, 22
    pw, ph = w - 2 * pad_x, h - 2 * pad_y
    x_min, x_max = min(xs), max(xs)
    y_max = max(max(ys), 1e-6)
    x_span = max(x_max - x_min, 1e-6)

    svg_parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="190" style="background:#090d16;border-radius:12px;border:1px solid #1e293b;">',
        f'<line x1="{pad_x}" y1="{h - pad_y}" x2="{w - pad_x}" y2="{h - pad_y}" stroke="#475569" stroke-width="1.5"/>',
        f'<line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{h - pad_y}" stroke="#475569" stroke-width="1.5"/>',
    ]
    if is_discrete:
        for x_val, y_val in zip(xs, ys):
            cx = pad_x + ((x_val - x_min) / x_span) * (pw - 20) + 10
            cy = (h - pad_y) - (y_val / y_max) * (ph - 10)
            svg_parts.append(
                f'<line x1="{cx:.1f}" y1="{h - pad_y}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="#ef4444" stroke-width="5.5" stroke-linecap="round"/>'
            )
            svg_parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#ffffff"/>')
    else:
        pts = []
        for x_val, y_val in zip(xs, ys):
            cx = pad_x + ((x_val - x_min) / x_span) * pw
            cy = (h - pad_y) - (y_val / y_max) * (ph - 10)
            pts.append(f"{cx:.1f},{cy:.1f}")
        poly_pts = f"{pad_x},{h - pad_y} " + " ".join(pts) + f" {w - pad_x},{h - pad_y}"
        svg_parts.append(f'<polygon points="{poly_pts}" fill="rgba(239, 68, 68, 0.18)"/>')
        svg_parts.append(f'<polyline fill="none" stroke="#ef4444" stroke-width="3" points="{" ".join(pts)}"/>')

    svg_parts.append(
        f'<text x="{pad_x + 8}" y="{pad_y + 12}" fill="#94a3b8" font-size="11" font-family="monospace">'
        f'Gráfico da Distribuição (x em [{x_min:.2f}, {x_max:.2f}])</text>'
    )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


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
        if os.uname().sysname == "Darwin" and shutil.which("swift"):
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
        return ""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
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

        qs = urllib.parse.parse_qs(parsed.query)
        written_query = qs.get("hint", [""])[0].strip()

        if written_query:
            report = self.analyzer.analyze(None, extra_text_hint=written_query)
            self._send_photomath_html(report=report, hint=written_query, prev_data_url="")
        else:
            # Inicia o aplicativo limpo, sem nenhum exemplo pré-carregado
            self._send_photomath_html(report=None, hint="", prev_data_url="")

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8", errors="ignore")
        params = urllib.parse.parse_qs(body)

        hint = params.get("hint", [""])[0].strip()
        data_url = params.get("image_data_url", [""])[0].strip()
        action = params.get("action", ["analyze"])[0]

        if action == "clear":
            self._send_photomath_html(report=None, hint="", prev_data_url="")
            return

        if action == "paste_system_clipboard":
            clip_b64 = self._read_macos_system_clipboard_image_b64()
            if clip_b64:
                data_url = f"data:image/png;base64,{clip_b64}"

        report: Optional[DistributionReport] = None

        if data_url.startswith("data:image/") and "," in data_url:
            b64_part = data_url.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_part)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(raw_bytes)
            tmp.close()
            try:
                report = self.analyzer.analyze(tmp.name, extra_text_hint=hint)
            finally:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
        elif hint:
            # Quando o usuário escreve/digita a distribuição diretamente (sem enviar imagem)
            report = self.analyzer.analyze(None, extra_text_hint=hint)

        self._send_photomath_html(report=report, hint=hint, prev_data_url=data_url)

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
        svg_overlay = _render_svg_curve(spec.id, num.get("parameters_used", {}))

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

        return f"""
        <!-- CARD DE RESULTADO IMEDIATO (SEMPRE VISÍVEL NO COMEÇO) -->
        <div class="pm-result-hero">
          <div class="pm-scanned-box">
            <div class="pm-scanned-label">✍️ Como Deveria Ser Escrito Formalmente (em LaTeX)</div>
            <div style="font-size:1.22rem;overflow-x:auto;">
              \\[{ctx.canonical_how_to_write_latex}\\]
            </div>
          </div>

          <span class="pm-dist-badge">DISTRIBUIÇÃO IDENTIFICADA</span>
          <h2 style="margin:0 0 6px 0;font-size:1.65rem;color:#fff;">{html.escape(spec.name)}</h2>
          <p style="margin:0 0 14px 0;color:#cbd5e1;font-size:0.93rem;">
            <strong>Família:</strong> Distribuição {spec.family} &nbsp;•&nbsp;
            <strong>Suporte:</strong> \\({spec.support_latex}\\) &nbsp;•&nbsp;
            <strong>Variável Aleatória:</strong> \\({ctx.primary_random_variable}\\) &nbsp;•&nbsp;
            <strong>Amostragem:</strong> {"i.i.d. (Independente e Identicamente Distribuída)" if ctx.is_iid else ("Independente" if ctx.is_independent else "Variável Aleatória")}
          </p>

          <div style="font-weight:700;color:#fda4af;font-size:0.92rem;margin-bottom:4px;">
            📐 {html.escape(spec.pdf_or_pmf_name)} Principal:
          </div>
          <div class="math-block">\\[{spec.pdf_or_pmf_latex}\\]</div>

          <div style="font-weight:700;color:#38bdf8;font-size:0.92rem;margin-top:10px;margin-bottom:4px;">
            📈 Função de Distribuição Acumulada — FDA / CDF (\\(F_X(x) = \\mathbb{{P}}(X \\le x)\\)):
          </div>
          <div class="math-block">\\[{spec.cdf_latex}\\]</div>

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
    ) -> None:
        has_user_image = prev_data_url.startswith("data:image/") and "," in prev_data_url
        img_src = prev_data_url if has_user_image else ""
        img_display = "block" if has_user_image else "none"
        placeholder_display = "none" if has_user_image else "flex"

        results_column_html = self._build_results_column_html(report)

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
    height: 270px;
    background: #04070d;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}
  .viewfinder img, .viewfinder video {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
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

<!-- NAVBAR SUPERIOR LIMPA -->
<nav class="app-navbar">
  <div class="brand">
    <div class="brand-icon">∑</div>
    <div>
      <span>StatMath</span>
      <span style="font-size:0.78rem;color:#fb7185;display:block;font-weight:600;">Leitor e Tutor de Distribuições Estatísticas</span>
    </div>
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
        <img id="previewImage" src="{img_src}" alt="Imagem Enviada" style="display:{img_display};"/>
        <video id="cameraVideo" autoplay playsinline style="display:none;"></video>

        <div class="crop-reticle" id="reticleBox">
          <div class="corner tl"></div>
          <div class="corner tr"></div>
          <div class="corner bl"></div>
          <div class="corner br"></div>
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

        <form method="POST" id="analyzeForm">
          <input type="hidden" name="action" id="formAction" value="analyze"/>
          <input type="hidden" name="image_data_url" id="imageDataUrl" value="{html.escape(prev_data_url)}"/>
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
  document.getElementById('imageDataUrl').value = dataUrl;
  document.getElementById('formAction').value = 'analyze';
  document.getElementById('analyzeForm').submit();
}}

function handleSelectedImageFile(file) {{
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {{
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
  document.getElementById('imageDataUrl').value = '';
  document.getElementById('hintInput').value = '';
  document.getElementById('formAction').value = 'clear';
  document.getElementById('analyzeForm').submit();
}}

function toggleAllSteps(expand) {{
  document.querySelectorAll('details.section-accordion').forEach(el => {{
    el.open = expand;
  }});
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
    server = HTTPServer(("127.0.0.1", port), StatVisionHTTPHandler)
    print(f"=====================================================================")
    print(f" StatMath rodando em: http://127.0.0.1:{port}")
    print(f" Pressione Ctrl+C para encerrar.")
    print(f"=====================================================================")
    server.serve_forever()




# ================================================================================================
# SEÇÃO 9. PONTO DE ENTRADA UNIFICADO E PORTÁTIL (AUTO-ABERTURA DE NAVEGADOR + CLI)
# ================================================================================================

def main_unified() -> int:
    import webbrowser
    import threading

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

    if not args.no_browser:
        url = f"http://127.0.0.1:{args.port}"
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()

    run_standalone_server(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_unified())
