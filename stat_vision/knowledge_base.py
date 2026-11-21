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
