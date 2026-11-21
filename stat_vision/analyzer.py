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
from stat_vision.knowledge_base import DistributionSpec, DISTRIBUTIONS_DB
from stat_vision.ocr_engine import OfflineOCREngine, OCRAnalysisResult
from stat_vision.curve_analyzer import (
    CurveShapeAnalyzer,
    CurveShapeAnalysis,
    ImageVisualProperties,
    enhance_image_for_ocr,
)
from stat_vision.symbolic_engine import compute_numeric_summary


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
