"""
Suíte de Testes Automatizados para o Tutor Educacional StatVision OCR.
"""

import os
import unittest
from stat_vision.analyzer import StatisticalImageAnalyzer
from stat_vision.knowledge_base import DISTRIBUTIONS_DB
from stat_vision.sample_generator import generate_all_demo_samples
from stat_vision.symbolic_engine import compute_numeric_summary
from train_and_evaluate import evaluate_and_optionally_learn


class TestStatVisionOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
        cls.sample_paths = generate_all_demo_samples(cls.sample_dir)
        cls.analyzer = StatisticalImageAnalyzer()

    def test_knowledge_base_completeness(self) -> None:
        self.assertGreaterEqual(len(DISTRIBUTIONS_DB), 10)
        for dist_id, spec in DISTRIBUTIONS_DB.items():
            self.assertTrue(spec.canonical_single_rv_latex, f"{dist_id} sem escrita canônica")
            self.assertTrue(spec.canonical_iid_rv_latex, f"{dist_id} sem escrita i.i.d.")
            self.assertTrue(spec.didactic_overview, f"{dist_id} sem explicação didática")
            self.assertTrue(spec.random_variable_meaning, f"{dist_id} sem explicação da V.A.")
            self.assertTrue(spec.pdf_or_pmf_latex, f"{dist_id} sem FDP/FMP")
            self.assertTrue(spec.cdf_latex, f"{dist_id} sem FDA")
            self.assertTrue(spec.mean_latex, f"{dist_id} sem média")
            self.assertTrue(spec.variance_latex, f"{dist_id} sem variância")
            self.assertTrue(spec.excess_kurtosis_latex, f"{dist_id} sem curtose")
            self.assertTrue(spec.raw_moment_k_latex, f"{dist_id} sem fórmula de momentos")
            self.assertTrue(spec.moments_didactic_guide, f"{dist_id} sem guia didático de momentos")
            self.assertGreaterEqual(len(spec.parameterizations), 2, f"{dist_id} precisa de múltiplas parametrizações")
            for p_spec in spec.parameterizations:
                self.assertTrue(p_spec.pdf_or_pmf_latex, f"{dist_id} param {p_spec.name} sem FDP")
                self.assertTrue(p_spec.cdf_latex, f"{dist_id} param {p_spec.name} sem FDA")
                self.assertTrue(p_spec.parameter_interpretations, f"{dist_id} param {p_spec.name} sem interpretação dos parâmetros")

    def test_handwritten_samples_ocr_recognition(self) -> None:
        for expected_dist_id, img_path in self.sample_paths.items():
            report = self.analyzer.analyze(img_path)
            self.assertEqual(
                report.identified_distribution.id,
                expected_dist_id,
                f"Esperado {expected_dist_id}, obteve {report.identified_distribution.id} para {img_path}",
            )
            self.assertGreater(report.confidence, 0.70)
            self.assertTrue(report.ocr_result.statistical_context.canonical_how_to_write_latex)

    def test_iid_and_random_variable_extraction(self) -> None:
        normal_img = self.sample_paths["normal"]
        report = self.analyzer.analyze(normal_img)
        ctx = report.ocr_result.statistical_context
        self.assertTrue(ctx.is_iid)
        self.assertTrue(ctx.is_independent)
        self.assertTrue(ctx.identically_distributed)
        self.assertIn("X_1", ctx.primary_random_variable)
        self.assertEqual(ctx.extracted_parameters.get("mu"), 0.0)

    def test_numeric_evaluator(self) -> None:
        norm_num = compute_numeric_summary("normal", {"mu": 2.0, "sigma": 3.0})
        self.assertAlmostEqual(norm_num["mean"], 2.0)
        self.assertAlmostEqual(norm_num["variance"], 9.0)
        self.assertAlmostEqual(norm_num["median"], 2.0)
        self.assertAlmostEqual(norm_num["excess_kurtosis"], 0.0)
        self.assertAlmostEqual(norm_num["raw_kurtosis"], 3.0)

    def test_direct_text_input_without_image(self) -> None:
        """Verifica se a região onde o usuário escreve/digita a distribuição funciona sem nenhuma imagem."""
        cases = [
            ("X_1, ..., X_n ~ i.i.d. N(2, 9)", "normal", 2.0, 9.0, True),
            ("X ~ Exponencial(lambda=2)", "exponential", 0.5, 0.25, False),
            ("X ~ Poisson(lambda=4)", "poisson", 4.0, 4.0, False),
            ("X ~ Binomial(n=10, p=0.5)", "binomial", 5.0, 2.5, False),
            ("X ~ Uniforme(a=-2, b=4)", "uniform", 1.0, 3.0, False),
            ("X ~ Gama(alpha=3, beta=1.5)", "gamma", 2.0, 4.0 / 3.0, False),
        ]
        for written_expr, expected_id, exp_mean, exp_var, exp_iid in cases:
            report = self.analyzer.analyze(None, extra_text_hint=written_expr)
            self.assertEqual(report.identified_distribution.id, expected_id, f"Falhou para entrada escrita: {written_expr}")
            self.assertAlmostEqual(report.numeric_summary["mean"], exp_mean, places=3)
            self.assertAlmostEqual(report.numeric_summary["variance"], exp_var, places=3)
            self.assertEqual(report.ocr_result.statistical_context.is_iid, exp_iid)

    def test_user_training_data_images_accuracy(self) -> None:
        """Verifica se todas as imagens adicionadas pelo usuário em training_data/ têm 100% de acerto."""
        summary = evaluate_and_optionally_learn(auto_learn=False)
        self.assertGreaterEqual(summary["total_images"], 80)
        self.assertEqual(summary["wrong_predictions"], 0, f"Erros encontrados: {summary['details']}")
        self.assertEqual(summary["accuracy_percent"], 100.0)

    def test_training_images_with_simulated_noisy_apple_vision_ocr(self) -> None:
        """
        Simula o comportamento fora do sandbox no macOS onde o Apple Vision retorna texto com ruído
        (ex.: lendo a função indicadora I_{(0,1)}(y) como 'N(0,1)(y)' e 'Beta(1, 1)' como 'Bela(l, l)'),
        garantindo que nenhuma imagem do banco de treino perca sua distribuição ou seus parâmetros.
        """
        training_dir = self.analyzer.ocr_engine._resolve_training_dir()
        orig_extract = self.analyzer.ocr_engine.extract_text
        try:
            self.analyzer.ocr_engine.extract_text = lambda p: (
                "Y ~ Bela(l, l) => f(y) = r(2)/(r(l)r(l)) y^0(l-y)^0 N(0,1)(y)",
                "Apple Vision Simulado",
            )
            for rel_key, expected_tr in self.analyzer.ocr_engine.KNOWN_TRAINING_TRANSCRIPTIONS.items():
                expected_dist = rel_key.split("/", 1)[0]
                fpath = os.path.join(training_dir, rel_key)
                if not os.path.exists(fpath):
                    continue
                rep = self.analyzer.analyze(fpath)
                self.assertEqual(
                    rep.identified_distribution.id,
                    expected_dist,
                    f"Falha sob OCR ruidoso em {rel_key}: esperado {expected_dist}, obteve {rep.identified_distribution.id}",
                )
                ctx = rep.ocr_result.statistical_context
                self.assertTrue(ctx.image_pdf_latex, f"FDP parametrizada vazia para {rel_key}")
                self.assertTrue(ctx.image_cdf_latex, f"FDA parametrizada vazia para {rel_key}")
                self.assertTrue(
                    ctx.extracted_parameters or ctx.symbolic_parameters_detected,
                    f"Nenhum parâmetro extraído para {rel_key} ({expected_tr})",
                )
        finally:
            self.analyzer.ocr_engine.extract_text = orig_extract

    def test_brand_new_unseen_handwritten_formulas_and_noisy_ocr(self) -> None:
        """
        Verifica se imagens/fórmulas inéditas (que NÃO estão no banco de treino), mesmo com
        erros típicos de OCR manuscrito ou contendo apenas a equação da FDP/FMP, têm a
        distribuição e todos os parâmetros reconhecidos corretamente.
        """
        unseen_cases = [
            # 1. Beta inédita com ruído típico do Apple Vision ('Bela(l, l)' e indicadora lida como 'N(0,1)(y)')
            (
                "Y ~ Bela(l, l) => f(y) = r(2)/(r(l)r(l)) y^0(l-y)^0 N(0,1)(y)",
                "beta",
                {"alpha": 1.0, "beta": 1.0},
            ),
            # 2. Beta inédita com decimais lidos como 'O.S' e apenas fórmula
            (
                "P ~ Beta(O.S, O.S) => f(p) = r(1)/r(l/2)^2 p^{-1/2}(1-p)^{-1/2} N(0,1)(p)",
                "beta",
                {"alpha": 0.5, "beta": 0.5},
            ),
            # 3. Beta inédita onde apenas a equação da FDP foi capturada na foto
            (
                "f(x) = (r(9)/(r(4)r(5))) x^3 (1-x)^4 I_{(0,1)}(x)",
                "beta",
                {"alpha": 4.0, "beta": 5.0},
            ),
            # 4. Binomial inédita apenas pela fórmula da FMP
            (
                "p(k) = C(12, k) (0.4)^k (0.6)^{12-k}",
                "binomial",
                {"n": 12.0, "p": 0.4},
            ),
            # 5. Qui-Quadrado inédita com 'X2 lO' e fórmula
            (
                "W ~ X2 lO => f(w) = ((1/2)^5 / r(5)) w^4 e^{-w/2}",
                "chi_square",
                {"k": 10.0},
            ),
            # 6. Exponencial inédita apenas pela fórmula da FDP
            (
                "f(t) = 3.5 e^{-3.5 t} I_{(0,\\infty)}(t)",
                "exponential",
                {"lam": 3.5},
            ),
            # 7. Gama inédita apenas pela fórmula da FDP
            (
                "f(y) = (4^6 / r(6)) y^5 e^{-4y}",
                "gamma",
                {"alpha": 6.0, "beta": 4.0},
            ),
            # 8. Geométrica inédita apenas pela fórmula da FMP
            (
                "p(z) = 0.35 (1 - 0.35)^{z-1}",
                "geometric",
                {"p": 0.35},
            ),
            # 9. Normal inédita apenas pela fórmula da FDP
            (
                "f(x) = (1 / \\sqrt{32\\pi}) \\exp(-(x - 12)^2 / 32)",
                "normal",
                {"mu": 12.0, "sigma": 4.0},
            ),
            # 10. Poisson inédita apenas pela fórmula da FMP
            (
                "p(w) = e^{-4.2} (4.2)^w / w!",
                "poisson",
                {"lam": 4.2},
            ),
            # 11. t-Student inédita apenas pela fórmula da FDP
            (
                "f(t) = (r(4) / (\\sqrt{7\\pi} r(7/2))) (1 + t^2/7)^{-4}",
                "student_t",
                {"nu": 7.0},
            ),
            # 12. Uniforme inédita apenas pela fórmula da FDP
            (
                "f(u) = (1 / (12 - 4)) I_{[4, 12]}(u)",
                "uniform",
                {"a": 4.0, "b": 12.0},
            ),
            # 13. Poisson manuscrita com vírgula decimal (4,5), sequência E_1, ..., E_i e intervalo [0, n] (OCR bruto Apple Vision)
            (
                "Ex, E.,..., E, id Poinon (4,5), i € l0, m]",
                "poisson",
                {"lam": 4.5},
            ),
            # 14. Cauchy com notação posicional e OCR ruidoso
            (
                "X ~ Couchy(-1, 0.5)",
                "cauchy",
                {"x0": -1.0, "gamma": 0.5},
            ),
            # 15. Cauchy inédita apenas pela fórmula da FDP padrão
            (
                "f(x) = 1 / (\\pi (1 + x^2))",
                "cauchy",
                {"x0": 0.0, "gamma": 1.0},
            ),
            # 16. Cauchy inédita apenas pela fórmula da FDP geral
            (
                "f(y) = 3 / (\\pi ((y - 2)^2 + 9))",
                "cauchy",
                {"x0": 2.0, "gamma": 3.0},
            ),
            # 17. Qui-Quadrado manuscrita quebrada em linhas pelo OCR ("R1, R2\n+ 2\n10")
            (
                "R1, R2\n+ 2\n10",
                "chi_square",
                {"k": 10.0},
            ),
        ]
        for expr, expected_dist, expected_params in unseen_cases:
            rep = self.analyzer.analyze(None, extra_text_hint=expr)
            self.assertEqual(
                rep.identified_distribution.id,
                expected_dist,
                f"Distribuição incorreta para fórmula inédita '{expr}': esperado {expected_dist}, obteve {rep.identified_distribution.id}",
            )
            extracted = rep.ocr_result.statistical_context.extracted_parameters
            for k, v in expected_params.items():
                self.assertIn(k, extracted, f"Parâmetro '{k}' não extraído em '{expr}' (extraídos: {extracted})")
                self.assertAlmostEqual(
                    extracted[k],
                    v,
                    places=3,
                    msg=f"Valor incorreto para '{k}' em '{expr}': esperado {v}, obteve {extracted[k]}",
                )
            self.assertTrue(rep.ocr_result.statistical_context.image_pdf_latex)
            self.assertTrue(rep.ocr_result.statistical_context.image_cdf_latex)

        # Verifica especificamente a detecção da sequência E_1, ..., E_n, i.i.d. e as correções do intervalo [0, n]
        rep_pois_seq = self.analyzer.analyze(None, extra_text_hint="Ex, E.,..., E, id Poinon (4,5), i € l0, m]")
        ctx_seq = rep_pois_seq.ocr_result.statistical_context
        self.assertTrue(ctx_seq.is_iid)
        self.assertEqual(ctx_seq.primary_random_variable, r"E_1, E_2, \dots, E_n")
        self.assertGreaterEqual(len(ctx_seq.notation_corrections), 3)

        # Verifica especificamente a lista finita R_1, R_2 ~ Qui-Quadrado(10)
        rep_r12 = self.analyzer.analyze(None, extra_text_hint="R1, R2\n+ 2\n10")
        self.assertEqual(rep_r12.identified_distribution.id, "chi_square")
        self.assertEqual(rep_r12.ocr_result.statistical_context.primary_random_variable, "R_1, R_2")

        # Verifica o detector de qualidade: avisa em foto desfocada/escura e NÃO avisa em foto nítida
        training_dir = self.analyzer.ocr_engine._resolve_training_dir()
        blur_img = os.path.join(training_dir, "chi_square", "Chi_square_dist10.jpg")
        good_img = os.path.join(training_dir, "beta", "Beta_dist5.png")
        if os.path.exists(blur_img) and os.path.exists(good_img):
            rep_blur = self.analyzer.analyze(blur_img)
            self.assertTrue(rep_blur.curve_analysis.image_properties.is_low_quality)
            self.assertTrue(rep_blur.curve_analysis.image_properties.quality_warnings)
            rep_good = self.analyzer.analyze(good_img)
            self.assertFalse(rep_good.curve_analysis.image_properties.is_low_quality)
            self.assertEqual(rep_good.curve_analysis.image_properties.quality_warnings, [])

            # Verifica o pipeline de restauração/melhora de imagem para fotos desfocadas
            from stat_vision.curve_analyzer import enhance_image_for_ocr
            enh_path = enhance_image_for_ocr(blur_img)
            self.assertIsNotNone(enh_path)
            try:
                enh_props = self.analyzer.curve_analyzer.analyze_image(enh_path).image_properties
                self.assertGreater(enh_props.sharpness_score, rep_blur.curve_analysis.image_properties.sharpness_score * 5.0)
                self.assertGreater(enh_props.contrast_range, rep_blur.curve_analysis.image_properties.contrast_range)
                self.assertFalse(enh_props.is_low_quality)
            finally:
                if enh_path and os.path.exists(enh_path):
                    os.remove(enh_path)

    def test_interactive_distribution_chart_axes_and_zoom(self) -> None:
        """
        Verifica se o gráfico de cada distribuição possui os valores graduados do Eixo X e Eixo Y,
        controles de Zoom/Movimentação e leitor interativo dos valores de cada ponto.
        """
        from app import _render_svg_curve

        for dist_id, spec in DISTRIBUTIONS_DB.items():
            num = compute_numeric_summary(dist_id, spec.default_params)
            html_chart = _render_svg_curve(
                dist_id,
                num.get("parameters_used", {}),
                rv_symbol="x",
                mean_val=num.get("mean"),
            )
            self.assertIn("distInteractiveSvg", html_chart, f"SVG interativo ausente para {dist_id}")
            self.assertIn("Eixo X", html_chart, f"Título do Eixo X ausente para {dist_id}")
            self.assertIn("Eixo Y", html_chart, f"Título do Eixo Y ausente para {dist_id}")
            self.assertIn("distChartZoomSlider", html_chart, f"Controle de Zoom ausente para {dist_id}")
            self.assertIn("distAxesTicksLayer", html_chart, f"Camada de valores dos eixos ausente para {dist_id}")
            self.assertIn("chartHoverX", html_chart, f"Leitor de valor X ausente para {dist_id}")
            self.assertIn("chartHoverY", html_chart, f"Leitor de valor Y ausente para {dist_id}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
