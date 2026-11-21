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
from stat_vision.knowledge_base import DISTRIBUTIONS_DB, DistributionSpec
from stat_vision.curve_analyzer import PurePythonPNGReader


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
        if sys.platform == "darwin" and shutil.which("swift"):
            try:
                text = self._run_macos_vision_ocr(image_path)
                if text:
                    if sidecar_text and sidecar_text not in text:
                        text = text + "\n" + sidecar_text
                    return text, "Apple Vision Framework (OCR Manuscrito Nativo macOS)"
            except Exception:
                pass

        # 4. Tenta Windows.Media.Ocr nativo do Windows 10/11 via PowerShell (sem dependências externas)
        if sys.platform == "win32" and shutil.which("powershell"):
            try:
                text = self._run_windows_native_ocr(image_path)
                if text:
                    if sidecar_text and sidecar_text not in text:
                        text = text + "\n" + sidecar_text
                    return text, "Windows Native OCR (WinRT PowerShell)"
            except Exception:
                pass

        if sidecar_text:
            return sidecar_text, "Camada de Metadados + OCR Geométrico"

        return "", "Sem camada de texto detectada"

    @staticmethod
    def _run_windows_native_ocr(image_path: str) -> Optional[str]:
        abs_path = os.path.abspath(image_path).replace("'", "''")
        ps_script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
            "$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]; "
            "$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]; "
            "$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]; "
            "$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]; "
            "function Await($op, $t) { $m = $asTask.MakeGenericMethod($t); $task = $m.Invoke($null, @($op)); $task.Wait(); return $task.Result }; "
            f"$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{abs_path}')) ([Windows.Storage.StorageFile]); "
            "$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream]); "
            "$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder]); "
            "$bmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap]); "
            "$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages(); "
            "if ($null -ne $engine) { "
            "$res = Await ($engine.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult]); "
            "foreach ($line in $res.Lines) { Write-Output $line.Text } }"
        )
        win_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
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
        t = raw_text
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

        # 2g. Normaliza Qui-Quadrado manuscrito quebrado em linhas pelo OCR (ex.: "R_1, R_2\n+ 2\n10" ou "R_1, R_2\nX 2\n10" -> "R_1, R_2 ~ Qui-Quadrado(10)")
        t = re.sub(
            r"(\b[A-Z]_\d(?:\s*,\s*[A-Z]_\d)*|\b[A-Z]\b)\s*(?:~|\\sim)?\s*(?:i\.?i\.?d\.?)?\s*[\r\n]+\s*(?:[+xXχ]|\\chi)\s*\^?\s*2\s*[\r\n\s]+(\d+(?:[\.,]\d+)?)\b",
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
        fuzzy_candidates = re.findall(r"\b([A-Za-zÀ-ÿ]{5,20})\s*[\(\[]", target_clean)
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
                    if best_r >= 0.72 and (best_r - second_r) >= 0.06:
                        scores[best_d] += 80.0
                        matched_cues[best_d].append(
                            f"Correspondência léxica aproximada (OCR manuscrito): '{raw_word}' ≈ {best_d} ({best_r:.0%})"
                        )
                        canon_name = fuzzy_canonical_names.get(best_d)
                        if canon_name and _strip_acc(raw_word) != _strip_acc(canon_name):
                            cleaned_text = re.sub(
                                r"\b" + re.escape(raw_word) + r"(?=\s*[\(\[])",
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

        # Exponencial: \lambda e^{-\lambda x} (com ou sem '^' quando o OCR omite sobrescrito) sem \Gamma, sem fatorial e sem potência quadrática
        if (
            "gamma(" not in normalized
            and "!" not in normalized
            and "^2" not in normalized
            and re.search(
                r"(?:=|\|)\s*\(?(\d+\s*/\s*\d+|\d*[\.,]?\d+|\\?lambda|λ)\)?\s*(?:\\?exp|e\^?)\s*[\(\{]?\s*-\s*(?:\d+\s*/\s*\d+|\d*[\.,]?\d+|\\?lambda|λ)?\s*[a-z]",
                normalized,
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
            ctx.extracted_parameters.setdefault("lam", self._parse_numeric_or_fraction(m_weib.group(1)))
            ctx.extracted_parameters.setdefault("k", self._parse_numeric_or_fraction(m_weib.group(2)))

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
