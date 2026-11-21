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
from stat_vision.knowledge_base import DISTRIBUTIONS_DB


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
