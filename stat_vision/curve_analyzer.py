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
from stat_vision.knowledge_base import DISTRIBUTIONS_DB


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
        tmp_png = os.path.join(tempfile.gettempdir(), f"statmath_conv_{os.getpid()}.png")
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
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    timeout=12,
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
