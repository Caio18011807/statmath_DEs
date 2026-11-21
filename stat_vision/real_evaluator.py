#!/usr/bin/env python3
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

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from stat_vision.analyzer import StatisticalImageAnalyzer
from stat_vision.knowledge_base import DISTRIBUTIONS_DB
from stat_vision.ocr_engine import OfflineOCREngine

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
