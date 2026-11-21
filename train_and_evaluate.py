#!/usr/bin/env python3
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

from stat_vision.analyzer import StatisticalImageAnalyzer
from stat_vision.knowledge_base import DISTRIBUTIONS_DB
from stat_vision.ocr_engine import OfflineOCREngine
from stat_vision.sample_generator import generate_all_demo_samples


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


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
