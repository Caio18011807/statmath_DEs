#!/usr/bin/env python3
"""
Script de linha de comando para rodar a Avaliação Cega na pasta real_test_data/
e atualizar a acurácia real exibida no topo do aplicativo StatMath.
"""

from __future__ import annotations

import os
import sys

# Garante que a raiz do projeto esteja no sys.path em qualquer SO
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stat_vision.real_evaluator import evaluate_real_test_folder, resolve_real_test_dir


def main() -> int:
    test_dir = resolve_real_test_dir()
    print("=" * 80)
    print(" 🧪 AVALIAÇÃO CEGA DE TESTE REAL — STATMATH")
    print("=" * 80)
    print(f" Pasta de teste: {test_dir}")

    result = evaluate_real_test_folder(force=True)
    items = result.get("items", [])

    if not items:
        print("\n ⚠️  Nenhuma imagem encontrada na pasta real_test_data/.")
        print("     Coloque suas 10 imagens lá (ex.: normal_1.jpg, poisson_2.png, cauchy.jpg)")
        print("     e execute novamente este script ou clique em 'Avaliar' no aplicativo.\n")
        return 0

    print(f"\n Avaliando {len(items)} imagem(ns) em modo 100% cego (1.0 = Dist+Params | 0.5 = Apenas 1 | 0.0 = Errou):\n")
    for it in items:
        sc = it.get("sample_score")
        if sc == 1.0:
            mark = "✅ 1.0 pt (ACERTOU)"
        elif sc == 0.5:
            mark = "⚠️  0.5 pt (PARCIAL)"
        elif sc == 0.0:
            mark = "❌ 0.0 pt (ERROU)  "
        else:
            mark = "⚠️  SEM GABARITO   "
        d_ok = "✅" if it.get("dist_ok") else "❌"
        p_ok = "✅" if it.get("params_ok") else "❌"
        print(
            f"  [{it['index']:02d}] {mark} | Arquivo: {it['filename']}\n"
            f"       Gabarito : {it['expected_name']} ({it['expected_key']})\n"
            f"       IA previu: {it['predicted_name']} ({it['predicted_key']}) | Dist: {d_ok} | Params: {p_ok} | VA: {it['random_variable'] or '-'}\n"
            f"       Parâmetr.: {it.get('extracted_parameters') or '(não extraído)'} | Tempo: {it['elapsed_ms']} ms\n"
            f"       OCR lido : {it['ocr_text'] or '(vazio)'}\n"
        )

    evaluated = result.get("evaluated", 0)
    total_pts = result.get("total_points", 0.0)
    full_hits = result.get("correct", 0)
    partial_hits = result.get("partial", 0)
    misses = result.get("wrong", 0)
    acc = result.get("accuracy_pct")
    print("=" * 80)
    if acc is not None:
        print(f" 🎯 ACURÁCIA REAL DA IA: {acc:.1f}% ({total_pts:g}/{evaluated} pontos)")
        print(f"    • Acertos totais (1.0 pt — Dist + Params): {full_hits}")
        print(f"    • Acertos parciais (0.5 pt — Apenas 1)   : {partial_hits}")
        print(f"    • Erros totais (0.0 pt — Nenhum)         : {misses}")
    else:
        print(" ⚠️  Nenhuma imagem com nome de distribuição reconhecível foi avaliada.")
    print(f" 💾 Resultado salvo em: {os.path.join(test_dir, 'evaluation_results.json')}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
