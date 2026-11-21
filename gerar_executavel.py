#!/usr/bin/env python3
"""
Script Construtor de Executável Multiplataforma (macOS / Windows / Linux) para o StatMath.

Mantém todo o código-fonte aberto (`stat_vision_completo_ai_studio.py` e pasta `stat_vision/`)
e permite gerar:
1. Um pacote portátil `.zip` pronto para copiar para outros computadores (com inicializadores de
   duplo-clique para Windows `.bat` e macOS `.command` / `StatMath.app`).
2. Um binário standalone via PyInstaller (se o PyInstaller estiver instalado no sistema).
"""

import os
import shutil
import subprocess
import sys
import zipfile


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_portable_open_source_zip() -> str:
    """
    Cria um arquivo `StatMath_Portatil_OpenSource.zip` contendo:
    - O código-fonte aberto completo (`stat_vision_completo_ai_studio.py` + pacote `stat_vision/`)
    - `StatMath.app` (Aplicativo de duplo-clique para macOS)
    - `Abrir_StatMath.command` (Atalho de duplo-clique para macOS/Linux)
    - `Iniciar_StatMath_Windows.bat` (Atalho de duplo-clique para Windows)
    - `README.md` com instruções de uso e edição do código aberto.
    """
    zip_path = os.path.join(PROJECT_DIR, "StatMath_Portatil_OpenSource.zip")
    include_items = [
        "stat_vision_completo_ai_studio.py",
        "app.py",
        "cli.py",
        "train_and_evaluate.py",
        "evaluate_real_test.py",
        "test_stat_vision.py",
        "requirements.txt",
        "README.md",
        "Abrir_StatMath.command",
        "Abrir_StatMath_Linux.sh",
        "Iniciar_StatMath_Windows.bat",
        "gerar_executavel.py",
        "stat_vision",
        "StatMath.app",
        "training_data",
        "real_test_data",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in include_items:
            full_path = os.path.join(PROJECT_DIR, item)
            if not os.path.exists(full_path):
                continue
            if os.path.isfile(full_path):
                zf.write(full_path, arcname=os.path.join("StatMath", item))
            else:
                for root, _, files in os.walk(full_path):
                    if "__pycache__" in root:
                        continue
                    for f in files:
                        if f.endswith(".pyc"):
                            continue
                        fp = os.path.join(root, f)
                        rel = os.path.relpath(fp, PROJECT_DIR)
                        zf.write(fp, arcname=os.path.join("StatMath", rel))

    return zip_path


def try_pyinstaller_build() -> bool:
    if not shutil.which("pyinstaller"):
        return False
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--name",
        "StatMath_Standalone",
        os.path.join(PROJECT_DIR, "stat_vision_completo_ai_studio.py"),
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_DIR)
    return proc.returncode == 0


def main() -> int:
    print("=" * 78)
    print(" 📦 EMPACOTADOR OPEN-SOURCE & EXECUTÁVEL MULTIPLATAFORMA (STATMATH)")
    print("=" * 78)
    zip_file = create_portable_open_source_zip()
    print(f"✅ Pacote Portátil Open-Source criado com sucesso em:\n   {zip_file}")
    print("\n   Este pacote funciona em outros computadores mantendo o código 100% aberto:")
    print("   - No macOS  : Basta dar dois cliques em 'StatMath.app' ou 'Abrir_StatMath.command'")
    print("   - No Windows: Basta dar dois cliques em 'Iniciar_StatMath_Windows.bat'")

    if try_pyinstaller_build():
        print("✅ Executável binário adicional gerado na pasta dist/ via PyInstaller!")
    else:
        print("\n💡 Dica opcional: Caso queira também compilar um .exe único em máquinas sem Python,")
        print("   basta rodar: pip install pyinstaller && python3 gerar_executavel.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
