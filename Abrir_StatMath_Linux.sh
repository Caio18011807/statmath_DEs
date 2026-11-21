#!/usr/bin/env bash
# Inicializador portatil do StatMath para Linux (Ubuntu, Fedora, Arch, Debian, etc.)
# Encontra automaticamente sua propria pasta e as pastas training_data/ e real_test_data/.

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

PORT=8501

PY_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PY_BIN" ]; then
  echo "============================================================================"
  echo " [AVISO] Python 3 nao encontrado neste sistema Linux."
  echo " Instale com: sudo apt install python3   (ou equivalente da sua distro)"
  echo "============================================================================"
  exit 1
fi

ENTRY="$DIR/stat_vision_completo_ai_studio.py"
if [ ! -f "$ENTRY" ]; then
  ENTRY="$DIR/app.py"
fi

echo "Iniciando StatMath em http://127.0.0.1:$PORT ..."
if [ "$ENTRY" = "$DIR/app.py" ]; then
  "$PY_BIN" "$ENTRY" "$PORT"
else
  "$PY_BIN" "$ENTRY" --port "$PORT"
fi
