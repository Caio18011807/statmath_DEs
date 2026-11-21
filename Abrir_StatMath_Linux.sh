#!/usr/bin/env bash
# Inicializador portátil do StatMath para Linux (Ubuntu, Fedora, Arch, Debian, etc.)
# Encontra automaticamente sua própria pasta e a pasta training_data/ em qualquer computador.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

PORT=8501

if command -v fuser >/dev/null 2>&1; then
  fuser -k ${PORT}/tcp >/dev/null 2>&1 || true
fi

if command -v python3 >/dev/null 2>&1; then
  python3 "$DIR/stat_vision_completo_ai_studio.py" --port $PORT
else
  echo "Python 3 não encontrado. Instale com: sudo apt install python3"
  exit 1
fi
