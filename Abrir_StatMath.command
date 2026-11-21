#!/bin/bash
# Clicar duas vezes neste arquivo no macOS (Finder) inicia o StatMath automaticamente!
# Usa caminhos 100% relativos para funcionar em QUALQUER computador onde a pasta for copiada.

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

PORT=8501

# Encerra instância anterior na mesma porta se houver
lsof -ti tcp:$PORT | xargs kill -9 2>/dev/null

# Inicia o StatMath (que já abre o navegador automaticamente)
python3 "$DIR/stat_vision_completo_ai_studio.py" --port $PORT
