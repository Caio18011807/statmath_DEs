#!/bin/bash
# Clicar duas vezes neste arquivo no macOS (Finder) inicia o StatMath automaticamente!
# Usa caminhos 100% relativos para funcionar em QUALQUER computador onde a pasta for copiada.

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

PORT=8501

# Encontra Python 3 real no macOS (Homebrew, python.org ou sistema)
PY_BIN=""
for cand in /opt/homebrew/bin/python3 /usr/local/bin/python3 /Library/Frameworks/Python.framework/Versions/Current/bin/python3 /usr/bin/python3 "$(command -v python3 2>/dev/null)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    PY_BIN="$cand"
    break
  fi
done

if [ -z "$PY_BIN" ]; then
  echo "============================================================================"
  echo " [AVISO] Python 3 nao encontrado neste Mac."
  echo " Instale em: https://www.python.org/downloads/"
  echo "============================================================================"
  read -r -p "Pressione Enter para fechar..."
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
RC=$?
if [ $RC -ne 0 ]; then
  echo ""
  echo "[ERRO] O StatMath encerrou com codigo $RC."
  read -r -p "Pressione Enter para fechar..."
fi
