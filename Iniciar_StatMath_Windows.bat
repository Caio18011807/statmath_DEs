@echo off
setlocal EnableExtensions
title StatMath - Leitor de Distribuicoes Estatisticas
chcp 65001 >nul 2>nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0"

echo ============================================================================
echo  STATMATH - Iniciando aplicativo local...
echo  Pasta: %CD%
echo ============================================================================
echo.

set "ENTRY_SCRIPT=stat_vision_completo_ai_studio.py"
if not exist "%ENTRY_SCRIPT%" set "ENTRY_SCRIPT=app.py"
if not exist "%ENTRY_SCRIPT%" (
    echo [ERRO] Arquivo principal do StatMath nao encontrado nesta pasta!
    echo Extraia todo o conteudo do ZIP antes de executar este arquivo.
    echo.
    pause
    exit /b 1
)

set "PY_CMD="

REM 1. Testa py -3 (Windows Python Launcher)
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 7) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=py -3"
    goto :check_ocr
)

REM 2. Testa python no PATH (ignora o atalho falso da Microsoft Store!)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 7) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :check_ocr
)

REM 3. Testa python3 no PATH
python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 7) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=python3"
    goto :check_ocr
)

REM 4. Procura instalacoes padrao do Python 3 fora do PATH
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*" "C:\Python3*" "%ProgramFiles%\Python3*") do (
    if exist "%%~D\python.exe" (
        "%%~D\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 7) else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "PY_CMD="%%~D\python.exe""
            goto :check_ocr
        )
    )
)

echo ============================================================================
echo  [AVISO] Python 3 nao foi encontrado neste computador Windows!
echo.
echo  Como resolver em 1 minuto:
echo   1. Acesse: https://www.python.org/downloads/
echo   2. Baixe e abra o instalador do Python 3.
echo   3. IMPORTANTE: marque a caixinha "Add python.exe to PATH" antes de instalar.
echo   4. Depois de instalar, clique duas vezes neste arquivo novamente!
echo ============================================================================
echo.
pause
exit /b 1

:check_ocr
echo Python 3 detectado (%PY_CMD%).
REM Garante o motor de OCR neural multiplataforma (RapidOCR ONNX, leve, apenas 1a vez)
%PY_CMD% -c "import rapidocr_onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo Preparando motor de OCR neural multiplataforma ^(apenas na 1a execucao^)...
    %PY_CMD% -m pip install --quiet --disable-pip-version-check rapidocr-onnxruntime >nul 2>nul
    if errorlevel 1 (
        echo [INFO] Seguindo com o OCR nativo do Windows ^(sem internet ou pip indisponivel^).
    ) else (
        echo [OK] Motor de OCR neural instalado com sucesso!
    )
)

:run_app
echo Abrindo StatMath no navegador...
echo (Mantenha esta janela aberta enquanto estiver usando o aplicativo.)
echo.
if /i "%ENTRY_SCRIPT%"=="app.py" (
    %PY_CMD% "%ENTRY_SCRIPT%" 8501
) else (
    %PY_CMD% "%ENTRY_SCRIPT%" --port 8501
)
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ============================================================================
    echo  [ERRO] O StatMath encerrou inesperadamente (codigo %EXIT_CODE%).
    echo  Tire um print desta tela se precisar de ajuda para diagnostico.
    echo ============================================================================
    echo.
    pause
)
endlocal
