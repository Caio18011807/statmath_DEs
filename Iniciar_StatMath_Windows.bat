@echo off
chcp 65001 >nul
REM Clicar duas vezes neste arquivo no Windows inicia o StatMath automaticamente sem precisar digitar comandos!
REM Encontra automaticamente a propria pasta em qualquer computador Windows.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 stat_vision_completo_ai_studio.py --port 8501
    goto :fim
)

where python >nul 2>nul
if %errorlevel%==0 (
    python stat_vision_completo_ai_studio.py --port 8501
    goto :fim
)

where python3 >nul 2>nul
if %errorlevel%==0 (
    python3 stat_vision_completo_ai_studio.py --port 8501
    goto :fim
)

echo.
echo ============================================================================
echo  Python 3 nao foi encontrado neste computador Windows.
echo  Instale em: https://www.python.org/downloads/
echo  (Marque a opcao "Add Python to PATH" durante a instalacao)
echo ============================================================================
echo.
pause
:fim
