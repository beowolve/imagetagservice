@echo off
REM ============================================================
REM  ImageTagService – Nuitka standalone Windows build
REM
REM  Prerequisites:
REM    pip install nuitka ordered-set zstandard
REM    (all Python dependencies must be installed in the venv)
REM
REM  Output:
REM    dist\ImageTagService.exe   (standalone single executable)
REM    dist\models\               (copy model files here after build)
REM ============================================================

setlocal

set OUTPUT_DIR=dist
set EXE_NAME=ImageTagService

REM Create output dir
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

python -m nuitka ^
    --standalone ^
    --onefile ^
    --output-dir="%OUTPUT_DIR%" ^
    --output-filename="%EXE_NAME%.exe" ^
    --include-package=ram ^
    --include-package=timm ^
    --include-package=transformers ^
    --include-package=fastapi ^
    --include-package=uvicorn ^
    --include-package=PIL ^
    --include-package=torch ^
    --include-package=torchvision ^
    --include-data-files=config.py=config.py ^
    --windows-console-mode=force ^
    --assume-yes-for-downloads ^
    app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo [OK] Build complete: %OUTPUT_DIR%\%EXE_NAME%.exe
echo.
echo Next steps:
echo   1. Run: %OUTPUT_DIR%\%EXE_NAME%.exe path\to\image.jpg
echo      The model (~5.6 GB) is downloaded automatically on first run.
echo   2. For server mode: %OUTPUT_DIR%\%EXE_NAME%.exe

endlocal
