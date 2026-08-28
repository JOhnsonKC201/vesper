@echo off
REM Vesper launcher. Creates the venv on first run, then starts listening.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating virtual environment...
  python -m venv .venv || goto :fail
  .venv\Scripts\python.exe -m pip install --quiet --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-tts.txt || goto :fail
  echo.
  echo Downloading the voice model, about 60MB, one time only...
  .venv\Scripts\python.exe -m piper.download_voices en_GB-alan-medium --data-dir var\voices
)

.venv\Scripts\python.exe -m vesper.main %*
goto :eof

:fail
echo.
echo Setup failed. Run "run.bat --check" once the problem is fixed.
exit /b 1
