@echo off
REM Vasper's hands. See the sibling "vasper" script for the POSIX version.
REM The cd is load bearing: the brain runs from the home directory, and
REM "python -m vesper.tool" from there cannot find the package.
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m vesper.tool %*
