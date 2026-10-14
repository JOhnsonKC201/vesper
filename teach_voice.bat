@echo off
REM Record your voice profile again. Double-click this, or the desktop
REM shortcut "Teach Vesper my voice". It shows four sentences to say, then
REM keeps the window open so you can read how well they agreed.
REM Quit Vesper first: every sentence starts with its name, so a running
REM Vesper would hear them and answer. Start it again afterwards.
setlocal
cd /d "%~dp0"
echo Quit Vesper first (tray icon, Quit). It would hear these sentences too.
pause
call run.bat --enroll
echo.
pause
