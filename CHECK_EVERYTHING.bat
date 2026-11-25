@echo off
title Smart Conveyor QC - automatic tests
cd /d "%~dp0"
where py >nul 2>nul && goto :usepy
where python >nul 2>nul && goto :usepython
echo.
echo Python is not installed on this PC.
echo Install Python 3.12 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" in the installer), then double-click this file again.
echo.
pause
exit /b 1
:usepy
py -3 tools\go.py test
goto :end
:usepython
python tools\go.py test
:end
echo.
echo (window stays open so you can read the messages - press a key to close)
pause >nul
