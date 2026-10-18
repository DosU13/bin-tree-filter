@echo off
rem Keep perform.py running at an installation: restart it if it exits with an error.
rem Usage:  tools\perform_loop.bat scenes\show.json [perform.py options]
rem Quitting on purpose (Ctrl+Shift+Q, exit code 0) ends the loop.
cd /d "%~dp0.."
:loop
".venv\Scripts\python.exe" perform.py %*
if %ERRORLEVEL% EQU 0 goto end
echo perform.py exited with code %ERRORLEVEL%, restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
:end
