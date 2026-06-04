@echo off
echo Starting ALPR System...
call .\.venv\Scripts\activate.bat
python main.py --gui
pause
