@echo off
title DataPrep ML Pro
echo Lancement de l'application Streamlit...
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
streamlit run app.py
pause
