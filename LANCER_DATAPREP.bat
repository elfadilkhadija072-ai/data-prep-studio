@echo off
title DataPrep Studio - Serveur Permanent
color 0A
chcp 65001 >nul

echo =====================================================================
echo           LANCEMENT PERMANENT DE DATAPREP STUDIO
echo =====================================================================
echo.
echo [*] Repertoire : %~dp0
cd /d "%~dp0"

echo [*] Activation de l'environnement virtuel Python...
if exist ".\venv\Scripts\activate.bat" (
    call ".\venv\Scripts\activate.bat"
) else (
    echo [!] venv introuvable, utilisation du python systeme...
)

echo [*] Lancement du serveur...
python run.py

echo.
echo [!] Le serveur s'est arrete.
pause
