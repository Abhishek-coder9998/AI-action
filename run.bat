@echo off
setlocal
cd /d "%~dp0"

:: Set window title
title Football AI Analytics

echo ============================================================
echo   ⚽ FOOTBALL AI ANALYTICS - STARTUP SYSTEM
echo ============================================================
echo.

:: Check for virtual environment
if exist venv\Scripts\activate (
    echo [INFO] Activating Virtual Environment...
    call venv\Scripts\activate
) else (
    echo [WARN] Virtual environment (venv) not found. 
    echo [INFO] Running with global python if available.
)

:: Get Local IP Address
echo [INFO] Checking Local Network Configuration...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr "IPv4 Address"') do (
    set IP=%%a
)
:: Remove leading space
set IP=%IP: =%

echo.
echo ------------------------------------------------------------
echo 📱 MOBILE ACCESS INSTRUCTIONS:
echo ------------------------------------------------------------
echo  1. Connect your phone to the SAME Wi-Fi as this PC.
echo  2. Open your phone's camera or browser.
echo  3. Visit: http://%IP%:8501
echo ------------------------------------------------------------
echo.

echo [INFO] Starting Streamlit Server...
streamlit run app.py --server.address 0.0.0.0

echo.
echo [ERROR] Streamlit has stopped.
pause
