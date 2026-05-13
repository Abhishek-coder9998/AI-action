#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "============================================================"
echo "  ⚽ FOOTBALL AI ANALYTICS - STARTUP SYSTEM (Unix)"
echo "============================================================"
echo ""

# Check for virtual environment
if [ -d "venv" ]; then
    echo "[INFO] Activating Virtual Environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[INFO] Activating Virtual Environment..."
    source .venv/bin/activate
else
    echo "[WARN] Virtual environment not found. Running with global python."
fi

# Get Local IP Address (cross-platform for Mac/Linux)
IP=$(hostname -I | awk '{print $1}')
if [ -z "$IP" ]; then
    IP=$(ip route get 1.2.3.4 | awk '{print $7}')
fi

echo ""
echo "------------------------------------------------------------"
echo "📱 MOBILE ACCESS INSTRUCTIONS:"
echo "------------------------------------------------------------"
echo " 1. Connect your phone to the SAME Wi-Fi as this PC."
echo " 2. Open your phone's browser."
echo " 3. Visit: http://$IP:8501"
echo "------------------------------------------------------------"
echo ""

echo "[INFO] Starting Streamlit Server..."
streamlit run app.py --server.address 0.0.0.0
