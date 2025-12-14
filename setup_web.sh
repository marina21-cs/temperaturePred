#!/bin/bash
# =============================================================================
# STEP-BY-STEP GUIDE: Deploy Temperature Prediction Web App with Ngrok
# =============================================================================

echo "=============================================="
echo "🌡️  Philippine Temperature Prediction Web App"
echo "=============================================="
echo ""

# Step 1: Install dependencies
echo "📦 Step 1: Installing Python dependencies..."
pip install flask -q
echo "✓ Flask installed"

# Step 2: Install ngrok
echo ""
echo "🔧 Step 2: Installing ngrok..."
if ! command -v ngrok &> /dev/null; then
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt update -qq && sudo apt install ngrok -qq
    echo "✓ ngrok installed"
else
    echo "✓ ngrok already installed"
fi

echo ""
echo "=============================================="
echo "📋 MANUAL STEPS TO COMPLETE:"
echo "=============================================="
echo ""
echo "STEP 3: Configure ngrok (one-time setup)"
echo "   - Get your free auth token from: https://dashboard.ngrok.com/get-started/your-authtoken"
echo "   - Run: ngrok config add-authtoken YOUR_TOKEN_HERE"
echo ""
echo "STEP 4: Start the Flask server"
echo "   - In Terminal 1, run: python app.py"
echo "   - Server will start at http://localhost:5000"
echo ""
echo "STEP 5: Start ngrok tunnel"
echo "   - In Terminal 2, run: ngrok http 5000"
echo "   - Copy the public URL (e.g., https://xxxx-xx-xx.ngrok-free.app)"
echo ""
echo "STEP 6: Access your web app"
echo "   - Open the ngrok URL in any browser"
echo "   - Share the URL with anyone to view the predictions!"
echo ""
echo "=============================================="
