#!/bin/bash
# Project Samarth - Quick Setup Script

echo "🌾 Project Samarth - Setup Script"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  No .env file found"
    echo "📝 Please create a .env file with your API keys:"
    echo ""
    echo "   DATA_GOV_API_KEY=your_key_here"
    echo "   LITELLM_API_KEY=your_key_here"
    echo "   LLM_MODEL=azure/gpt-4"
    echo ""
fi

# Check for secrets.toml
if [ ! -f .streamlit/secrets.toml ]; then
    echo "⚠️  No .streamlit/secrets.toml file found"
    echo "📝 Alternatively, configure secrets in .streamlit/secrets.toml"
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "  1. Configure your API keys in .env or .streamlit/secrets.toml"
echo "  2. Update resource IDs in config.py"
echo "  3. Run: streamlit run app.py"
echo ""
echo "Happy analyzing! 🚀"

