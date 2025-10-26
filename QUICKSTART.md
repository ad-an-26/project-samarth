# 🚀 Quick Start Guide - Project Samarth

## Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] API key from [data.gov.in](https://data.gov.in)
- [ ] LLM API credentials (Azure GPT-4, OpenAI, etc.)

## 5-Minute Setup

### Option A: Automated Setup (macOS/Linux)

```bash
cd "Project Samarth"
./setup.sh
```

### Option B: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets (choose one method)
```

## Configure Your API Keys

### Method 1: Using .env file (Recommended)

Create `.env` in project root:

```env
DATA_GOV_API_KEY=your_data_gov_key
LITELLM_API_KEY=your_llm_key
LLM_MODEL=azure/gpt-4
```

### Method 2: Using Streamlit secrets

Edit `.streamlit/secrets.toml`:

```toml
DATA_GOV_API_KEY = "your_data_gov_key"
LITELLM_API_KEY = "your_llm_key"
LLM_MODEL = "azure/gpt-4"
```

## Update Configuration

1. Open `config.py`
2. Replace the placeholder resource IDs:

```python
RESOURCES = {
    "agriculture": {
        "id": "9ef84268-d588-465a-a308-a864a43d0070",  # ← Update this
        ...
    },
    "climate": {
        "id": "your-actual-climate-resource-id",  # ← Update this
        ...
    }
}
```

**Where to find resource IDs:**
- Go to [data.gov.in](https://data.gov.in)
- Search for agriculture/climate datasets
- Copy the resource ID from the dataset page URL

## Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## First Steps in the App

1. **Click "Load Data & Start"** - This will:
   - Fetch data from data.gov.in
   - Cache it locally
   - Initialize the analytics agent

2. **Ask a Question** - Try these examples:
   - "What's the trend in rice production?"
   - "Show correlation between rainfall and yield"
   - "Which states produce the most wheat?"

3. **Explore Results** - You'll see:
   - 📊 Automatic visualizations
   - 📝 AI-generated insights
   - 💻 The Python code that ran
   - 📋 Raw data tables

## Troubleshooting

### "No module named 'X'"
```bash
pip install -r requirements.txt
```

### "Missing API key"
- Check your `.env` or `secrets.toml` file
- Ensure keys are spelled correctly
- No quotes needed in `.env`, but needed in `.toml`

### "Could not fetch data"
- Verify your DATA_GOV_API_KEY is valid
- Check internet connection
- Verify resource IDs in `config.py` are correct

### "LLM Error"
- Verify LITELLM_API_KEY is correct
- Check if you have API credits/quota
- Ensure LLM_MODEL matches your provider

## Understanding the Architecture

### Flow 1: Data Loading (Automatic)
- Happens once per day
- Only fetches if data changed
- Caches everything locally

### Flow 2: Query Processing (Interactive)
1. You ask a question
2. AI writes Python code
3. Code runs on your data
4. Results visualized + explained

## Next Steps

- Read `README.md` for detailed documentation
- Check `config.py` for customization options
- Explore the code in `agent_setup.py` and `presentation.py`
- Try complex queries combining multiple data sources

## Need Help?

- Check the example queries in the sidebar
- View raw output and generated code (expand sections)
- All data processing is local - your data never leaves your machine

---

**You're ready!** Start asking questions and get data-driven insights. 🌾

