# 🌾 Project Samarth - Sovereign-Ready Analytics Agent

An intelligent analytics platform for Indian agriculture and climate data, designed with data sovereignty and efficiency at its core.

## 🎯 Overview

Project Samarth combines smart data caching, natural language processing, and advanced analytics to provide policy-oriented insights from government data sources. The system is architected with two independent flows:

1. **Flow 1: Version-Aware Data Pipeline** - Intelligent data ingestion with minimal API calls
2. **Flow 2: Analytics Agent Runtime** - Natural language query processing with LLM-powered insights

## ✨ Features

- **Smart Caching**: Version-aware data pipeline that only fetches when data is stale
- **Natural Language Queries**: Ask questions in plain English
- **Advanced Analytics**: Built-in statistical analysis (correlation, regression, etc.)
- **Data Sovereignty**: All data processing happens locally
- **Policy-Oriented**: Insights tailored for policymakers and stakeholders
- **Flexible LLM Support**: Works with external APIs (OpenAI, Azure) or local models (Llama)

## 🏗️ Architecture

### Flow 1: The "Version-Aware" Data Pipeline

1. Read local cache log
2. Check if last API check was < 24 hours ago (Daily Gate)
3. If needed, perform lightweight freshness check
4. Compare remote vs local timestamps
5. **Fast Path**: Load from cache if data is current
6. **Slow Path**: Fetch, transform, and cache if data is stale

### Flow 2: The "Analytics Agent" Runtime

1. User submits natural language query
2. Agent injects context (data schemas + capabilities)
3. LLM generates Python code for analysis
4. Execute code securely against in-memory data
5. **Dual Presentation Path**:
   - **Path A**: Deterministic visualization (charts/graphs)
   - **Path B**: LLM-powered synthesis (insights & recommendations)

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- API key from [data.gov.in](https://data.gov.in)
- LLM API credentials (Azure OpenAI, OpenAI, or local LLM)

### Installation

1. **Clone or navigate to the project directory**

```bash
cd "Project Samarth"
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Create a `.env` file in the project root:

```env
# Data.gov.in API Key
DATA_GOV_API_KEY=your_data_gov_api_key_here

# LiteLLM Configuration
LITELLM_API_KEY=your_llm_api_key_here
LITELLM_API_BASE=your_api_base_url  # Optional
LLM_MODEL=azure/gpt-4  # Or your preferred model
```

Alternatively, create `.streamlit/secrets.toml`:

```toml
DATA_GOV_API_KEY = "your_data_gov_api_key_here"
LITELLM_API_KEY = "your_llm_api_key_here"
LITELLM_API_BASE = "your_api_base_url"  # Optional
LLM_MODEL = "azure/gpt-4"
```

5. **Update resource IDs in `config.py`**

Replace the placeholder resource IDs with actual IDs from data.gov.in:

```python
RESOURCES = {
    "agriculture": {
        "id": "your-agriculture-resource-id",  # Update this
        ...
    },
    "climate": {
        "id": "your-climate-resource-id",  # Update this
        ...
    }
}
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **First Run**: Click "Load Data & Start" to fetch and cache data
2. **Ask Questions**: Type natural language queries about agriculture and climate
3. **View Results**: Get visualizations, statistics, and policy insights
4. **Explore Data**: Use the expanders to view raw data and generated code

### Example Queries

- "What's the trend in rice production over the years?"
- "Show me the correlation between rainfall and crop yield"
- "Which states have the highest wheat production?"
- "Compare production across different seasons"
- "What's the impact of rainfall on agriculture in Punjab?"

## 🗂️ Project Structure

```
project_samarth/
│
├── .streamlit/
│   └── secrets.toml         # API keys (gitignored)
│
├── app.py                   # Main Streamlit UI
├── data_loader.py           # Flow 1: Version-aware caching
├── agent_setup.py           # Flow 2: Analytics agent
├── presentation.py          # Visualization & synthesis
├── config.py                # Configuration constants
│
├── cache_log.json           # Cache metadata (autocreated)
├── agri_data.parquet        # Data cache (autocreated)
├── climate_data.parquet     # Data cache (autocreated)
│
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🔧 Configuration

### Cache Settings

Edit `config.py` to adjust cache behavior:

```python
CACHE_REFRESH_INTERVAL_HOURS = 24  # How often to check for updates
API_LIMIT_PER_REQUEST = 1000       # Records per API call
API_MAX_RECORDS = 300000           # Safety limit
```

### LLM Settings

```python
LLM_MODE = "external"          # "external" or "local"
LLM_TEMPERATURE = 0.1          # Lower = more deterministic
LLM_MAX_TOKENS = 2000          # Response length
```

### Agent Settings

```python
AGENT_VERBOSE = True           # Show reasoning steps
AGENT_MAX_ITERATIONS = 10      # Max iterations to solve query
```

## 🔐 Data Sovereignty

- All data processing happens **locally** on your machine
- No sensitive data is sent to external services (except for LLM prompts)
- For 100% sovereignty, use a locally-hosted LLM (Llama, etc.)
- Cache files are stored locally and never transmitted

## 🤝 Contributing

This is a prototype/demo project. Key areas for enhancement:

- [ ] Add more data sources (soil quality, irrigation, etc.)
- [ ] Implement local LLM support (Llama integration)
- [ ] Add user authentication and multi-tenancy
- [ ] Export reports to PDF/Word
- [ ] Advanced visualizations (maps, interactive charts)
- [ ] Real-time data streaming

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- Data source: [data.gov.in](https://data.gov.in)
- Built with: Streamlit, Pandas, LiteLLM, SciPy
- Inspired by: India's digital public infrastructure initiatives

## 📧 Contact

[Your Contact Information]

---

**Project Samarth** - Empowering data-driven agricultural policy with sovereign AI analytics.

