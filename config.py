"""
Configuration file for Project Samarth
Contains all non-secret constants and settings.
"""

# --- Data.gov.in API Configuration ---
API_BASE_URL = "https://api.data.gov.in/resource"

# Resource IDs for our curated datasets
RESOURCES = {
    "agriculture": {
        # This is the "District-wise, season-wise crop production statistics"
        "id": "35be999b-0208-4354-b557-f6ca9a5355de", 
        "cache_file": "agri_data.parquet",
        "description": "District-wise crop production data (Annual) [1997-2025]"
    },
    "climate": {
        # This is the "Daily District-wise Rainfall Data"
        "id": "6c05cd1b-ed59-40c2-bc31-e314f39c6971", 
        "cache_file": "climate_data.parquet",
        "description": "District-wise rainfall data (Daily/Monthly) [2018-2025]"
    }
}

# --- Cache Configuration ---
CACHE_LOG_FILE = "cache_log.json"
# We check for new data version once every 24 hours
CACHE_CHECK_INTERVAL_SECONDS = 24 * 60 * 60 

# --- API Pagination ---
# How many records to fetch in a single API call
API_LIMIT_PER_REQUEST = 5000 
# Safety cap to prevent infinite loops. Agri data has ~250k records.
API_MAX_RECORDS_SAFETY_CAP = 350000 

# --- LLM & Agent Configuration ---
# LLM Mode: "external" (uses API like Gemini/OpenAI) or "local" (uses Ollama)
LLM_MODE = "external"  

# Model name for LiteLLM (examples below)
# For external APIs:
#   - "gpt-4o-mini" (OpenAI GPT-4o mini - Recommended, fast & affordable)
#   - "gpt-4o" (OpenAI GPT-4o - Most capable)
#   - "gpt-3.5-turbo" (OpenAI GPT-3.5 - Cheapest)
#   - "gemini/gemini-pro" (Google Gemini)
#   - "claude-3-opus-20240229" (Anthropic)
# For local:
#   - "ollama/llama3:8b" (Ollama with Llama 3)
LLM_MODEL_NAME = "gpt-4o-mini"

# Local LLM API base (only used if LLM_MODE = "local")
LLM_LOCAL_API_BASE = "http://localhost:11434"

# LLM parameters
LLM_TEMPERATURE = 0.0  # Set to 0.0 for maximum accuracy and deterministic code-gen
LLM_MAX_TOKENS = 2000

# Agent parameters
AGENT_VERBOSE = True   # MUST be True for debugging
AGENT_MAX_ITERATIONS = 4
AGENT_HANDLE_PARSING_ERRORS = True # Good robustness setting

# --- Visualization Configuration ---
DEFAULT_CHART_HEIGHT = 400
MAX_CHART_POINTS = 1000  # Limit data points for performance

# --- Climate Data Chunking Configuration ---
# States to fetch for climate data (bypasses 10k offset API limit)
INDIAN_STATES = [
    "Uttar Pradesh", "Madhya Pradesh", "Karnataka", "Bihar", "Assam", "Odisha", 
    "Tamil Nadu", "Maharashtra", "Rajasthan", "Chhattisgarh", "Andhra Pradesh", 
    "West Bengal", "Gujarat", "Haryana", "Telangana", "Uttarakhand", "Kerala", 
    "Nagaland", "Punjab", "Meghalaya", "Arunachal Pradesh", "Himachal Pradesh", 
    "Jammu and Kashmir", "Tripura", "Manipur", "Jharkhand", "Mizoram", 
    "Puducherry", "Sikkim", "Dadra and Nagar Haveli", "Goa", 
    "Andaman and Nicobar Islands"
]

# Years to fetch for climate data
CLIMATE_YEARS_RANGE = list(range(1997, 2026))  # 1997 to 2025

# --- Semantic Layer: Column Standardization ---
# This is the "brain" of our data cleaning.
# Keys = Raw column names from the API (based on your schemas & images)
# Values = Standardized names our agent will use
COLUMN_MAPPINGS = {
    "agriculture": {
        # Confirmed by your image:
        'state_name': 'State',
        'district_name': 'District',
        'crop_year': 'Year',
        'season': 'Season',
        'crop': 'Crop',
        'area_': 'Area',
        'production_': 'Production'
    },
    "climate": {
        # Based on your text schema, corrected by your image:
        'State': 'State',
        'District': 'District',
        'Year': 'Year',
        'Month': 'Month',
        'Avg Rainfall': 'Rainfall',
        'Agency_name': 'Agency'  # <-- THIS IS THE FIX (was 'Agency Name')
    }
}
