""""
Project Samarth - Sovereign-Ready Analytics Agent
Main Streamlit Application (Custom Agent + LiteLLM + DataFrame Capture)

**MODIFIED to use Streamlit Chat Interface**
"""

import streamlit as st
import pandas as pd # Import pandas for type hinting
import time

# Import our custom modules
import data_loader
import agent_setup # Uses the custom AnalyticsAgent class
import presentation # Uses the version expecting a DataFrame object
from config import LLM_MODE # Import to check which keys are needed

# Page Configuration
st.set_page_config(
    page_title="Project Samarth - Analytics Agent",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_api_keys() -> bool:
    """Verify that required API keys are configured in secrets.toml"""
    missing = []

    # Check Data.gov.in key
    if "DATA_GOV_API_KEY" not in st.secrets or not st.secrets["DATA_GOV_API_KEY"]:
        missing.append("DATA_GOV_API_KEY")

    # Check LLM keys based on mode
    if LLM_MODE == "external":
        # Needs *some* key LiteLLM can use (could be provider-specific or a proxy key)
        # Check common ones explicitly for better error messages
        has_openai = "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]
        has_gemini = "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]
        has_anthropic = "ANTHROPIC_API_KEY" in st.secrets and st.secrets["ANTHROPIC_API_KEY"]
        # Add checks for other major providers if needed (Azure, Cohere, etc.)
        has_litellm_proxy = "LITELLM_API_KEY" in st.secrets and st.secrets["LITELLM_API_KEY"]

        if not (has_openai or has_gemini or has_anthropic or has_litellm_proxy):
            missing.append("An LLM API Key (e.g., OPENAI_API_KEY, GEMINI_API_KEY, or LITELLM_API_KEY for a proxy)")

    elif LLM_MODE == "local":
        # No API key needed, but maybe check if API base is set if required by config
        pass # Assuming local model doesn't need key

    if missing:
        st.error(f"❌ Missing required API keys/config in `.streamlit/secrets.toml`: {', '.join(missing)}")
        st.info("""
        Please create/update your `.streamlit/secrets.toml` file.

        **Required:**
        - `DATA_GOV_API_KEY` = "your_data.gov.in_api_key"

        **For LLM (check `LLM_MODE` in `config.py`):**

        If `LLM_MODE` = "**external**":
        Provide API key for your chosen model provider (LiteLLM uses standard env var names):
        - `OPENAI_API_KEY` = "sk-..."
        - `GEMINI_API_KEY` = "..."
        - `ANTHROPIC_API_KEY` = "..."
        - *Or*, if using a LiteLLM proxy:
          - `LITELLM_API_KEY` = "your_proxy_key"
          - `LITELLM_API_BASE` = "your_proxy_url"

        If `LLM_MODE` = "**local**":
        - Ensure your local LLM server (e.g., Ollama) is running.
        - Set `LLM_MODEL_NAME` in `config.py` (e.g., "ollama/llama3").
        - Optionally set `LLM_LOCAL_API_BASE` in `config.py` if not default.
        """)
        return False

    return True


def init_session_state():
    """Initialize Streamlit session state variables"""
    # Use more descriptive names if helpful
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False # Tracks if load_app_components succeeded
    if 'df_agri' not in st.session_state:
        st.session_state.df_agri = pd.DataFrame() # Initialize as empty DF
    if 'df_climate' not in st.session_state:
        st.session_state.df_climate = pd.DataFrame()
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = None
    if 'analytics_agent' not in st.session_state: # Renamed from 'agent'
        st.session_state.analytics_agent = None
    if 'presentation_layer' not in st.session_state: # Renamed from 'presenter'
        st.session_state.presentation_layer = None
    
    # NEW: Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # NEW: Add welcome message if chat is new and pipeline is ready
    # Note: We'll add the welcome message *after* pipeline init in render_main_content


def load_app_components():
    """
    Execute Flow 1: Load and cache data
    Execute Flow 2: Initialize Agent and Presenter using CUSTOM agent factory
    """
    st.session_state.pipeline_initialized = False # Reset status
    try:
        # 1. Get LLM config using the function from agent_setup (custom version)
        llm_config = agent_setup.get_llm_config()
        if llm_config is None:
            st.error("Failed to configure LLM. Cannot proceed.")
            return False
        st.session_state.llm_config = llm_config

        # 2. Load data (Flow 1) - Function from data_loader.py
        df_agri, df_climate = data_loader.load_all_datasets()

        # Check if data loading was successful before proceeding
        if df_agri.empty or df_climate.empty:
            st.error("Data loading failed or returned empty dataframes. Cannot initialize agent.")
            # Keep existing (potentially empty) DFs in state for display
            st.session_state.df_agri = df_agri
            st.session_state.df_climate = df_climate
            return False

        st.session_state.df_agri = df_agri
        st.session_state.df_climate = df_climate

        # 3. Create agent (Flow 2 - CUSTOM agent factory)
        # This now uses the create_analytics_agent from the *custom* agent_setup.py
        st.session_state.analytics_agent = agent_setup.create_analytics_agent(df_agri, df_climate)
        if st.session_state.analytics_agent is None:
            st.error("Failed to create the analytics agent.")
            return False

        # 4. Create presenter (Flow 2 - Presentation layer factory)
        # Pass the llm_config needed for the synthesis step
        st.session_state.presentation_layer = presentation.create_presentation_layer(llm_config)
        if st.session_state.presentation_layer is None:
            st.error("Failed to create the presentation layer.")
            return False

        st.session_state.pipeline_initialized = True # Mark successful init
        return True

    except Exception as e:
        st.error(f"Fatal error during application component loading: {e}")
        st.exception(e) # Show full traceback
        return False


def render_sidebar():
    """Render the sidebar with project info and controls"""
    with st.sidebar:
        st.title("🌾 Project Samarth")
        st.caption("Sovereign-Ready Analytics Agent")
        st.divider()
        st.subheader("📊 Data Status")
        if st.session_state.pipeline_initialized:
            st.success("✓ Data Pipeline Active")
            # Show metrics only if DFs are not None and not empty
            agri_len = len(st.session_state.df_agri) if st.session_state.df_agri is not None else 0
            climate_len = len(st.session_state.df_climate) if st.session_state.df_climate is not None else 0
            st.metric("Agriculture Records", f"{agri_len:,}")
            st.metric("Climate Records", f"{climate_len:,}")
        else:
            st.warning("⚠ Data Pipeline Inactive")
            st.caption("Click 'Initialize' in main panel.")

        st.divider()
        st.subheader("💡 Example Queries")
        st.markdown("""
        - What was the trend in rice production in Maharashtra between 2010 and 2014?
        - Which 5 districts had the highest sugarcane production in 2014?
        - Compare total annual rainfall in Pune vs Nashik for 2019, 2020, 2021.
        - Synthesize arguments for promoting drought-resistant crops in Vidarbha based on historical agri data (1997-2014) and recent climate trends (2018-2025).
        """)
        st.divider()
        with st.expander("ℹ️ About This Project"):
            st.markdown(f"""
            **Project Samarth** uses a **custom agent** powered by **LiteLLM** for flexible model integration (current mode: `{LLM_MODE}`). It performs **version-aware caching** and reasons across data gaps.

            **Features:**
            - Smart Caching (Checks daily)
            - Natural Language Query -> Code Execution (`pandas`, `scipy`)
            - Handles Data Gaps (1997-2014 vs 2018-2025) via Indirect Reasoning
            - **Direct DataFrame Capture** for robust charting
            - Model Agnostic (via LiteLLM)
            """)
        
        # REMOVED: Query history is now in the main chat panel
        # st.divider()
        # st.subheader("📜 Recent Queries")


def render_main_content():
    """Render the main content area"""

    if not st.session_state.pipeline_initialized:
        st.title("🌾 Welcome to Project Samarth")
        st.info("Click the button below to initialize the data pipeline and activate the agent.")
        # Disable button if API keys are missing
        keys_ok = check_api_keys()
        if st.button("Initialize Data Pipeline", use_container_width=True, type="primary", disabled=not keys_ok):
            if keys_ok:
                with st.spinner("Initializing data pipeline... This may take several minutes on first run or if data is stale."):
                    if load_app_components():
                        # Use success message before rerun
                        st.success("Initialization Complete!")
                        time.sleep(1) # Brief pause before rerun
                        st.rerun()
                    else:
                        # Error messages handled within load_app_components
                        st.error("Initialization Failed. Please check the error messages above and your configuration.")
            else:
                 # Button is disabled, message handled by check_api_keys
                 pass

        st.divider()
        return

    # --- Main CHAT Interface (if data is loaded and agent/presenter exist) ---
    st.title("🤖 Chat with Samarth")
    st.caption("I am an AI assistant with access to Agriculture (1997-2014) and Climate (2018-2025) data.")

     # Data viewer
    st.divider()
    with st.expander("📋 View Loaded Data (First 100 Rows)"):
        tab1, tab2 = st.tabs(["Agriculture Data (1997-2014)", "Climate Data (2018-2025)"])
        with tab1:
            # Check if dataframe exists and is not empty
            if st.session_state.df_agri is not None and not st.session_state.df_agri.empty:
                st.dataframe(st.session_state.df_agri.head(100), use_container_width=True, height=300)
            else:
                 st.caption("Agriculture data not loaded.")
        with tab2:
            if st.session_state.df_climate is not None and not st.session_state.df_climate.empty:
                st.dataframe(st.session_state.df_climate.head(100), use_container_width=True, height=300)
            else:
                 st.caption("Climate data not loaded.")
    st.divider()

    # Add welcome message if chat is new
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "is_complex": False, # Flag for simple text
            "content": "Hello! I am Samarth, your policy analytics assistant. The data pipeline is active. Please ask me a question about the provided agriculture and climate data."
        })

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("is_complex", False):
                # This is a complex agent response (a result dict)
                # We need the original query that led to this response
                query = message["content"].get("query", "...") 
                st.session_state.presentation_layer.present_results(query, message["content"])
            else:
                # This is a simple text message (user or welcome)
                st.markdown(message["content"])

    # Get new user input
    if prompt := st.chat_input("Ask your question... (e.g., What was the rice production in Maharashtra in 2014?)"):
        
        # Add user message to history and display it immediately
        st.session_state.messages.append({"role": "user", "is_complex": False, "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and add assistant response (with spinner)
        # This will be displayed *after* the st.rerun()
        with st.spinner("Analyzing... (Generating and executing code)"):
            try:
                # This is the main CUSTOM agent call
                result_dict = st.session_state.analytics_agent.query(prompt)
                
                # Add the *entire* agent response dictionary to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "is_complex": True, # Flag for complex agent output
                    "content": result_dict
                })

            except Exception as e:
                st.error(f"An unexpected error occurred while running the agent: {e}")
                st.exception(e)
                # Add a simple error message to chat history
                error_msg = f"Sorry, I encountered an error and could not complete your request: {e}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "is_complex": False,
                    "content": error_msg
                })
        
        # Rerun the app to display the new assistant message
        st.rerun()

   

def main():
    """Main application entry point"""
    init_session_state() # Initialize state variables first
    render_sidebar() # Render sidebar which might show data status

    # Perform API key check early
    if not check_api_keys():
        st.warning("API Keys not configured. Please check `.streamlit/secrets.toml`.")
        # Display main content skeleton even if keys missing, button will be disabled
        render_main_content()
        st.stop() # Stop further execution if keys are missing

    # Render main content - includes the 'Initialize' button OR the chat interface
    render_main_content()

if __name__ == "__main__":
    main()

