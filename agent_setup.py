""""
Flow 2: The "Analytics Agent" Runtime (Custom LiteLLM Implementation)
Uses custom exec() for better control and direct DataFrame capture for charts.
Injects the sophisticated "Indirect Reasoning" prompt.
"""

import os
import pandas as pd
import scipy.stats # Import scipy for the exec scope
import streamlit as st
from typing import Dict, Any, Optional, Tuple
from litellm import completion
import io
import sys
import re

from config import (
    LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    AGENT_VERBOSE, AGENT_MAX_ITERATIONS, AGENT_HANDLE_PARSING_ERRORS, LLM_MODE,
    LLM_LOCAL_API_BASE # For LiteLLM local mode if configured
)

# --- THE "INDIRECT REASONING" PROMPT (using df1, df2 for custom exec scope) ---
# MODIFIED per user request to be more explicit about data-first output.
AGENT_SYSTEM_PROMPT = """
**CRITICAL INSTRUCTION: START**
You are an expert Indian policy data analyst. You MUST use the two pandas DataFrames provided in the execution scope:
1. `df1` (Agriculture data, **Years: 1997-2014**)
2. `df2` (Climate data, **Years: 2018-2025**)
**DO NOT define or create your own sample DataFrames. ONLY use `df1` and `df2`.**
**CRITICAL INSTRUCTION: END**

# Available DataFrames & **TIME PERIODS**:
## df1 (Agriculture Data)
- **Covers Years:** 1997 to 2014
- **Columns:** {df_agri_cols}

## df2 (Climate Data)
- **Covers Years:** 2018 to 2025
- **Columns:** {df_climate_cols}

**!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**
**CRITICAL DATA GAP:** `df1` ends 2014. `df2` starts 2018. **NO OVERLAP.**
Direct year-on-year correlation/comparison between `df1` and `df2` is **IMPOSSIBLE**.
**!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

**YOUR TASK: REASON ACROSS THE GAP INTELLENTLY**
1.  **If asked for direct correlation/comparison spanning the gap:**
    * State the impossibility due to the gap.
    * Analyze trends within `df1` (1997-2014).
    * Analyze trends within `df2` (2018-2025).
    * Present findings separately.

2.  **If asked for policy arguments or synthesis:**
    * Acknowledge the gap.
    * Analyze relevant patterns in `df1` (1997-2014).
    * Analyze relevant patterns in `df2` (2018-2025).
    * Synthesize findings logically, stating the time period for each data point.

3.  **If asked a question answerable by only ONE dataset:**
    * Use only that dataset (`df1` or `df2`).


**TOOLS:**
You have `pandas` (as `pd`) and `scipy.stats`. **Ensure you `import scipy.stats` in your code if you use it.**

**TIME AGGREGATION REMINDER (Use ONLY when analyzing WITHIN df2):**
- **By YEAR:** `groupby(['State', 'District', 'Year'])['Rainfall'].sum()`
- **By SEASON:** Filter `df2` by `Month` (`Kharif`=[6-10], `Rabi`=[11-2], `Summer`=[3-5]) *before* grouping and summing `Rainfall`.

**OUTPUT FORMAT (CRITICAL):**
Your Python code MUST perform two tasks:
1.  **Calculate Data for Charts:** Calculate the final DataFrame result relevant to the primary query. This DataFrame will be *automatically captured* for visualization.
    * **Example:** `chart_df = df1[...].groupby(...).sum()`
    * **This variable (`chart_df` in the example) MUST be the last DataFrame calculated in your code.**

2.  **Print Key Numeric Findings:** You MUST use `print()` statements to output the key numeric data, statistics, and tables that directly answer the user's query. This printed text is used to generate the final analysis summary.
    * **Data-First:** Print the *data* (numbers, tables) first.
    * **Tabular Data:** If the result is a small table (e.g., top 5, year-over-year comparison), `print()` the DataFrame directly (e.g., `print(my_small_table_df)`).
    * **Key-Value Data:** If the result is a single number or a few key points, print them clearly as key-value pairs (e.g., `print(f"Total Rice Production (2014): {{total_rice_2014}} tonnes")`).
    * **Analysis Summary:** After printing the data, you may add a *brief* one or two-line text summary, BUT the main synthesis will be done by another process.
    * **ALWAYS** cite sources AND **time periods** in your `print()` statements: `(Source: Agriculture Data, 1997-2014)` or `(Source: Climate Data, 2018-2025)`.

* **Do NOT** define or create sample DataFrames. Use ONLY `df1` and `df2`.
**FINAL REMINDER: USE `df1` & `df2`. ACKNOWLEDGE GAP. CALCULATE THE FINAL RESULT DATAFRAME. PRINT THE NUMERIC FINDINGS.**
"""

class AnalyticsAgent:
    """
    Custom agent using LiteLLM and direct exec() for robust DataFrame capture.
    """

    def __init__(self, df_agri: pd.DataFrame, df_climate: pd.DataFrame, llm_config: Dict[str, str]):
        self.df_agri = df_agri
        self.df_climate = df_climate
        self.llm_config = llm_config
        self.execution_history = [] # Store past attempts for context

    def _get_dataframe_schemas(self) -> str:
        """Generate schema descriptions for the agent's context"""
        # (Keep your existing schema generation logic - it was good!)
        schema_text = "# Available DataFrames & TIME PERIODS:\n"
        agri_years = "N/A"
        climate_years = "N/A"
        max_agri_yr = None
        min_climate_yr = None

        if self.df_agri is not None and not self.df_agri.empty and 'Year' in self.df_agri.columns:
            min_agri_yr = int(self.df_agri['Year'].min())
            max_agri_yr = int(self.df_agri['Year'].max())
            agri_years = f"{min_agri_yr}-{max_agri_yr}"
            schema_text += f"## df1 (Agriculture Data) - Years: {agri_years}\n"
            schema_text += f"- **Columns:** {self.df_agri.columns.tolist()}\n\n"

        if self.df_climate is not None and not self.df_climate.empty and 'Year' in self.df_climate.columns:
            min_climate_yr = int(self.df_climate['Year'].min())
            max_climate_yr = int(self.df_climate['Year'].max())
            climate_years = f"{min_climate_yr}-{max_climate_yr}"
            schema_text += f"## df2 (Climate Data) - Years: {climate_years}\n"
            schema_text += f"- **Columns:** {self.df_climate.columns.tolist()}\n\n"

        # Add the Gap warning again for emphasis right with the schemas
        if max_agri_yr and min_climate_yr and max_agri_yr < min_climate_yr -1:
             schema_text += "**WARNING: Data Gap - No overlap between df1 and df2.**\n\n"

        return schema_text

    def _build_system_prompt(self) -> str:
        """Build the system prompt, injecting dynamic schema"""
        # Format the main prompt with schema info
        # Note: Ensure df_agri and df_climate are available here or pass them in
        return AGENT_SYSTEM_PROMPT.format(
            df_agri_cols=self.df_agri.columns.tolist() if self.df_agri is not None else "N/A",
            df_climate_cols=self.df_climate.columns.tolist() if self.df_climate is not None else "N/A"
        )

    def _build_user_prompt(self, query: str) -> str:
        """Build the user prompt (simple query for this agent)"""
        # No need to inject schema here, it's in the system prompt
        return f"User Query: {query}\n\nPlease write Python code to answer this query using df1 and df2."


    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Calls LiteLLM to generate code."""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = completion(
                messages=messages,
                **self.llm_config # Pass model, key, temp etc directly
            )

            # Extract code, handle potential markdown fences
            code = response.choices[0].message.content.strip()
            if code.startswith('```python'):
                code = code.split('```python', 1)[1].rsplit('```', 1)[0].strip()
            elif code.startswith('```'):
                code = code.split('```', 1)[1].rsplit('```', 1)[0].strip()

            # Basic validation: ensure it imports pandas
            if "import pandas" not in code:
                code = "import pandas as pd\n" + code
            # Ensure scipy is imported if used (optional enhancement)
            # if "scipy.stats" in code and "import scipy.stats" not in code:
            #     code = "import scipy.stats\n" + code


            return code
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return None

    def _find_last_dataframe_variable(self, code: str) -> Optional[str]:
        """Simple heuristic to find the last assigned DataFrame variable name."""
        # Look for lines like "result_df = ..." or "... .some_pandas_method()"
        # This is a basic heuristic and might fail on complex code.
        df_assign_pattern = r'^\s*([a-zA-Z_]\w*)\s*=\s*(pd\.DataFrame\(|df[12]\.|[a-zA-Z_]\w*\.)'
        # More robust: find last assignment overall
        last_var_assigned = None
        lines = code.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('print'):
                continue
            # Match simple assignments like "var = ..."
            match = re.match(r'^([a-zA-Z_]\w*)\s*=', line)
            if match:
                last_var_assigned = match.group(1)
                # print(f"DEBUG: Identified last variable: {last_var_assigned}") # Debug
                return last_var_assigned
        # print("DEBUG: Could not identify last assigned variable.") # Debug
        return None # Fallback if no variable found


    def _execute_code(self, code: str) -> Tuple[str, bool, Any, Optional[pd.DataFrame]]:
        """
        Safely execute code, capture print output, and the final DataFrame result.
        Returns: (output_text, success, exception, result_dataframe)
        """
        output_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer # Redirect stdout
        result_df = None
        exec_success = False
        exec_exception = None

        # Prepare execution environment
        exec_globals = {
            'pd': pd,
            'scipy': scipy, # Make scipy available
            'df1': self.df_agri, # Use df1/df2 names here
            'df2': self.df_climate,
            '__builtins__': __builtins__, # Allow basic builtins
        }
        exec_locals = {} # Capture assigned variables here

        # --- NEW: Identify target DataFrame variable ---
        target_var_name = self._find_last_dataframe_variable(code)
        # print(f"DEBUG: Target variable for capture: {target_var_name}") # Debug

        try:
            exec(code, exec_globals, exec_locals)
            exec_success = True

            # --- NEW: Capture the target DataFrame ---
            if target_var_name and target_var_name in exec_locals:
                potential_df = exec_locals[target_var_name]
                if isinstance(potential_df, pd.DataFrame):
                    result_df = potential_df
                    # print(f"DEBUG: Successfully captured DataFrame '{target_var_name}'. Shape: {result_df.shape}") # Debug
                # else:
                #     print(f"DEBUG: Variable '{target_var_name}' found but is not a DataFrame (Type: {type(potential_df)}).") # Debug
            # elif target_var_name:
            #     print(f"DEBUG: Target variable '{target_var_name}' not found in exec_locals.") # Debug

        except Exception as e:
            exec_exception = e
            print(f"--- Code Execution Error ---", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            print(f"Code:\n{code}", file=sys.stderr)
            exec_success = False
        finally:
            sys.stdout = original_stdout # Restore stdout

        output_text = output_buffer.getvalue()
        return output_text, exec_success, exec_exception, result_df


    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Main query method - uses custom exec for robust DataFrame capture.
        """
        result = {
            'query': user_query,
            'code': None,
            'output_text': None, # Stdout capture
            'result_df': None, # Captured DataFrame object
            'success': False,
            'error': None,
            'attempts': []
        }

        max_retries = AGENT_MAX_ITERATIONS if isinstance(AGENT_MAX_ITERATIONS, int) else 2 # Use config or default

        for attempt in range(max_retries + 1):
            if AGENT_VERBOSE: st.info(f"Agent Attempt {attempt+1}/{max_retries+1}")

            # Step 1: Build Prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(user_query)

            # Add error feedback for retries
            if attempt > 0 and result['attempts']:
                last_error = result['attempts'][-1]['error']
                user_prompt += f"\n\nPrevious attempt failed:\n{last_error}\nPlease fix the code."
                if AGENT_VERBOSE: st.warning(f"Retrying with error context: {last_error}")

            # Step 2: Generate Code
            if AGENT_VERBOSE: st.info("🧠 Generating code...")
            code = self._call_llm(system_prompt, user_prompt)

            if not code:
                result['error'] = "LLM failed to generate code."
                break # Exit loop if code generation fails

            # Step 3: Execute Code and Capture Results
            if AGENT_VERBOSE:
                st.info("⚙️ Executing code...")
                with st.expander("View Generated Code"): st.code(code, language='python')

            output_text, success, exception, result_df = self._execute_code(code)

            # Record attempt details
            attempt_details = {
                'code': code,
                'output_text': output_text,
                'success': success,
                'error': str(exception) if exception else None,
                'result_df_captured': result_df is not None
            }
            result['attempts'].append(attempt_details)

            if success:
                result['code'] = code
                result['output_text'] = output_text
                result['result_df'] = result_df # Store the captured DataFrame
                result['success'] = True
                result['error'] = None
                if AGENT_VERBOSE: st.success("✓ Code executed successfully.")
                break # Exit loop on success
            else:
                result['error'] = str(exception)
                if AGENT_VERBOSE: st.error(f"Execution Error: {exception}")
                # Continue to next attempt (retry loop)

        # Store final state in history
        self.execution_history.append(result)
        return result

# --- Factory Function ---

def get_llm_config() -> dict:
    """Gets LLM config from secrets/env, ready for LiteLLM."""
    # Prioritize environment variables, then secrets
    # Ensure keys match LiteLLM expectations (e.g., OPENAI_API_KEY)
    # This example assumes OpenAI via LiteLLM
    api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    api_base = os.getenv('LITELLM_API_BASE') or st.secrets.get('LITELLM_API_BASE')
    model = os.getenv('LLM_MODEL_NAME') or st.secrets.get("LLM_MODEL_NAME", LLM_MODEL_NAME)

    if not api_key and LLM_MODE == "external":
        st.error("Missing OpenAI API Key (OPENAI_API_KEY) in secrets or environment.")
        return None # Cannot proceed without key for external mode

    # Construct config for LiteLLM completion call
    config = {
        'model': model,
        'api_key': api_key,
        'api_base': api_base, # Optional: for custom endpoints/local models via OpenAI API format
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
    }
    # Remove None values as LiteLLM handles defaults or they might cause errors
    config = {k: v for k, v in config.items() if v is not None}

    if LLM_MODE == 'local' and 'api_base' not in config:
         st.warning("LLM_MODE is 'local' but no LITELLM_API_BASE is set. LiteLLM might default.")
         # You might need specific model names like "ollama/llama3" for local mode

    st.caption(f"✓ LLM configured for model: `{config.get('model', 'Default')}` (Mode: {LLM_MODE})")
    return config


def create_analytics_agent(df_agri: pd.DataFrame, df_climate: pd.DataFrame) -> Optional[AnalyticsAgent]:
    """
    Factory function to create the custom Analytics Agent.
    """
    llm_config = get_llm_config()
    if llm_config is None:
        return None

    if df_agri is None or df_agri.empty or df_climate is None or df_climate.empty:
         st.error("One or both dataframes are missing or empty. Agent cannot be created.")
         return None

    try:
        agent = AnalyticsAgent(df_agri, df_climate, llm_config)
        st.caption("✓ Custom Analytics Agent created successfully.")
        return agent
    except Exception as e:
        st.error(f"Failed to create Custom Analytics Agent: {e}")
        st.exception(e)
        return None
