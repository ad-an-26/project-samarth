"""
Presentation layer for data visualization and synthesis.
"""

import json
import re
import io
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import streamlit as st
from litellm import completion
from config import (
    LLM_TEMPERATURE,
    AGENT_VERBOSE, DEFAULT_CHART_HEIGHT, MAX_CHART_POINTS
)

class PresentationLayer:
    """
    Handles presentation using the captured DataFrame and text output.
    """

    def __init__(self, llm_config: Dict[str, str]):
        self.llm_config = llm_config


    def _format_year_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to format year columns to display without commas.
        Returns a copy of the DataFrame with formatted year columns.
        """
        df_formatted = df.copy()
        year_cols = [col for col in df_formatted.columns if 'year' in col.lower()]
        for col in year_cols:
            if df_formatted[col].dtype == 'int64' or df_formatted[col].dtype == 'float64':
                df_formatted[col] = df_formatted[col].astype(str).str.replace(',', '')
        return df_formatted

    def _detect_datasets_used(self, query: str, output_text: str, code: str) -> list:
        """
        Detect which datasets (agriculture/climate) are used in the analysis.
        Returns a list of dataset names.
        """
        datasets_used = []
        
        agri_keywords = ['rice', 'wheat', 'sugarcane', 'crop', 'production', 'agriculture', 
                        'yield', 'area', 'season', 'kharif', 'rabi', 'df1', 'agri']
        
        climate_keywords = ['rainfall', 'temperature', 'climate', 'weather', 'humidity', 
                           'precipitation', 'df2', 'rain']
        
        query_lower = query.lower()
        output_lower = output_text.lower()
        code_lower = code.lower()
        
        combined_text = query_lower + " " + output_lower + " " + code_lower
        
        if any(keyword in combined_text for keyword in agri_keywords) or 'df1' in code_lower:
            datasets_used.append('agriculture')
        
        if any(keyword in combined_text for keyword in climate_keywords) or 'df2' in code_lower:
            datasets_used.append('climate')
        
        return datasets_used

    def _format_data_sources(self, datasets_used: list) -> str:
        """
        Format the data source citations based on which datasets were used.
        """
        if not datasets_used:
            return ""
        
        citations = []
        
        if 'agriculture' in datasets_used:
            citations.append(
                "**Agriculture Data Source:** District-wise, season-wise crop production statistics from 1997 | "
                "[data.gov.in](https://www.data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)"
            )
        
        if 'climate' in datasets_used:
            citations.append(
                "**Climate Data Source:** Daily District-wise Rainfall Data | "
                "[data.gov.in](https://www.data.gov.in/resource/daily-district-wise-rainfall-data)"
            )
        
        return "\n\n---\n\n**Data Sources:**\n" + "\n".join(f"- {citation}" for citation in citations)

    def _detect_chart_type(self, df: Optional[pd.DataFrame]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Intelligently detect the appropriate chart type"""
        if df is None or df.empty:
            return None, {}

        columns = df.columns.tolist()
        numeric_cols = []
        for col in columns:
             if pd.api.types.is_numeric_dtype(df[col]):
                   numeric_cols.append(col)
             else:
                  try:
                       if pd.to_numeric(df[col], errors='coerce').notna().any():
                            numeric_cols.append(col)
                  except Exception: pass

        time_cols = [col for col in columns if any(
            word in col.lower() for word in ['year', 'date', 'month', 'time']
        )]
        numeric_data_cols = [col for col in numeric_cols if col not in time_cols]

        if time_cols and numeric_data_cols:
            return 'line', {'x': time_cols[0], 'y': numeric_data_cols, 'height': DEFAULT_CHART_HEIGHT}
        if len(numeric_data_cols) >= 1 and len(columns) > len(numeric_data_cols):
            categorical_cols = [col for col in columns if col not in numeric_data_cols and col not in time_cols]
            if categorical_cols:
                return 'bar', {'x': categorical_cols[0], 'y': numeric_data_cols, 'height': DEFAULT_CHART_HEIGHT}
        if len(numeric_data_cols) >= 2 and not time_cols:
             return 'scatter', {'x': numeric_data_cols[0], 'y': numeric_data_cols[1], 'height': DEFAULT_CHART_HEIGHT}
        if len(numeric_data_cols) == 1:
             return 'bar', {'y': numeric_data_cols[0], 'height': DEFAULT_CHART_HEIGHT}

        st.caption("Could not determine specific chart type, defaulting.")
        return 'line', {'height': DEFAULT_CHART_HEIGHT}


    def visualize(self, result_df: Optional[pd.DataFrame]) -> bool:
        """Visualization using the captured DataFrame."""
        if result_df is None or result_df.empty:
            if AGENT_VERBOSE:
                st.info("ℹ️ No result DataFrame captured for visualization.")
            return False

        df_display = result_df

        agri_cols = ['Production', 'Area', 'Yield']
        climate_cols = ['Rainfall', 'Temperature', 'Humidity']

        df_cols_lower = [str(c).lower() for c in df_display.columns]
        
        has_agri_data = any(
            any(agri_proxy.lower() in col for agri_proxy in agri_cols)
            for col in df_cols_lower
        )
        
        has_climate_data = any(
            any(climate_proxy.lower() in col for climate_proxy in climate_cols)
            for col in df_cols_lower
        )
        
        is_mixed_era = has_agri_data and has_climate_data
        
        if is_mixed_era:
            st.info("ℹ️ A single chart is not displayed because the underlying data combines metrics from different, non-overlapping time periods (e.g., Agriculture 1997-2014 and Climate 2018-2025). Please refer to the data table below and the analysis summary for findings.")
            st.subheader("📊 Combined Data Table")
            df_mixed_formatted = self._format_year_columns(df_display)
            
            with st.expander("📋 View Full Result Data Table"):
                st.dataframe(df_mixed_formatted, use_container_width=True)
            return False

        if len(df_display) > MAX_CHART_POINTS:
            st.warning(f"Data has {len(df_display)} points. Displaying first {MAX_CHART_POINTS} for chart performance.")
            df_chart = df_display.head(MAX_CHART_POINTS).copy()
        else:
            df_chart = df_display.copy()
        
        df_chart = self._format_year_columns(df_chart)

        chart_type, config = self._detect_chart_type(df_chart)

        if chart_type is None:
            if AGENT_VERBOSE:
                st.info("ℹ️ Could not determine a suitable chart type for the result data.")
            df_nochart_formatted = self._format_year_columns(df_display)
            
            with st.expander("📋 View Result Data Table (No Chart Type Detected)"):
                 st.dataframe(df_nochart_formatted, use_container_width=True)
            return False

        try:
            st.subheader("📊 Visualization")
            x_col = config.get('x')
            y_col = config.get('y')

            valid_x = x_col and x_col in df_chart.columns
            valid_y = y_col and (
                (isinstance(y_col, list) and all(c in df_chart.columns for c in y_col)) or
                (isinstance(y_col, str) and y_col in df_chart.columns)
            )
            use_index_x = not valid_x and chart_type in ['line', 'bar']

            if chart_type == 'line':
                st.line_chart(df_chart.set_index(x_col) if valid_x else df_chart,
                                 y=y_col if valid_y else None, height=config.get('height'))
            elif chart_type == 'bar':
                 st.bar_chart(df_chart.set_index(x_col) if valid_x else df_chart,
                                 y=y_col if valid_y else None, height=config.get('height'))
            elif chart_type == 'scatter' and valid_x and valid_y:
                st.scatter_chart(df_chart, x=x_col, y=y_col, height=config.get('height'))
            else:
                 st.warning(f"Could not plot {chart_type} due to missing columns. Displaying table.")
                 st.dataframe(df_chart, use_container_width=True)


            df_display_formatted = self._format_year_columns(df_display)
            
            with st.expander("📋 View Full Result Data Table"):
                st.dataframe(df_display_formatted, use_container_width=True)
            return True

        except Exception as e:
            st.error(f"❌ Error creating visualization: {e}")
            st.exception(e)
            df_error_formatted = self._format_year_columns(df_display)
            
            with st.expander("📋 View Data Table (Chart Failed)"):
                st.dataframe(df_error_formatted, use_container_width=True)
            return False

    def synthesize(self, query: str, output_text: str, code: str, result_df: Optional[pd.DataFrame]) -> str:
        """LLM-Powered Synthesis using captured text output and DataFrame."""
        datasets_used = self._detect_datasets_used(query, output_text, code)
        
        system_prompt = """You are a policy analyst specializing in Indian agriculture and rural development.

Your task is to:
1. Analyze the computed results (text output and potentially a data table).
2. Summarize the key findings in clear, accessible language.
3. Provide data-backed policy recommendations or insights based *only* on the results.
4. Cite specific numbers and statistics found in the results.
5. Acknowledge any data limitations mentioned (like the 1997-2014 vs 2018-2025 gap).

Important constraints:
- Base your response ONLY on the provided computed results and the original query context.
- Do not make up or infer data that wasn't computed or present in the output.
- Be specific with numbers and time periods.
- Frame insights for policymakers.
- Maintain objectivity."""

        data_summary = ""
        if result_df is not None and not result_df.empty:
             data_summary = f"\n\nA data table with {result_df.shape[0]} rows and columns {result_df.columns.tolist()} was also produced:\n{result_df.head().to_string()}\n..."
             if len(result_df) > 5: data_summary += f"(Showing first 5 of {len(result_df)} rows)"

        user_prompt = f"""Original User Query: {query}

Code Executed:
```python
{code if code else 'N/A'}
```

Computed Results (Text Output):
```
{output_text if output_text else 'No text output captured.'}
```
{data_summary}

Please provide a synthesis for a policy brief, covering:
1. Clear summary of findings.
2. Key statistics (cite time periods: Agri 1997-2014, Climate 2018-2025).
3. Data-backed insights/recommendations acknowledging the data gap if relevant.
4. Any limitations mentioned in the text output."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = completion(
                messages=messages,
                **self.llm_config
            )

            synthesis = response.choices[0].message.content.strip()
            
            data_sources = self._format_data_sources(datasets_used)
            synthesis_with_sources = synthesis + data_sources
            
            return synthesis_with_sources
        except Exception as e:
            st.error(f"Synthesis LLM call failed: {e}")
            st.warning("Could not generate summary. Displaying raw output.")
            fallback_output = f"**Raw Code Output:**\n```\n{output_text if output_text else 'No text output captured.'}\n```"
            data_sources = self._format_data_sources(datasets_used)
            return fallback_output + data_sources


    def present_results(self, query: str, result: Dict[str, Any]) -> None:
        """Main presentation method using captured DataFrame and text output."""
        if not result['success']:
            st.error("❌ Analysis Failed")
            error_msg = result.get('error', 'Unknown error.')
            st.error(f"Error: {error_msg}")
            last_attempt = result['attempts'][-1] if result.get('attempts') else {}
            if last_attempt.get('code'):
                with st.expander("🔍 Debug: View Last Attempted Code"):
                    st.code(last_attempt['code'], language='python')
            if last_attempt.get('output_text'):
                 with st.expander("🔍 Debug: View Last Output"):
                       st.text(last_attempt['output_text'])
            return

        st.success("✓ Analysis Complete")

        result_df = result.get('result_df')
        self.visualize(result_df)
        st.divider()

        st.subheader("📝 Analysis Summary")
        output_text = result.get('output_text', '')
        code = result.get('code', '')

        with st.spinner("Generating insights..."):
            synthesis = self.synthesize(query, output_text, code, result_df)

        st.markdown(synthesis)

        if AGENT_VERBOSE:
              with st.expander("🔍 View Raw Code Output (stdout)"):
                    st.text(output_text if output_text else "No text output captured.")
              with st.expander("💻 View Final Executed Code"):
                  st.code(code if code else "No code executed.", language='python')
              with st.expander("🔄 View Agent Attempts"):
                   attempts = result.get('attempts', [])
                   for i, attempt in enumerate(attempts):
                        st.caption(f"Attempt {i+1}: Success={attempt['success']}, DF Captured={attempt['result_df_captured']}")
                        if attempt['error']: st.error(f"Error: {attempt['error']}")


def create_presentation_layer(llm_config: dict) -> PresentationLayer:
    """Factory function to create the presentation layer."""
    if not llm_config:
         st.error("LLM Configuration missing, cannot create Presentation Layer.")
         return None
    return PresentationLayer(llm_config)

