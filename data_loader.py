"""
Flow 1: The "Version-Aware" Data Pipeline
Handles intelligent data caching with freshness checks.
Includes State+Year+Month chunking for climate data BUT disables auto-refresh
for climate dataset in prototype mode for performance.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from config import (
    API_BASE_URL, RESOURCES, CACHE_LOG_FILE,
    CACHE_CHECK_INTERVAL_SECONDS, API_LIMIT_PER_REQUEST,
    API_MAX_RECORDS_SAFETY_CAP, COLUMN_MAPPINGS,
    INDIAN_STATES, CLIMATE_YEARS_RANGE # Ensure these are defined in config.py
)


class DataLoader:
    """Manages version-aware data loading and caching."""

    def __init__(self, api_key: str):
        """Initialize the DataLoader with an API key."""
        self.api_key = api_key
        self.cache_log = self._load_cache_log()

    # ... (Keep _load_cache_log, _save_cache_log, _check_daily_gate,
    #      _get_remote_metadata, _fetch_all_records,
    #      _fetch_climate_records_chunked, _clean_and_standardize
    #      exactly as they were in the previous correct version) ...
    # --- Paste the previous versions of these helper methods here ---

    def _load_cache_log(self) -> Dict:
        """Load the cache log file or create a new one."""
        if os.path.exists(CACHE_LOG_FILE):
            try:
                with open(CACHE_LOG_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                st.warning(f"Could not parse {CACHE_LOG_FILE}. Starting with empty log.")
                return {}
        return {}

    def _save_cache_log(self):
        """Save the cache log to disk."""
        try:
            with open(CACHE_LOG_FILE, 'w') as f:
                json.dump(self.cache_log, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save cache log: {e}")

    def _check_daily_gate(self, resource_key: str) -> bool:
        """
        Check if we need to check API for freshness.
        Returns True if cache is recent (< CACHE_CHECK_INTERVAL_SECONDS), False otherwise.
        """
        if resource_key not in self.cache_log:
            return False

        last_checked = self.cache_log[resource_key].get('last_checked_time')
        if not last_checked:
            return False

        try:
            last_checked_dt = datetime.fromisoformat(last_checked)
            seconds_since_check = (datetime.now() - last_checked_dt).total_seconds()
            return seconds_since_check < CACHE_CHECK_INTERVAL_SECONDS
        except ValueError:
            st.warning("Invalid timestamp found in cache log. Forcing freshness check.")
            return False


    def _get_remote_metadata(self, resource_id: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Lightweight freshness check - fetch only 1 record.
        Returns the ('updated' timestamp, 'total' records) from API.
        """
        url = f"{API_BASE_URL}/{resource_id}"
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": 1
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=20) # Increased timeout slightly
            response.raise_for_status()
            data = response.json()

            timestamp_str = data.get('updated')
            total_records_str = data.get('total')

            timestamp = int(timestamp_str) if timestamp_str and str(timestamp_str).isdigit() else None
            total_records = int(total_records_str) if total_records_str and str(total_records_str).isdigit() else None

            if timestamp is None:
                st.warning(f"API response missing 'updated' timestamp for {resource_id}.")
            # total_records might legitimately be missing or zero, don't warn unless needed for progress
            # if total_records is None:
            #      st.warning(f"API response missing 'total' records count for {resource_id}.")

            return timestamp, total_records

        except requests.exceptions.Timeout:
             st.warning(f"Timeout checking remote metadata for {resource_id}.")
             return None, None
        except requests.exceptions.RequestException as e:
            # Be more specific about potential 404s etc.
            st.warning(f"Could not check remote metadata for {resource_id} (Status: {response.status_code if 'response' in locals() else 'N/A'}): {e}")
            return None, None
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            st.warning(f"Error parsing metadata response for {resource_id}: {e}")
            return None, None


    def _fetch_all_records(self, resource_id: str, resource_name: str, total_records: Optional[int]) -> pd.DataFrame:
        """
        Fetch all records from API using pagination (the "slow path").
        Uses `total_records` for an accurate progress bar if available.
        """
        all_records = []
        offset = 0
        url = f"{API_BASE_URL}/{resource_id}"

        # Determine max_fetch and if progress can be shown accurately
        if total_records is not None and total_records > 0:
             max_fetch = min(total_records, API_MAX_RECORDS_SAFETY_CAP)
             show_progress = True
        else:
             max_fetch = API_MAX_RECORDS_SAFETY_CAP # Use safety cap if total unknown
             show_progress = False # Can't show accurate progress

        spinner_text = f"Fetching {resource_name} data from data.gov.in..."
        if show_progress:
            spinner_text += f" ({max_fetch:,} records estimate)..."
        else:
            spinner_text += f" (up to {max_fetch:,} records)..."


        with st.spinner(spinner_text):
            progress_bar = st.progress(0.0) if show_progress else None
            status_text = st.empty()

            while offset < max_fetch: # Loop until max_fetch or no more records
                params = {
                    "api-key": self.api_key,
                    "format": "json",
                    "limit": API_LIMIT_PER_REQUEST,
                    "offset": offset
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                try:
                    status_text.text(f"Requesting {resource_name} records from offset {offset:,}...")
                    response = requests.get(url, params=params, headers=headers, timeout=45) # Increased timeout
                    response.raise_for_status()
                    data = response.json()

                    records = data.get('records', [])
                    if not records:
                        status_text.text(f"No more {resource_name} records found at offset {offset:,}.")
                        break  # No more data

                    all_records.extend(records)
                    fetched_count = len(records)
                    offset += fetched_count

                    # Update progress bar if possible
                    if show_progress and progress_bar:
                        progress = min(offset / max_fetch, 1.0)
                        progress_bar.progress(progress, text=f"Fetched {offset:,} / {max_fetch:,}...")

                    # Break if we got fewer than requested (likely end of data)
                    if fetched_count < API_LIMIT_PER_REQUEST:
                        status_text.text(f"Received fewer records than limit, assuming end of data for {resource_name}.")
                        break

                    time.sleep(0.1) # Polite rate limiting

                except requests.exceptions.Timeout:
                    st.warning(f"Timeout fetching data at offset {offset}. Retrying after delay...")
                    status_text.warning(f"Timeout fetching data at offset {offset}. Retrying...")
                    time.sleep(5) # Wait longer after timeout
                    continue # Retry current offset
                except Exception as e:
                    st.error(f"Error fetching {resource_name} data at offset {offset}: {e}")
                    status_text.error(f"Error fetching {resource_name} data at offset {offset}: {e}")
                    break # Stop fetching on error

            if progress_bar:
                progress_bar.empty()
            status_text.empty()

        if not all_records:
            st.error(f"No records fetched for {resource_name}")
            return pd.DataFrame()

        st.caption(f"Fetch complete for {resource_name}. Total records retrieved: {len(all_records):,}")
        return pd.DataFrame(all_records)


    def _fetch_climate_records_chunked(self, resource_id: str, resource_name: str) -> pd.DataFrame:
        """
        Fetch climate records using State+Year+Month filtering to bypass 10k offset limit.
        """
        all_records = []
        url = f"{API_BASE_URL}/{resource_id}"

        MONTHS = list(range(1, 13))

        if not INDIAN_STATES or not CLIMATE_YEARS_RANGE:
            st.error("Configuration error: INDIAN_STATES or CLIMATE_YEARS_RANGE is empty or not defined in config.py.")
            return pd.DataFrame()

        total_chunks = len(INDIAN_STATES) * len(CLIMATE_YEARS_RANGE) * len(MONTHS)
        chunks_done = 0

        st.info(f"Fetching {resource_name} using **State+Year+Month** chunked strategy...")
        st.caption(f"Total chunks: {total_chunks:,} ({len(INDIAN_STATES)} states × {len(CLIMATE_YEARS_RANGE)} years × {len(MONTHS)} months). This will take a significant amount of time.")

        # --- User Confirmation ---
        # It's polite to confirm before starting a multi-hour process
        # However, for the prototype, we might skip this and just run it.
        # confirmed = st.button("Start Full Climate Data Fetch (Estimated 1-3 hours)?")
        # if not confirmed:
        #     st.warning("Fetch cancelled. Using existing cache if available.")
        #     # Attempt to load from cache as fallback
        #     cache_file = RESOURCES[resource_name]['cache_file']
        #     if os.path.exists(cache_file):
        #         try: return pd.read_parquet(cache_file)
        #         except: return pd.DataFrame()
        #     return pd.DataFrame()
        # -------------------------

        start_time = time.time()

        with st.spinner(f"Fetching {resource_name} data in chunks..."):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for i_state, state in enumerate(INDIAN_STATES):
                for i_year, year in enumerate(CLIMATE_YEARS_RANGE):
                    for i_month, month in enumerate(MONTHS):
                        chunk_name = f"{state}, {year}-{month:02d}"
                        status_text.text(f"Fetching [{chunks_done+1}/{total_chunks}]: {chunk_name}...")

                        offset = 0
                        chunk_records_count = 0
                        hit_10k_limit_in_chunk = False

                        while True:
                            if offset >= 10000:
                                st.warning(f"⚠️ Hit 10k offset limit *within* {chunk_name}. Data might be incomplete for this month.")
                                hit_10k_limit_in_chunk = True
                                break

                            params = {
                                'api-key': self.api_key,
                                'format': 'json',
                                'limit': API_LIMIT_PER_REQUEST,
                                'offset': offset,
                                'filters[State]': state,
                                'filters[Year]': str(year),
                                'filters[Month]': str(month)
                            }
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                            }

                            try:
                                response = requests.get(url, params=params, headers=headers, timeout=60)
                                response.raise_for_status()
                                data = response.json()

                                records = data.get('records', [])
                                if not records:
                                    break # No more records for this chunk

                                all_records.extend(records)
                                fetched_this_call = len(records)
                                chunk_records_count += fetched_this_call
                                offset += fetched_this_call

                                if fetched_this_call < API_LIMIT_PER_REQUEST:
                                    break

                                time.sleep(0.15) # Rate limiting

                            except requests.exceptions.Timeout:
                                st.warning(f"Timeout fetching {chunk_name} at offset {offset}. Retrying...")
                                time.sleep(5)
                                continue # Retry
                            except Exception as e:
                                st.error(f"❌ Error fetching {chunk_name} at offset {offset}: {e}")
                                # Option: Decide whether to continue to next chunk or stop entirely
                                break # Stop this chunk on error, continue to next State/Year/Month

                        # Update progress after each chunk (State+Year+Month) is done
                        chunks_done += 1
                        progress = chunks_done / total_chunks if total_chunks > 0 else 0.0
                        progress_bar.progress(progress)

                        elapsed_time = time.time() - start_time
                        est_total_time = (elapsed_time / chunks_done * total_chunks) if chunks_done > 0 else 0
                        est_remaining = est_total_time - elapsed_time

                        status_msg = f"{'⚠️ ' if hit_10k_limit_in_chunk else ''}[{chunks_done}/{total_chunks}] {chunk_name}: {chunk_records_count} recs | Total: {len(all_records):,} | ETA: {timedelta(seconds=int(est_remaining)) if est_remaining > 0 else 'Calculating...'}"
                        status_text.text(status_msg)
                        # Optionally print to console too
                        # print(status_msg)

            progress_bar.empty()
            status_text.empty()

        total_time_taken = time.time() - start_time
        st.success(f"✅ Chunked fetch complete: Fetched {len(all_records):,} total records in {timedelta(seconds=int(total_time_taken))}.")

        if not all_records:
            st.error(f"No records fetched for {resource_name}")
            return pd.DataFrame()

        return pd.DataFrame(all_records)


    def _clean_and_standardize(self, df: pd.DataFrame, resource_key: str) -> pd.DataFrame:
        """
        Transform and standardize raw data (The "Semantic Layer").
        """
        if df.empty:
            st.warning(f"Attempting to clean an empty DataFrame for resource '{resource_key}'.")
            return df

        st.info(f"Cleaning and standardizing {resource_key} data ({len(df):,} rows)...")
        initial_columns = df.columns.tolist() # Keep track for logging

        # Apply column mappings from config.py
        rename_dict = {}
        if resource_key in COLUMN_MAPPINGS:
            mappings = COLUMN_MAPPINGS[resource_key]
            # Build rename dict only for columns present in the DataFrame
            rename_dict = {old: new for old, new in mappings.items() if old in df.columns}
            if rename_dict:
                 df = df.rename(columns=rename_dict)
                 st.caption(f"Renamed columns based on config: {len(rename_dict)} columns.")
            else:
                 st.caption("No column renames applied based on config (or columns not found).")
        else:
             st.caption("No column mappings found in config for this resource.")

        # Get final column names after potential renaming
        final_cols = df.columns.tolist()

        # --- Resource-specific cleaning ---
        if resource_key == 'climate':
            col_year = next((c for c in final_cols if c.lower() == 'year'), None)
            col_month = next((c for c in final_cols if c.lower() == 'month'), None)
            col_rainfall = next((c for c in final_cols if c.lower() == 'rainfall'), None)
            col_date = next((c for c in final_cols if c.lower() == 'date'), None)
            col_state = next((c for c in final_cols if c.lower() == 'state'), None)
            col_district = next((c for c in final_cols if c.lower() == 'district'), None)
            col_agency = next((c for c in final_cols if c.lower() == 'agency'), None)

            # Convert numeric columns
            numeric_cols_to_convert = [col_year, col_month, col_rainfall]
            for col in numeric_cols_to_convert:
                if col: # Check if column exists
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert Date column if present
            if col_date:
                df[col_date] = pd.to_datetime(df[col_date], errors='coerce')

            # Clean text fields
            text_cols_to_clean = [col_state, col_district, col_agency]
            for col in text_cols_to_clean:
                if col:
                    df[col] = df[col].astype(str).str.strip() # Ensure string type

            # Drop rows with critical missing data for climate
            critical_cols = [col_year, col_month, col_rainfall, col_state, col_district]
            existing_critical = [col for col in critical_cols if col] # Filter out None values
            initial_rows = len(df)
            df = df.dropna(subset=existing_critical)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                st.caption(f"Dropped {dropped_rows:,} rows due to missing critical climate data.")

            # Convert Year and Month to integers after dropping NaNs
            if col_year and not df[col_year].isnull().all():
                df[col_year] = df[col_year].astype('Int64') # Use nullable integer
            if col_month and not df[col_month].isnull().all():
                df[col_month] = df[col_month].astype('Int64')

        elif resource_key == 'agriculture':
            col_state = next((c for c in final_cols if c.lower() == 'state'), None)
            col_district = next((c for c in final_cols if c.lower() == 'district'), None)
            col_year = next((c for c in final_cols if c.lower() == 'year'), None)
            col_area = next((c for c in final_cols if c.lower() == 'area'), None)
            col_production = next((c for c in final_cols if c.lower() == 'production'), None)
            col_season = next((c for c in final_cols if c.lower() == 'season'), None) # Added Season
            col_crop = next((c for c in final_cols if c.lower() == 'crop'), None) # Added Crop

            # Standardize state names
            if col_state:
                df[col_state] = df[col_state].astype(str).str.title().str.strip()
            # Clean other text cols
            text_cols_agri = [col_district, col_season, col_crop]
            for col in text_cols_agri:
                if col:
                     df[col] = df[col].astype(str).str.strip()

            # Convert numeric columns for agriculture
            numeric_cols_agri = [col_area, col_production, col_year]
            for col in numeric_cols_agri:
                if col:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where critical identifiers are missing
            critical_cols_agri = [col_state, col_district, col_year]
            existing_critical_agri = [col for col in critical_cols_agri if col]
            initial_rows = len(df)
            df = df.dropna(subset=existing_critical_agri)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                 st.caption(f"Dropped {dropped_rows:,} rows due to missing critical agriculture data.")

             # Convert Year to integer after dropping NaNs
            if col_year and not df[col_year].isnull().all():
                df[col_year] = df[col_year].astype('Int64')

        else:
            st.warning(f"No specific cleaning rules defined for resource key: {resource_key}")

        st.caption(f"Cleaning complete. Final shape: {df.shape}")
        return df


    def load_data(self, resource_key: str) -> pd.DataFrame:
        """
        Main entry point for loading data with version-aware caching.
        *** MODIFIED TO SKIP CLIMATE REFRESH FOR PROTOTYPE ***
        """
        if resource_key not in RESOURCES:
            st.error(f"Unknown resource key: {resource_key}")
            return pd.DataFrame()

        resource_info = RESOURCES[resource_key]
        resource_id = resource_info['id']
        cache_file = resource_info['cache_file']

        # Initialize cache log entry if needed
        if resource_key not in self.cache_log:
            self.cache_log[resource_key] = {
                'last_checked_time': None,
                'current_data_timestamp': None
            }

        # --- Check if cache file exists ---
        cache_exists = os.path.exists(cache_file)

        # --- If it's the climate dataset and cache exists, ALWAYS use it ---
        if resource_key == 'climate' and cache_exists:
            st.info("Using pre-downloaded baseline cache for climate data (refresh disabled for performance).")
            try:
                # Still perform a quick metadata check to *inform* the user
                remote_timestamp, _ = self._get_remote_metadata(resource_id)
                local_timestamp = self.cache_log[resource_key].get('current_data_timestamp')
                if remote_timestamp and local_timestamp and remote_timestamp > local_timestamp:
                    st.warning(" newer climate data detected on data.gov.in. Using cached version.")
                elif remote_timestamp and not local_timestamp:
                     st.warning("Could not determine local cache version. Using cached version.")

                df = pd.read_parquet(cache_file)
                st.success(f"✓ Using cached climate data ({len(df):,} records)")
                return df
            except Exception as e:
                st.error(f"Failed to load existing climate cache file {cache_file}: {e}. Trying to fetch.")
                # Fall through to fetch if cache is corrupted

        # --- For AGRICULTURE data, or if climate cache is MISSING ---
        # Proceed with normal version-aware logic

        # Step 2: The "Daily Gate"
        if self._check_daily_gate(resource_key):
            if cache_exists:
                try:
                    df = pd.read_parquet(cache_file)
                    # st.success(f"✓ Using recent cache for {resource_key.capitalize()}") # Less verbose
                    return df
                except Exception as e:
                    st.warning(f"Failed to load recent cache file {cache_file}: {e}. Will attempt refresh.")

        # Step 3: Lightweight "Freshness Check"
        st.caption(f"Checking for updates for {resource_key}...")
        remote_timestamp, total_records = self._get_remote_metadata(resource_id)
        current_time = datetime.now().isoformat()

        # Update last checked time immediately
        self.cache_log[resource_key]['last_checked_time'] = current_time
        self._save_cache_log()

        if remote_timestamp is None:
            if cache_exists:
                st.warning(f"Could not verify freshness for {resource_key}, using existing cache.")
                try: return pd.read_parquet(cache_file)
                except Exception as e:
                     st.error(f"Fatal: Failed to load existing cache file {cache_file}: {e}")
                     return pd.DataFrame()
            else:
                st.error(f"Fatal: No cache available and could not fetch metadata for {resource_key}")
                return pd.DataFrame()

        # Step 4 & 5: The "Smart Comparison" and "Fast Path"
        local_timestamp = self.cache_log[resource_key].get('current_data_timestamp')

        if local_timestamp == remote_timestamp and cache_exists:
            st.success(f"✓ {resource_key.capitalize()} data is up to date")
            try: return pd.read_parquet(cache_file)
            except Exception as e:
                st.warning(f"Failed to load up-to-date cache file {cache_file}: {e}. Will attempt refresh.")

        # --- Step 6: The "Slow Path" - Data IS Stale or Cache Missing/Corrupt ---
        st.info(f"⏳ Refreshing data for {resource_key.capitalize()}...")

        # 6a: API Loop - Use appropriate fetcher
        if resource_key == 'climate':
            # This path is now only hit if climate cache was MISSING or CORRUPTED
            st.warning("Climate cache missing or corrupted. Attempting full chunked fetch (this will take hours)...")
            df = self._fetch_climate_records_chunked(resource_id, resource_key)
        else: # Agriculture
            if total_records is None:
                 st.warning(f"Total records count unavailable for {resource_key}. Using safety cap.")
            df = self._fetch_all_records(resource_id, resource_key, total_records)

        if df.empty:
            st.error(f"Fetch process returned no data for {resource_key}.")
            if cache_exists:
                st.warning(f"Attempting to use existing (stale or corrupted) cache for {resource_key}")
                try: return pd.read_parquet(cache_file)
                except Exception as e:
                     st.error(f"Fatal: Failed to load stale/corrupted cache file {cache_file}: {e}")
                     return pd.DataFrame()
            return pd.DataFrame()

        # 6b: Transform & Standardize
        df = self._clean_and_standardize(df, resource_key)

        if df.empty:
             st.error(f"Data for {resource_key} became empty after cleaning. Cannot update cache.")
             if cache_exists:
                  st.warning(f"Returning existing (stale) cache for {resource_key}")
                  try: return pd.read_parquet(cache_file)
                  except Exception as e:
                       st.error(f"Fatal: Failed to load stale cache file {cache_file}: {e}")
                       return pd.DataFrame()
             return pd.DataFrame()

        # 6c: Cache Write
        try:
            df.to_parquet(cache_file, index=False)
            st.caption(f"Cache file '{cache_file}' updated.")
        except Exception as e:
            st.error(f"Failed to write cache file '{cache_file}': {e}")
            st.warning("Returning in-memory data frame, but it won't be cached for next run.")
            return df

        # 6d: Log Update
        self.cache_log[resource_key]['current_data_timestamp'] = remote_timestamp
        self._save_cache_log()

        st.success(f"✓ Fresh {resource_key.capitalize()} data loaded and cached ({len(df):,} records)")

        return df


# --- Main Functions for app.py ---

@st.cache_resource(show_spinner=False) # Caches the DataLoader instance
def get_data_loader(api_key: str) -> DataLoader:
    """Gets a cached instance of the DataLoader."""
    return DataLoader(api_key)

# This is the function your app.py will call
def load_all_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for Streamlit app.
    Loads all required datasets (agriculture and climate).
    Returns tuple of (agriculture_df, climate_df).
    """
    try:
        api_key = st.secrets["DATA_GOV_API_KEY"]
        if not api_key: raise KeyError # Handle empty key
    except (FileNotFoundError, AttributeError): # Handle missing secrets object or file
        st.error("Missing `.streamlit/secrets.toml` file.")
        st.info("Please create a `.streamlit/secrets.toml` file with `DATA_GOV_API_KEY = \"your_key_here\"`.")
        return pd.DataFrame(), pd.DataFrame()
    except KeyError:
        st.error("Missing or empty `DATA_GOV_API_KEY` in `.streamlit/secrets.toml`.")
        st.info("Please add `DATA_GOV_API_KEY = \"your_key_here\"` to your `.streamlit/secrets.toml` file.")
        return pd.DataFrame(), pd.DataFrame()

    loader = get_data_loader(api_key)

    st.subheader("📊 Data Pipeline Status")
    col1, col2 = st.columns(2)

    df_agri = pd.DataFrame()
    df_climate = pd.DataFrame()

    with col1:
        st.caption("Loading Agriculture Data...")
        df_agri = loader.load_data('agriculture')

    with col2:
        st.caption("Loading Climate Data...")
        df_climate = loader.load_data('climate')

    st.divider() # Add separator

    if df_agri.empty or df_climate.empty:
        st.error("Critical Error: One or more essential datasets could not be loaded. Check messages above. The application cannot proceed.")
        # Optionally: st.stop() here if datasets are absolutely required
    else:
        st.success("✓ All datasets loaded successfully.")

    return df_agri, df_climate