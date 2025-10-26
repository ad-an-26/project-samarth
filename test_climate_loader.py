import streamlit as st
import requests
import pandas as pd
import sys
from config import COLUMN_MAPPINGS # Import cleaning rules from config
import time

# --- Constants for your CLIMATE Dataset ---
CLIMATE_RESOURCE_ID = "6c05cd1b-ed59-40c2-bc31-e314f39c6971"
# Hardcoding key for this test, as requested.
API_KEY = "579b464db66ec23bdd0000015e88a0c17c074bda65c4a0ca646a06b6"

BASE_URL = "https://api.data.gov.in/resource/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- SOLUTION: We MUST filter our queries to bypass the 10k offset limit ---

# 1. We need a list of years. Let's assume 1997-2025 based on the agri dataset.
YEARS_TO_FETCH = list(range(1997, 2026))

# 2. We need the list of States. Extracted from your API doc screenshot.
STATES_TO_FETCH = [
    "Uttar Pradesh", "Madhya Pradesh", "Karnataka", "Bihar", "Assam", "Odisha", 
    "Tamil Nadu", "Maharashtra", "Rajasthan", "Chhattisgarh", "Andhra Pradesh", 
    "West Bengal", "Gujarat", "Haryana", "Telangana", "Uttarakhand", "Kerala", 
    "Nagaland", "Punjab", "Meghalaya", "Arunachal Pradesh", "Himachal Pradesh", 
    "Jammu and Kashmir", "Tripura", "Manipur", "Jharkhand", "Mizoram", 
    "Puducherry", "Sikkim", "Dadra and Nagar Haveli", "Goa", 
    "Andaman and Nicobar Islands"
    # NOTE: Your screenshot was cut off. We'll use these for the test.
]

# We are not using @st.cache_data for this test to ensure it re-runs
def load_and_clean_climate_data(resource_id, api_key):
    """
    Fetches, cleans, and returns the Climate dataset.
    SOLVES the 10k offset limit by looping through filters (State, Year).
    """
    print(f"--- Starting CLIMATE data load for {resource_id} ---")
    
    all_records = []
    limit_per_call = 5000 # Max limit per API call
    
    # We are testing with a *small subset* to prove the logic
    states_to_test = ["Punjab", "Haryana"]
    years_to_test = [2018, 2019]
    
    print(f"TESTING with States: {states_to_test} and Years: {years_to_test}")

    total_chunks = len(states_to_test) * len(years_to_test)
    chunks_done = 0

    progress_bar = st.progress(0.0, text="Initializing filtered data fetch...")
    status_text = st.empty()

    for state in states_to_test:
        for year in years_to_test:
            
            chunk_name = f"State: {state}, Year: {year}"
            print(f"--- Fetching Chunk: {chunk_name} ---")
            status_text.text(f"Fetching: {chunk_name}...")
            
            offset = 0
            
            # This inner loop handles pagination *within* the (State, Year) filter
            # It will stop if it hits the 10k offset, but that's fine,
            # because no State+Year chunk should have > 10k records.
            while True: 
                params = {
                    'api-key': api_key,
                    'format': 'json',
                    'limit': limit_per_call,
                    'offset': offset,
                    'filters[State]': state,
                    'filters[Year]': str(year) # Year must be a string for API filter
                }

                try:
                    print(f"Requesting records: offset={offset}, limit={limit_per_call} for {chunk_name}")
                    response = requests.get(f"{BASE_URL}{resource_id}", headers=HEADERS, params=params, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    
                    records = data.get('records', [])
                    if not records:
                        print(f"--- No more records for {chunk_name} at offset {offset}. ---")
                        break # Go to the next year
                        
                    all_records.extend(records)
                    print(f"Got {len(records)} records. Total so far: {len(all_records)}")
                    
                    # If we got fewer records than the limit, we are done with this chunk
                    if len(records) < limit_per_call:
                        break
                    
                    # If we hit 10k, we must break this inner loop
                    if offset + limit_per_call >= 10000:
                        print(f"--- WARN: Hit 10k offset limit for {chunk_name}. Moving to next chunk. ---")
                        st.warning(f"Hit 10k offset limit for {chunk_name}. Some data may be missing.")
                        break

                    offset += limit_per_call
                    time.sleep(0.2)

                except Exception as e:
                    st.error(f"API Request Failed for {chunk_name}: {e}")
                    print(f"API Request Failed: {e}", file=sys.stderr)
                    break # Stop this chunk and move to the next
            
            # Update overall progress
            chunks_done += 1
            progress_bar.progress(float(chunks_done) / total_chunks, text=f"Processing: {chunk_name}")

    progress_bar.progress(1.0, text="Fetch complete! Cleaning data...")
    status_text.text("Fetch complete! Cleaning data...")
    print(f"--- Fetch loop finished. Got {len(all_records)} records. Converting & cleaning. ---")

    if not all_records:
        st.error("No climate records were fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # --- Climate Data Cleaning ---
    print("Raw columns:", df.columns.to_list())
    
    final_rename_dict = {}
    if 'climate' in COLUMN_MAPPINGS:
        mappings = COLUMN_MAPPINGS['climate']
        final_rename_dict = {old: new for old, new in mappings.items() if old in df.columns}
        print("Applying renames:", final_rename_dict)
        df = df.rename(columns=final_rename_dict)
        print("Columns AFTER rename:", df.columns.to_list())
    
    # Define final column names *after* rename
    # Use .get(original_name, original_name) as fallback
    col_state = final_rename_dict.get('State', 'State')
    col_district = final_rename_dict.get('District', 'District')
    col_year = final_rename_dict.get('Year', 'Year')
    col_month = final_rename_dict.get('Month', 'Month')
    col_rainfall = final_rename_dict.get('Avg_rainfall', 'Avg_rainfall') # This was the bug from your log
    col_date = 'Date' # Was not in mapping
    col_agency = final_rename_dict.get('Agency_name', 'Agency_name')

    # 2. Convert numeric columns
    numeric_cols = [col_year, col_month, col_rainfall]
    print(f"Attempting to convert to numeric: {numeric_cols}")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert Date column
    if col_date in df.columns:
        print(f"Attempting to convert '{col_date}' to datetime...")
        df[col_date] = pd.to_datetime(df[col_date], errors='coerce')

    # 3. Clean text fields
    text_cols = [col_state, col_district, col_agency]
    for col in text_cols:
         if col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                 df[col] = df[col].str.strip()
            else:
                 df[col] = df[col].astype(str).str.strip()

    # 4. Drop rows with critical missing data
    critical_cols = [col_year, col_month, col_rainfall, col_state, col_district, col_date]
    existing_critical_cols = [col for col in critical_cols if col in df.columns]
    print(f"Dropping rows with NaN in: {existing_critical_cols}")
    df = df.dropna(subset=existing_critical_cols)

    # 5. Convert Year/Month to integers
    if col_year in df.columns:
         df[col_year] = df[col_year].astype(int)
    if col_month in df.columns:
         df[col_month] = df[col_month].astype(int)

    print("--- Climate data cleaning complete. ---")
    progress_bar.empty()
    status_text.empty()
    
    return df

# --- Main Application Logic ---
st.title("🇮🇳 Project Samarth - Climate Data Loader Test (FIXED)")

# Note: This test does not use st.secrets, per your instruction.
if API_KEY == "YOUR_API_KEY_HERE":
    st.error("Please hardcode your API_KEY in the script to run this test.")
    st.stop()

try:
    df_climate = load_and_clean_climate_data(CLIMATE_RESOURCE_ID, API_KEY)
    
    if not df_climate.empty:
        st.success(f"Successfully loaded and cleaned {len(df_climate)} climate records (Test query: {['Punjab', 'Haryana']} for {['2018', '2019']}).")
        st.write("Cleaned Data Preview (first 10 rows):")
        st.dataframe(df_climate.head(10))
        st.write("Cleaned Columns:")
        st.write(df_climate.columns.to_list())
    else:
        st.error("Failed to load climate data. Check logs.")

except Exception as e:
    st.error(f"An unexpected error occurred in the main logic: {e}")
    st.exception(e)
    print(f"An unexpected error occurred in main logic: {e}", file=sys.stderr)

