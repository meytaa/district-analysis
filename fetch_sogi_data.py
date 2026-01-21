import requests
import pandas as pd
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
YEAR = 2023
DATASET = "acs/acs1"
API_KEY = None  # No key in .env, attempting public access

BASE_URL = f"https://api.census.gov/data/{YEAR}/{DATASET}"

# Variables from B09019
VARIABLES = [
    "NAME",
    "B09019_011E",  # Same-sex spouse (persons)
    "B09019_013E",  # Same-sex unmarried partner (persons)
    "B09019_009E",  # Opposite-sex spouse (male householder)
    "B09019_010E",  # Opposite-sex spouse (female householder)
    # MOEs omitted for brevity as we just need the estimate for scoring
]

PARAMS = {
    "get": ",".join(VARIABLES),
    "for": "congressional district:*",
}

if API_KEY:
    PARAMS["key"] = API_KEY

print(f"Fetching data from {BASE_URL}...")

try:
    # -------------------------------
    # API REQUEST
    # -------------------------------
    response = requests.get(BASE_URL, params=PARAMS)
    response.raise_for_status()
    data = response.json()

    # -------------------------------
    # DATAFRAME
    # -------------------------------
    df = pd.DataFrame(data[1:], columns=data[0])

    # Convert numeric columns
    numeric_cols = VARIABLES[1:]  # all except NAME
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # -------------------------------
    # DERIVED METRICS
    # -------------------------------
    # Estimated number of same-sex couples (Mean of persons / 2)
    # Variable is "Persons in...", so count/2 is couples.
    df["same_sex_couples_est"] = (df["B09019_011E"] + df["B09019_013E"]) / 2

    # Total persons in couples (same-sex + opposite-sex)
    df["total_persons_in_couples"] = (
        df["B09019_011E"] + df["B09019_013E"] + df["B09019_009E"] + df["B09019_010E"]
    )
    
    # Total Couples = Total Persons / 2
    df["total_couples_est"] = df["total_persons_in_couples"] / 2

    # Rate: Same-Sex Couples / Total Couples
    df["same_sex_couple_rate"] = df["same_sex_couples_est"] / df["total_couples_est"]
    df["sogi_score_raw"] = df["same_sex_couple_rate"] * 100.0

    # -------------------------------
    # CLEAN OUTPUT
    # -------------------------------
    # Format State and District from Census "state", "congressional district"
    # Census returns "state" (FIPS) and "congressional district" (Number)
    # We need to map FIPS to State Abbr (e.g. 06 -> CA) to match our pipeline.
    
    # Simple FIPS mapping (partial list or load from pypi? us library?)
    # I'll rely on our existing pipeline to map it?
    # Actually, cipi_pipeline maps state_po based on state name or FIPS.
    # The API output has "NAME" like "Congressional District 1 (118th Congress), Alabama"
    # I can parse "NAME".
    
    def parse_name(name):
        # "Congressional District 1 (118th Congress), Alabama"
        parts = name.split(", ")
        if len(parts) >= 2:
            state_name = parts[-1]
            dist_part = parts[0]
            # dist_part: "Congressional District 1 (118th Congress)"
            try:
                dist_num = int(dist_part.split(" ")[2])
            except:
                dist_num = 0 # At Large? "Congressional District (at Large)..."
                if "at Large" in dist_part:
                    dist_num = 1 # Usually mapped to 00 or 01. 
                    # Our pipeline uses 00 for AL usually? Or 01?
                    # pipeline uses 01 usually.
                    pass
            return state_name, dist_num
        return None, None

    # Load state abbr mapping from pipeline helpers if possible?
    # Or just save the raw NAME/state/district columns and let cipi_pipeline handle the merge using FIPS.
    # cipi_pipeline uses `load_cvap(state_abbr)` mapping.
    # I'll save the raw columns `state` and `congressional district` and `NAME`.
    # And let cipi_pipeline handle the join.
    
    output_path = Path("all data") / "District Same Sex Couples 2023.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(output_path, index=False)
    print(f"Success! Saved {len(df)} rows to {output_path}")
    print(df[["NAME", "sogi_score_raw"]].head())

except Exception as e:
    print(f"Error: {e}")
