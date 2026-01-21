from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("all data")
OUTPUT_DIR = Path("output")

# Weights
VACUUM_WEIGHT = 0.35
PROTEST_WEIGHT = 0.25
APATHY_WEIGHT = 0.20
DEMO_WEIGHT = 0.20

VACUUM_ICI_WEIGHT = 1.0  # Replaces Cook PVI with our Independent Context Index
VACUUM_OSBORN_WEIGHT = 0.5  # Replaces Hegemony - measures separate "Osborn" strategy path
VACUUM_DROPOFF_WEIGHT = 1.0
VACUUM_SWING_HISTORY_WEIGHT = 1.0

PROTEST_MAVERICK_WEIGHT = 1.0
PROTEST_SPLIT_WEIGHT = 1.0

APATHY_MIDTERM_WEIGHT = 1.5
APATHY_PRES_WEIGHT = 0.5
APATHY_REGISTRATION_WEIGHT = 0.5

DEMO_GEN_SHIFT_WEIGHT = 1.4
DEMO_NEW_RESIDENT_WEIGHT = 1.2
DEMO_SOGI_WEIGHT = 2.0           # SOGI - Dominant factor (Boosted request)
DEMO_DIVERSITY_WEIGHT = 0.8
DEMO_ORIGIN_DIVERSITY_WEIGHT = 0.6

# Dynamic Weighting Profiles
PROFILES = {
    "volatile_swing": {
        "Vacuum": 0.50, "Protest": 0.30, "Apathy": 0.10, "Demo": 0.10
    },
    "lazy_giant": {
        "Vacuum": 0.45, "Protest": 0.25, "Apathy": 0.20, "Demo": 0.10
    },
    "sleeping_giant": {
        "Vacuum": 0.15, "Protest": 0.10, "Apathy": 0.50, "Demo": 0.25
    },
    "freedom_coalition": {
        "Vacuum": 0.10, "Protest": 0.20, "Apathy": 0.20, "Demo": 0.50
    },
    "maverick_rebellion": {
        "Vacuum": 0.20, "Protest": 0.50, "Apathy": 0.15, "Demo": 0.15
    },
    "unawakened_future": {
        "Vacuum": 0.10, "Protest": 0.10, "Apathy": 0.40, "Demo": 0.40
    },
    "cultural_wave": {
        "Vacuum": 0.15, "Protest": 0.15, "Apathy": 0.15, "Demo": 0.55
    },
    "balanced_general": {
        "Vacuum": 0.25, "Protest": 0.25, "Apathy": 0.25, "Demo": 0.25
    }
}

# Differentiated thresholds per core (based on score distributions)
PROFILE_THRESHOLDS = {
    "Vacuum": 30,   # Max ~58, lower bar needed
    "Protest": 35,  # Most districts have 0, need low bar
    "Apathy": 55,   # Good distribution, moderate bar
    "Demo": 55      # Good distribution, moderate bar  
}

# =============================================================================
# INDEPENDENT CONTEXT INDEX (ICI)
# =============================================================================
# ICI replaces Cook PVI with our own calculation using presidential vote data.
#
# NEW FORMULA (Universal Competitiveness):
#   We want districts that are competitive in BOTH contexts (Universal).
#   We penalize deviation from National trends more heavily than State deviation.
#
#   penalty_national = |nat_dev| * ICI_NATIONAL_FACTOR (3)
#   penalty_state = |state_dev| * ICI_STATE_FACTOR (2)
#   ici_penalty = max(penalty_national, penalty_state)
#   score_ici = max(0, 100 - ici_penalty)
#
# INTERPRETATION:
#   - To score high, a district MUST be moderate vs Nation AND State.
#   - Being an outlier in just one is not enough; you must be generally competitive.
#   - State deviation is forgiven slightly more (factor 2) than national (factor 3).
# =============================================================================

# National presidential margins (D - R, positive = Democratic lean)
NATIONAL_MARGINS = {
    2016: 2.1,   # Clinton 48.2% - Trump 46.1%
    2012: 3.9,   # Obama 51.1% - Romney 47.2%
}
NATIONAL_AVG_MARGIN = sum(NATIONAL_MARGINS.values()) / len(NATIONAL_MARGINS)  # D+3.0

# ICI Scoring Factors (Multipliers for deviation)
ICI_STATE_FACTOR = 2      # Lower penalty for state deviation
ICI_NATIONAL_FACTOR = 3   # Higher penalty for national deviation


def to_float(s):
    s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"nan": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")


def load_ballot_and_state_map():
    ballot = pd.read_csv(DATA_DIR / "District ballot 1976-2024.csv")
    states = ballot[["state", "state_po"]].drop_duplicates()
    states["state_key"] = states["state"].str.upper()
    state_abbr = dict(zip(states["state_key"], states["state_po"]))
    
    recent = ballot[ballot["year"].isin([2022, 2024])]
    turn = recent.pivot_table(index="district_code", columns="year", values="totalvotes", aggfunc="first")
    turn = turn.rename(columns={2022: "votes_2022", 2024: "votes_2024"}).reset_index()
    
    base = (
        ballot[ballot["year"] == 2024][["district_code", "state_po", "state", "district"]]
        .drop_duplicates()
        .rename(columns={"state": "State", "district": "District"})
    )
    return turn, state_abbr, base


def load_cvap(state_abbr):
    cvap = pd.read_csv(DATA_DIR / "District CVAP Demographic 2024.csv")
    cvap["state_key"] = cvap["State"].str.upper()
    cvap["state_po"] = cvap["state_key"].map(state_abbr)
    cvap["district_code"] = cvap.apply(
        lambda r: f"{r['state_po']}-{int(r['District']):02d}" if pd.notna(r["state_po"]) else np.nan,
        axis=1,
    )
    return cvap[[
        "district_code", "Citizen_Voting_Age_Population", "Median_Age", "Median_Home_Value",
        "Median_Household_Income", "Total_Population", "White_Alone", 
        "Black_or_African_American_Alone", "Hispanic_or_Latino"
    ]]


def load_ici_data():
    """
    Load presidential voting data and calculate Independent Context Index (ICI).
    
    ICI measures partisan opportunity by finding the minimum deviation from 
    either national or state baseline, with context-adjusted scoring factors.
    
    Returns DataFrame with columns:
        - district_code: District identifier (e.g., "CA-21")
        - district_margin: Avg presidential margin (D-R), positive = Democratic
        - state_margin: State's average presidential margin
        - national_deviation: |district_margin - national_margin|
        - state_deviation: |district_margin - state_margin|
        - ici_value: min(national_deviation, state_deviation)
        - ici_context: "state" or "national" - which provides the minimum
    """
    # Load presidential data
    pres = pd.read_csv(DATA_DIR / "District presidential 2012-2016.csv")
    
    # Calculate district margins (D - R, positive = Democratic lean)
    pres["margin_2016"] = pres["Clinton %"] - pres["Trump %"]
    pres["margin_2012"] = pres["Obama %"] - pres["Romney %"]
    pres["district_margin"] = (pres["margin_2016"] + pres["margin_2012"]) / 2
    
    # Extract state from district code
    pres["state_po"] = pres["Dist"].str[:2]
    
    # Calculate state margins (aggregate from districts)
    state_margins = pres.groupby("state_po")["district_margin"].mean()
    pres["state_margin"] = pres["state_po"].map(state_margins)
    
    # Calculate deviations
    pres["national_deviation"] = (pres["district_margin"] - NATIONAL_AVG_MARGIN).abs()
    pres["state_deviation"] = (pres["district_margin"] - pres["state_margin"]).abs()
    
    # Identify at-large states (only 1 district in state)
    # For these, state_deviation is meaningless (district = state)
    state_district_counts = pres.groupby("state_po")["Dist"].count()
    at_large_states = state_district_counts[state_district_counts == 1].index
    pres["is_at_large"] = pres["state_po"].isin(at_large_states)
    
    # ICI = Maximum Weighted Deviation model
    # We penalize national deviation more (factor 3) than state deviation (factor 2)
    # The score is minimal penalty approach: 100 - max(nat_dev*3, state_dev*2)
    pres["penalty_national"] = pres["national_deviation"] * ICI_NATIONAL_FACTOR
    pres["penalty_state"] = np.where(
        pres["is_at_large"],
        0,  # At-large states have no "state deviation" penalty (or just rely on national)
        pres["state_deviation"] * ICI_STATE_FACTOR
    )
    
    # ICI Value is now the "Effective Deviation Penalty"
    # We take the MAXIMUM penalty to ensure the district is good in BOTH contexts
    pres["ici_penalty"] = pres[["penalty_national", "penalty_state"]].max(axis=1)
    
    # Track which context was the limiting factor (the one with higher penalty)
    pres["ici_context"] = np.where(
        pres["penalty_state"] > pres["penalty_national"], 
        "state", 
        "national"
    )
    
    # Clean up at-large context
    pres.loc[pres["is_at_large"], "ici_context"] = "national"
    
    # Standardize district code format
    pres["district_code"] = pres["Dist"].apply(
        lambda x: x if "-" in str(x) else np.nan
    )
    
    return pres[[
        "district_code", "district_margin", "state_margin",
        "national_deviation", "state_deviation", "ici_penalty", "ici_context"
    ]]


def parse_pvi(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    if s == "EVEN": return 0.0
    if not s or "+" not in s: return np.nan
    try:
        return float(s.split("+")[1])
    except ValueError:
        return np.nan


def parse_pvi_party(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    if not s or s == "EVEN": return np.nan
    return s[0] if s[0] in ("R", "D") else np.nan


def parse_inc_party(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if "(" in s and ")" in s:
        p = s.split("(")[-1].split(")")[0].strip().upper()
        if p in ("R", "D"): return p
    return np.nan


def hist_district_code(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    return s[:-3] + "-AL" if s.endswith("-00") else s


def load_pvi_history_swing():
    hist_dir = DATA_DIR / "District PVI 1999-2022"
    files = sorted(hist_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["district_code", "swing_flip_count", "swing_crossover_count", "PVI_value", "pvi_party"])

    frames = []
    for path in files:
        try:
            # Extract year from filename (e.g., "2022.csv" -> 2022)
            # Some files might be named "District PVI 2022.csv", need to handle that if stem isn't just int
            # Assuming files are just year numbers or have year at end?
            # Let's be robust: find the year in the stem
            import re
            match = re.search(r'\d{4}', path.stem)
            if not match: continue
            seq = int(match.group(0))
            
            # 12-Year Rolling Cap (2014-2026)
            if seq < 2014: continue
        except ValueError:
            continue
        df = pd.read_csv(path)
        pvi_cols = [c for c in df.columns if "PVI" in c and "Raw" not in c and "Rank" not in c]
        if not pvi_cols: continue
        
        tmp = pd.DataFrame()
        tmp["district_code"] = df["Dist"].apply(hist_district_code)
        tmp["pvi_party_hist"] = df[pvi_cols[0]].apply(parse_pvi_party)
        tmp["PVI_value_hist"] = df[pvi_cols[0]].apply(parse_pvi)
        tmp["inc_party_hist"] = df["Party"].astype(str).str.strip().str.upper() if "Party" in df.columns else np.nan
        tmp["seq"] = seq
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["district_code", "swing_flip_count", "swing_crossover_count", "PVI_value", "pvi_party"])

    hist = pd.concat(frames, ignore_index=True).dropna(subset=["district_code"]).sort_values(["district_code", "seq"])
    hist["sign"] = hist["pvi_party_hist"].map({"R": 1, "D": -1})
    hist["prev_sign"] = hist.groupby("district_code")["sign"].shift()
    hist["flip"] = hist["sign"].notna() & hist["prev_sign"].notna() & (hist["sign"] != hist["prev_sign"])
    hist["cross"] = hist["pvi_party_hist"].isin(["R", "D"]) & hist["inc_party_hist"].isin(["R", "D"]) & (hist["pvi_party_hist"] != hist["inc_party_hist"])

    flip_counts = hist.groupby("district_code")["flip"].sum().astype(float)
    cross_counts = hist.groupby("district_code")["cross"].sum().astype(float)
    
    latest = hist.sort_values(["district_code", "seq"]).groupby("district_code").tail(1)
    latest_pvi = latest[["district_code", "PVI_value_hist", "pvi_party_hist"]].rename(
        columns={"PVI_value_hist": "PVI_value", "pvi_party_hist": "pvi_party"}
    )

    swing = pd.DataFrame({"district_code": flip_counts.index})
    swing["swing_flip_count"] = flip_counts.values
    swing = swing.merge(cross_counts.rename("swing_crossover_count").reset_index(), on="district_code", how="left").fillna(0.0)
    return swing.merge(latest_pvi, on="district_code", how="left")


def load_pvi_and_pres():
    pres = pd.read_csv(DATA_DIR / "District presidential 2012-2016.csv").rename(columns={"Dist": "district_code"})
    pres["inc_party"] = pres["Incumbent"].apply(parse_inc_party)
    pres["clinton_pct"] = to_float(pres["Clinton %"])
    pres["trump_pct"] = to_float(pres["Trump %"])
    return pres[["district_code", "inc_party", "clinton_pct", "trump_pct"]]


def load_party(state_abbr):
    party = pd.read_csv(DATA_DIR / "District Party 2012-2024.csv")
    party["state_po"] = party["state"].str.upper().map(state_abbr)

    def make_code(row):
        if pd.isna(row["state_po"]): return np.nan
        d = row["district"]
        return f"{row['state_po']}-AL" if isinstance(d, str) and not d.isdigit() else f"{row['state_po']}-{int(d):02d}"

    party["district_code"] = party.apply(make_code, axis=1)
    cols = [
        "2016_republican_pct", "2016_democratic_pct", "2022_republican_pct", "2022_democratic_pct",
        "2016_libertarian_pct", "2016_green_pct", "2016_independent_pct",
        "2024_republican_pct", "2024_democratic_pct", "2024_libertarian_pct", "2024_green_pct", "2024_independent_pct"
    ]
    for c in cols:
        if c in party.columns: party[c] = to_float(party[c])
    return party


def load_state_osborn(state_abbr):
    sp = pd.read_csv(DATA_DIR / "State Party 2024.csv")
    sp["state_po"] = sp["state"].str.upper().map(state_abbr)
    for c, new in [("current_democratic_house", "dem"), ("current_republican_house", "rep"), ("current_independent_house", "ind")]:
        sp[new] = pd.to_numeric(sp[c], errors="coerce") if c in sp.columns else np.nan
    sp["house_total"] = sp[["dem", "rep", "ind"]].sum(axis=1, min_count=1)
    # The "Osborn Score" is simply the percentage control of the dominant party (State Hegemony)
    # Strategy: Independent runs against the "Lazy Giant" in a one-party state
    sp["score_osborn"] = sp[["dem", "rep", "ind"]].max(axis=1) / sp["house_total"] * 100.0
    return sp[["state_po", "score_osborn"]]


def load_state_registration():
    df = pd.read_excel(DATA_DIR / "State CVAP Percentage 1074-2022.xlsx", header=[3, 4])
    df.columns = [f"{top}_{sub}" if top != "State" else "State" for top, sub in df.columns]
    df["state_key"] = df["State"].str.upper()
    return df[["state_key", "2022_Total"]].rename(columns={"2022_Total": "state_registration_total"})


def _get_demo_value(df, subject, title):
    m = (df["Subject"] == subject) & (df["Title"] == title)
    if not m.any(): return np.nan
    try:
        return float(str(df.loc[m, "Value"].iloc[0]).replace(",", "").replace("\"", ""))
    except ValueError:
        return np.nan


def _load_demo_df(row):
    if pd.isna(row.get("State")) or pd.isna(row.get("District")): return None
    path = DATA_DIR / "District Demographic Distribution" / f"{str(row['State']).title().replace(' ', '_')}_District_{int(row['District']):02d}.csv"
    return pd.read_csv(path) if path.exists() else None


def generational_share(row):
    df = _load_demo_df(row)
    if df is None: return [np.nan, np.nan]
    
    pop18 = _get_demo_value(df, "Sex and Age", "18 years and over")
    
    # Updated: Define "Older" as 45+ (Gen X and Boomers). 
    # This leaves the target group as 18-44 (Gen Z + Millennials).
    older = sum(_get_demo_value(df, "Sex and Age", t) for t in [
        "45 to 54 years", "55 to 59 years", "60 to 64 years", "65 to 74 years", "75 to 84 years", "85 years and over"
    ])
    
    if not pop18 or np.isnan(pop18) or np.isnan(older): return [np.nan, np.nan]
    
    # Young = (18+) - (45+) = 18-44
    return [max(0.0, (pop18 - older) / pop18), max(0.0, older / pop18)]


def residence_mobility(row):
    df = _load_demo_df(row)
    if df is None: return [np.nan] * 7
    
    return [
        _get_demo_value(df, "Residence 1 Year Ago", "Population 1 year and over"),
        _get_demo_value(df, "Residence 1 Year Ago", "Same house"),
        _get_demo_value(df, "Residence 1 Year Ago", "Different house (in the U.S. or abroad)"),
        _get_demo_value(df, "Residence 1 Year Ago", "Same county"),
        _get_demo_value(df, "Residence 1 Year Ago", "Different county"),
        _get_demo_value(df, "Residence 1 Year Ago", "Different state"),
        _get_demo_value(df, "Residence 1 Year Ago", "Abroad")
    ]


def race_counts(row):
    df = _load_demo_df(row)
    if df is None: return [np.nan] * 8
    
    return [
        _get_demo_value(df, "Race", "Total population"),
        _get_demo_value(df, "Race", "White"),
        _get_demo_value(df, "Race", "Black or African American"),
        _get_demo_value(df, "Race", "American Indian and Alaska Native"),
        _get_demo_value(df, "Race", "Asian"),
        _get_demo_value(df, "Race", "Native Hawaiian and Other Pacific Islander"),
        _get_demo_value(df, "Race", "Some other race"),
        _get_demo_value(df, "Race", "Two or more races")
    ]


def origin_place(row):
    df = _load_demo_df(row)
    if df is None: return [np.nan] * 5
    
    return [
        _get_demo_value(df, "Place of Birth", "Total population"),
        _get_demo_value(df, "Place of Birth", "State of residence"),
        _get_demo_value(df, "Place of Birth", "Different state"),
        _get_demo_value(df, "Place of Birth", "Born in Puerto Rico, U.S. Island areas, or born abroad to American parent(s)"),
        _get_demo_value(df, "Place of Birth", "Foreign-born")
    ]


def load_midterm_dropoff_history():
    ballot = pd.read_csv(DATA_DIR / "District ballot 1976-2024.csv")
    sub = ballot[ballot["year"].isin([2014, 2016, 2018, 2020, 2022, 2024])].copy()
    if "totalvotes" in sub.columns:
        sub["totalvotes"] = pd.to_numeric(sub["totalvotes"], errors="coerce")
        sub = sub[sub["totalvotes"] > 1]
    
    if sub.empty: return pd.DataFrame(columns=["district_code", "gap_2024", "avg_gap_hist", "excess_gap"])

    votes = sub.pivot_table(index="district_code", columns="year", values="totalvotes", aggfunc="sum")
    frames = []
    for pres_year, mid_year in [(2016, 2014), (2020, 2018), (2024, 2022)]:
        if pres_year in votes.columns and mid_year in votes.columns:
            tmp = votes[[pres_year, mid_year]].dropna().reset_index()
            tmp["year_pres"] = pres_year
            tmp["gap_pct"] = (tmp[pres_year] - tmp[mid_year]) / tmp[pres_year] * 100.0
            frames.append(tmp[["district_code", "year_pres", "gap_pct"]])

    if not frames: return pd.DataFrame(columns=["district_code", "gap_2024", "avg_gap_hist", "excess_gap"])
    
    hist = pd.concat(frames, ignore_index=True)
    avg_gap = hist[hist["year_pres"] < 2024].groupby("district_code")["gap_pct"].mean()
    g2024 = hist[hist["year_pres"] == 2024].set_index("district_code")["gap_pct"]
    
    summary = pd.DataFrame(index=avg_gap.index.union(g2024.index))
    summary["gap_2024"] = g2024
    summary["avg_gap_hist"] = avg_gap
    summary["excess_gap"] = summary["gap_2024"] - summary["avg_gap_hist"]
    return summary.reset_index().rename(columns={"index": "district_code"})


def load_maverick_history():
    votes = pd.read_csv(DATA_DIR / "District Party Vote 1976-2024.csv")
    mask = (votes["office"] == "US HOUSE") & (votes["stage"] == "GEN") & votes["year"].isin([2020, 2022, 2024])
    sub = votes[mask].copy()
    if sub.empty: return pd.DataFrame(columns=["district_code", "minor_2020", "minor_2022", "minor_2024", "minor_hist_avg"])

    sub["major_contrib"] = np.where(sub["party"].isin(["DEMOCRAT", "REPUBLICAN"]), sub["candidatevotes"], 0)
    grp = sub.groupby(["year", "state_po", "district"])
    agg = pd.DataFrame({"totalvotes": grp["totalvotes"].max(), "majorvotes": grp["major_contrib"].sum()})
    agg["minor_pct"] = ((agg["totalvotes"] - agg["majorvotes"]) / agg["totalvotes"]) * 100.0
    agg = agg.reset_index()

    ballot = pd.read_csv(DATA_DIR / "District ballot 1976-2024.csv")
    agg = agg.merge(ballot[["year", "state_po", "district", "district_code"]].drop_duplicates(), on=["year", "state_po", "district"], how="left")

    pivot = agg.pivot_table(index="district_code", columns="year", values="minor_pct", aggfunc="first")
    for y in [2020, 2022, 2024]:
        if y not in pivot.columns: pivot[y] = np.nan
    
    pivot = pivot.rename(columns={2020: "minor_2020", 2022: "minor_2022", 2024: "minor_2024"})
    pivot["minor_hist_avg"] = pivot[["minor_2020", "minor_2022", "minor_2024"]].mean(axis=1)
    return pivot.reset_index()


def load_split_ticket_history(state_abbr):
    pres = pd.read_csv(DATA_DIR / "District presidential 2012-2016.csv")
    frames = []
    for y, d_col, r_col in [(2012, "Obama %", "Romney %"), (2016, "Clinton %", "Trump %")]:
        tmp = pres[["Dist", d_col, r_col]].copy()
        tmp.columns = ["district_code", "dem", "rep"]
        tmp["year"] = y
        tmp["dem"] = to_float(tmp["dem"])
        tmp["rep"] = to_float(tmp["rep"])
        frames.append(tmp)
    pres_long = pd.concat(frames)
    pres_long["pres_winner"] = np.where(pres_long["dem"] > pres_long["rep"], "D", np.where(pres_long["rep"] > pres_long["dem"], "R", ""))

    house = pd.read_csv(DATA_DIR / "District Party 2012-2024.csv")
    house["state_po"] = house["state"].str.upper().map(state_abbr)
    house["district_code"] = house.apply(lambda r: f"{r['state_po']}-{int(r['district']):02d}" if str(r['district']).isdigit() else f"{r['state_po']}-AL", axis=1)
    
    h_frames = []
    for y in [2012, 2016]:
        rp, dp = f"{y}_republican_pct", f"{y}_democratic_pct"
        if rp in house.columns and dp in house.columns:
            tmp = house[["district_code", rp, dp]].copy()
            tmp.columns = ["district_code", "rep", "dem"]
            tmp["year"] = y
            tmp["rep"] = to_float(tmp["rep"])
            tmp["dem"] = to_float(tmp["dem"])
            h_frames.append(tmp)
    
    if not h_frames: return pd.DataFrame(columns=["district_code", "split_2012", "split_2016", "split_rate", "split_years"])
    
    house_long = pd.concat(h_frames)
    house_long["house_winner"] = np.where(house_long["dem"] > house_long["rep"], "D", np.where(house_long["rep"] > house_long["dem"], "R", ""))
    
    hist = pres_long.merge(house_long, on=["district_code", "year"], how="inner")
    valid = hist["pres_winner"].isin(["D", "R"]) & hist["house_winner"].isin(["D", "R"])
    hist["split_flag"] = np.where(valid & (hist["pres_winner"] != hist["house_winner"]), 1.0, np.where(valid, 0.0, np.nan))

    pivot = hist.pivot_table(index="district_code", columns="year", values="split_flag", aggfunc="first")
    for y in [2012, 2016]:
        if y not in pivot.columns: pivot[y] = np.nan
    pivot = pivot.rename(columns={2012: "split_2012", 2016: "split_2016"})
    pivot["split_rate"] = pivot[["split_2012", "split_2016"]].mean(axis=1)
    pivot["split_years"] = pivot[["split_2012", "split_2016"]].notna().sum(axis=1)
    return pivot.reset_index()


def load_sogi_data(state_abbr):
    """
    Load Same-Sex Couple data (ACS 2023).
    Parses 'NAME' column to extract State Name and District Number, acts as SOGI proxy.
    """
    path = DATA_DIR / "District Same Sex Couples 2023.csv"
    if not path.exists(): return pd.DataFrame(columns=["district_code", "sogi_score_raw"])
    
    df = pd.read_csv(path)
    
    rows = []
    for _, row in df.iterrows():
        name = row["NAME"]
        # Format: "Congressional District 1 (118th Congress), Alabama"
        parts = name.split(", ")
        if len(parts) < 2: continue
        state_name = parts[-1]
        
        # Get State Abbr
        # We need a reverse map from Full Name to Abbr. 
        # state_abbr passed in is Name -> Abbr? Let's check usage.
        # Yes, load_ballot_and_state_map returns state_abbr dict.
        # But wait, state_key in that dict is usually UPPERCASE.
        state_po = state_abbr.get(state_name.upper())
        if not state_po: continue
        
        dist_part = parts[0]
        # "Congressional District 1 (118th Congress)"
        # "Congressional District (at Large) (118th Congress)"
        # "Resident Commissioner District (at Large)..."
        
        dist_num = "00" # Default for at Large
        try:
            # Try to grab digit. "District 1 ..."
            # Split by space
            p = dist_part.split(" ")
            # p[2] should be number if "Congressional District X"
            if p[2].isdigit():
                dist_num = f"{int(p[2]):02d}"
            elif "at Large" in dist_part:
                if state_po in ["AK", "DE", "ND", "SD", "VT", "WY"]:
                    dist_num = "00"
                else: 
                    # Some states map At Large to 01 (e.g. MT before split? No, MT has 2 now).
                    # Check our ballot data convention. Usually 00 for true At Large.
                    dist_num = "00"
        except:
            pass
            
        district_code = f"{state_po}-{dist_num}"
        rows.append({
            "district_code": district_code,
            "sogi_score_raw": row["sogi_score_raw"] # This is the Rate % (0.5 to 5.0)
        })
        
    return pd.DataFrame(rows)


def build_master_table():
    turnout, state_abbr, base = load_ballot_and_state_map()
    
    # Load all data sources
    dfs = [
        load_cvap(state_abbr),
        load_party(state_abbr),
        load_pvi_and_pres(),
        load_state_osborn(state_abbr),
        load_state_registration(),
        load_pvi_history_swing(),
        load_midterm_dropoff_history(),
        load_maverick_history(),
        load_split_ticket_history(state_abbr),
        load_sogi_data(state_abbr),
        load_ici_data(),  # Independent Context Index (replaces Cook PVI)
    ]
    
    df = base.merge(turnout, on="district_code", how="left")
    for d in dfs:
        # Merge on available keys (usually district_code or state_po/state_key)
        if "district_code" in d.columns:
            df = df.merge(d, on="district_code", how="left", suffixes=("", "_party"))
        elif "state_po" in d.columns:
            df = df.merge(d, on="state_po", how="left")
        elif "state_key" in d.columns:
            df["state_key"] = df["State"].str.upper()
            df = df.merge(d, on="state_key", how="left")

    # Demographics
    df[["gen_18_54_share", "gen_55_plus_share"]] = df.apply(generational_share, axis=1, result_type="expand")
    
    mob_cols = ["mob_pop1plus", "mob_same_house", "mob_diff_house", "mob_same_county", "mob_diff_county", "mob_diff_state", "mob_abroad"]
    df[mob_cols] = df.apply(residence_mobility, axis=1, result_type="expand")
    
    race_cols = ["race_total", "race_white", "race_black", "race_aian", "race_asian", "race_nhopi", "race_other", "race_two_plus"]
    df[race_cols] = df.apply(race_counts, axis=1, result_type="expand")
    
    origin_cols = ["place_total", "place_native_state", "place_diff_state", "place_pr_island_abroadparents", "place_foreign"]
    df[origin_cols] = df.apply(origin_place, axis=1, result_type="expand")
    
    return df


def compute_scores(df):
    df = df.copy()
    
    # 1. Vacuum
    # ICI (Independent Context Index) - replaces Cook PVI
    # Formula: score = 100 - ici_penalty  (where penalty is already weighted)
    df["score_ici"] = np.maximum(
        0.0, 
        100.0 - df["ici_penalty"].fillna(50)
    )
    # score_osborn is already calculated in load_state_osborn (0-100 scale based on party dominance)
    
    # Refined Dropoff Score (Absolute Vacuum)
    # Using Avg Gap % (Presidential vs Midterm Turnout Dropoff)
    # Mean dropoff is ~33%. We map: 10% -> 0 score, 33% -> 55 score, 50% -> 100 score.
    # Formula: (gap - 10) * 2.5
    avg_gap = df["avg_gap_hist"].fillna(32.0) # Fill missing with mean
    df["score_dropoff"] = ((avg_gap - 10.0) * 2.5).clip(0, 100)
    
    df["pvi_crossover_flag"] = np.where(
        df["pvi_party"].isin(["R", "D"]) & df["inc_party"].isin(["R", "D"]) & (df["pvi_party"] != df["inc_party"]), 1, 0
    )
    # Swing History Boosted: Sum instead of Mean, higher multipliers
    # 1 Flip = 50pts. 1 Crossover = 30pts.
    # Logic: Proven swing behavior is the strongest predictor of future swing behavior.
    df["score_swing_flip"] = (df["swing_flip_count"].fillna(0) * 50.0).clip(upper=100)
    df["score_swing_crossover"] = (df["swing_crossover_count"].fillna(0) * 30.0).clip(upper=100)
    df["SwingHistory"] = (df["score_swing_flip"] + df["score_swing_crossover"]).clip(upper=100)

    # 2. Protest
    minor_cols = ["2024_libertarian_pct", "2024_green_pct", "2024_independent_pct"]
    df["maverick_minor_pct"] = sum(df[c].fillna(0) for c in minor_cols if c in df.columns)
    
    margins = []
    for y in [2016, 2022, 2024]:
        if f"{y}_republican_pct" in df.columns:
            df[f"margin_{y}"] = (df[f"{y}_republican_pct"] - df[f"{y}_democratic_pct"]).abs()
            margins.append(f"margin_{y}")
    
    df["two_party_margin_hist_avg"] = df[margins].mean(axis=1)
    
    # Updated Maverick Score (Step 738, Refined Step 815):
    # 1. Relaxed Margin Penalty: 25 -> 10. (Margin 0=100, Margin 10=0).
    # 2. Added Third Party Bonus: +2 * Minor Pct (using robust minor_hist_avg).
    base_maverick = np.maximum(0.0, 100.0 - df["two_party_margin_hist_avg"] * 10.0)
    
    # Use minor_hist_avg (calculated in load_maverick_history) which captures ALL non-major votes
    bonus_maverick = (df["minor_hist_avg"].fillna(0) * 2.0)
    
    df["score_maverick"] = (base_maverick + bonus_maverick).clip(upper=100)
    
    df["score_split"] = df["split_rate"] * 100.0

    # 3. Apathy
    df["turnout_rate_2022"] = (df["votes_2022"] / df["Citizen_Voting_Age_Population"]) * 100.0
    df["score_midterm_apathy"] = np.minimum(100.0, np.maximum(0.0, 30.0 + (52.0 - df["turnout_rate_2022"]) * 12.0))
    
    df["turnout_rate_2024"] = (df["votes_2024"] / df["Citizen_Voting_Age_Population"]) * 100.0
    # Presidential turnout is higher, so we set baseline at 65% (vs 52% for Midterm)
    df["score_pres_apathy"] = np.minimum(100.0, np.maximum(0.0, 30.0 + (65.0 - df["turnout_rate_2024"]) * 12.0))
    
    df["score_registration"] = 100.0 - df["state_registration_total"]

    # 4. Demo
    df["score_gen_shift"] = np.minimum(100.0, (df["gen_18_54_share"] / 0.65) * 100.0)
    
    move_share = np.where(df["mob_pop1plus"] > 0, df["mob_diff_house"] / df["mob_pop1plus"], np.nan)
    df["mob_move_share"] = move_share
    df["score_new_resident"] = np.minimum(100.0, move_share * 500.0)
    
    denom = df["Total_Population"].replace(0, np.nan)
    shares = [df[c] / denom for c in ["White_Alone", "Black_or_African_American_Alone", "Hispanic_or_Latino"]]
    shares.append((df["Total_Population"] - sum(df[c].fillna(0) for c in ["White_Alone", "Black_or_African_American_Alone", "Hispanic_or_Latino"])).clip(lower=0) / denom)
    df["race_white_share"], df["race_black_share"], df["race_hispanic_share"], df["race_other_share"] = shares
    df["score_diversity"] = (1.0 - sum(s.pow(2) for s in shares)) * 100.0
    
    denom_p = df["place_total"].replace(0, np.nan)
    p_native = df["place_native_state"] / denom_p
    p_us_other = (df["place_diff_state"].fillna(0) + df["place_pr_island_abroadparents"].fillna(0)) / denom_p
    p_foreign = df["place_foreign"] / denom_p
    df["origin_native_share"], df["origin_us_other_share"], df["origin_foreign_share"] = p_native, p_us_other, p_foreign
    df["score_origin_diversity"] = (1.0 - (p_native.pow(2) + p_us_other.pow(2) + p_foreign.pow(2))) * 100.0

    # Updated Demo with SOGI (Added Jan 2026)
    # Formula: Rate * 20.
    # Logic: 5.0% rate (San Francisco/Seattle levels) = 100 Score.
    #        1.0% rate (Average) = 20 Score.
    # Removing dynamic quantile thresholding for consistency.
    df["score_sogi"] = (df["sogi_score_raw"].fillna(0) * 20.0).clip(0, 100)

    # Aggregates
    def w_avg(cols, weights):
        num = sum(df[c].fillna(0) * w for c, w in zip(cols, weights))
        den = sum(df[c].notna() * w for c, w in zip(cols, weights))
        return num / den.replace(0, np.nan)

    df["Vacuum"] = w_avg(
        ["score_ici", "score_osborn", "score_dropoff", "SwingHistory"],
        [VACUUM_ICI_WEIGHT, VACUUM_OSBORN_WEIGHT, VACUUM_DROPOFF_WEIGHT, VACUUM_SWING_HISTORY_WEIGHT]
    )
    df["Protest"] = w_avg(
        ["score_maverick", "score_split"],
        [PROTEST_MAVERICK_WEIGHT, PROTEST_SPLIT_WEIGHT]
    )
    df["Apathy"] = w_avg(
        ["score_midterm_apathy", "score_pres_apathy", "score_registration"],
        [APATHY_MIDTERM_WEIGHT, APATHY_PRES_WEIGHT, APATHY_REGISTRATION_WEIGHT]
    )
    df["Demo"] = w_avg(
        ["score_gen_shift", "score_new_resident", "score_sogi", "score_diversity", "score_origin_diversity"],
        [DEMO_GEN_SHIFT_WEIGHT, DEMO_NEW_RESIDENT_WEIGHT, DEMO_SOGI_WEIGHT, DEMO_DIVERSITY_WEIGHT, DEMO_ORIGIN_DIVERSITY_WEIGHT]
    )
    # Determine strategic profile for each district
    def determine_profile(row):
        """Determine the strategic profile for a district based on its core scores."""
        vacuum = row.get("Vacuum", 0) or 0
        protest = row.get("Protest", 0) or 0
        apathy = row.get("Apathy", 0) or 0
        demo = row.get("Demo", 0) or 0
        
        # HIERARCHICAL WATERFALL LOGIC (8 Types)
        
        # 1. LAZY GIANT (Hegemony Check)
        ici_score = row.get("score_ici", 0)
        osborn_score = row.get("score_osborn", 0)
        if osborn_score > 75 and ici_score < 40:
            return "lazy_giant"

        # 2. SLEEPING GIANT (Apathy Spike)
        score_midterm = row.get("score_midterm_apathy", 0)
        score_pres = row.get("score_pres_apathy", 0)
        if score_midterm > 80 or score_pres > 80:
            return "sleeping_giant"

        # 3. FREEDOM COALITION (Lifestyle Spike)
        score_sogi = row.get("score_sogi", 0)
        if score_sogi > 80:
            return "freedom_coalition"

        # 4. MAVERICK REBELLION (Protest Spike)
        score_maverick = row.get("score_maverick", 0)
        if score_maverick > 30: # 30 is extremely high for margin/3rd party
            return "maverick_rebellion"

        # 5. CULTURAL WAVE (Migration Spike)
        score_new = row.get("score_new_resident", 0)
        score_origin = row.get("score_origin_diversity", 0)
        if score_new > 70 or score_origin > 70:
            return "cultural_wave"
            
        # 6. UNAWAKENED FUTURE (Apathy + Youth Combination)
        score_gen = row.get("score_gen_shift", 0)
        if apathy > 50 and score_gen > 60:
            return "unawakened_future"

        # 7. VOLATILE SWING (Vacuum/Protest General)
        if ici_score > 50 or vacuum > 55 or protest > 55:
            return "volatile_swing"

        # 8. BALANCED (Fallback)
        return "balanced_general"
    
    df["Profile"] = df.apply(determine_profile, axis=1)
    
    # Compute CIPI with dynamic weights based on profile
    def compute_dynamic_cipi(row):
        profile = PROFILES[row["Profile"]]
        cores = ["Vacuum", "Protest", "Apathy", "Demo"]
        
        # Weighted average with dynamic weights
        num = sum((row[c] or 0) * profile[c] for c in cores)
        den = sum((1 if pd.notna(row[c]) else 0) * profile[c] for c in cores)
        
        return num / den if den > 0 else np.nan
    
    df["CIPI"] = df.apply(compute_dynamic_cipi, axis=1)
    
    # Compute Tier based on CIPI
    def determine_tier(score):
        if pd.isna(score): return 4
        if score >= 55: return 1
        if score >= 45: return 2
        if score >= 35: return 3
        return 4

    df["Tier"] = df["CIPI"].apply(determine_tier)

    return df
