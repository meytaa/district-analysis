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

VACUUM_PVI_WEIGHT = 1.0
VACUUM_HEGEMONY_WEIGHT = 0.5
VACUUM_DROPOFF_WEIGHT = 1.0
VACUUM_SWING_HISTORY_WEIGHT = 1.0

PROTEST_MAVERICK_WEIGHT = 1.0
PROTEST_SPLIT_WEIGHT = 1.0

APATHY_MIDTERM_WEIGHT = 1.0
APATHY_PRES_WEIGHT = 0.6
APATHY_REGISTRATION_WEIGHT = 0.5

DEMO_GEN_SHIFT_WEIGHT = 1.0
DEMO_NEW_RESIDENT_WEIGHT = 1.0
DEMO_DIVERSITY_WEIGHT = 1.0
DEMO_ORIGIN_DIVERSITY_WEIGHT = 1.0


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
            seq = int(path.stem)
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


def load_state_hegemony(state_abbr):
    sp = pd.read_csv(DATA_DIR / "State Party 2024.csv")
    sp["state_po"] = sp["state"].str.upper().map(state_abbr)
    for c, new in [("current_democratic_house", "dem"), ("current_republican_house", "rep"), ("current_independent_house", "ind")]:
        sp[new] = pd.to_numeric(sp[c], errors="coerce") if c in sp.columns else np.nan
    sp["house_total"] = sp[["dem", "rep", "ind"]].sum(axis=1, min_count=1)
    sp["majority_pct"] = sp[["dem", "rep", "ind"]].max(axis=1) / sp["house_total"] * 100.0
    return sp[["state_po", "majority_pct"]]


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
    older = sum(_get_demo_value(df, "Sex and Age", t) for t in [
        "55 to 59 years", "60 to 64 years", "65 to 74 years", "75 to 84 years", "85 years and over"
    ])
    
    if not pop18 or np.isnan(pop18) or np.isnan(older): return [np.nan, np.nan]
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


def build_master_table():
    turnout, state_abbr, base = load_ballot_and_state_map()
    
    # Load all data sources
    dfs = [
        load_cvap(state_abbr),
        load_party(state_abbr),
        load_pvi_and_pres(),
        load_state_hegemony(state_abbr),
        load_state_registration(),
        load_pvi_history_swing(),
        load_midterm_dropoff_history(),
        load_maverick_history(),
        load_split_ticket_history(state_abbr)
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
    df["score_pvi"] = np.maximum(0.0, 100.0 - df["PVI_value"].abs() * 10.0)
    df["score_hegemony"] = np.where(df["majority_pct"] > 80.0, 100.0, 50.0)
    df["score_dropoff"] = ((df["excess_gap"] / 20.0) * 100.0).clip(0, 100)
    
    df["pvi_crossover_flag"] = np.where(
        df["pvi_party"].isin(["R", "D"]) & df["inc_party"].isin(["R", "D"]) & (df["pvi_party"] != df["inc_party"]), 1, 0
    )
    df["score_swing_flip"] = (df["swing_flip_count"].fillna(0) * 25.0).clip(upper=100)
    df["score_swing_crossover"] = (df["swing_crossover_count"].fillna(0) * 10.0).clip(upper=100)
    df["SwingHistory"] = df[["score_swing_flip", "score_swing_crossover"]].mean(axis=1)

    # 2. Protest
    minor_cols = ["2024_libertarian_pct", "2024_green_pct", "2024_independent_pct"]
    df["maverick_minor_pct"] = sum(df[c].fillna(0) for c in minor_cols if c in df.columns)
    
    margins = []
    for y in [2016, 2022, 2024]:
        if f"{y}_republican_pct" in df.columns:
            df[f"margin_{y}"] = (df[f"{y}_republican_pct"] - df[f"{y}_democratic_pct"]).abs()
            margins.append(f"margin_{y}")
    
    df["two_party_margin_hist_avg"] = df[margins].mean(axis=1)
    df["score_maverick"] = np.maximum(0.0, 100.0 - df["two_party_margin_hist_avg"] * 25.0)
    df["score_split"] = df["split_rate"] * 100.0

    # 3. Apathy
    df["turnout_rate_2022"] = (df["votes_2022"] / df["Citizen_Voting_Age_Population"]) * 100.0
    df["score_midterm_apathy"] = np.minimum(100.0, np.maximum(0.0, 30.0 + (52.0 - df["turnout_rate_2022"]) * 8.0))
    
    df["turnout_rate_2024"] = (df["votes_2024"] / df["Citizen_Voting_Age_Population"]) * 100.0
    df["score_pres_apathy"] = np.minimum(100.0, np.maximum(0.0, 30.0 + (52.0 - df["turnout_rate_2024"]) * 8.0))
    
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

    # Aggregates
    def w_avg(cols, weights):
        num = sum(df[c].fillna(0) * w for c, w in zip(cols, weights))
        den = sum(df[c].notna() * w for c, w in zip(cols, weights))
        return num / den.replace(0, np.nan)

    df["Vacuum"] = w_avg(
        ["score_pvi", "score_hegemony", "score_dropoff", "SwingHistory"],
        [VACUUM_PVI_WEIGHT, VACUUM_HEGEMONY_WEIGHT, VACUUM_DROPOFF_WEIGHT, VACUUM_SWING_HISTORY_WEIGHT]
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
        ["score_gen_shift", "score_new_resident", "score_diversity", "score_origin_diversity"],
        [DEMO_GEN_SHIFT_WEIGHT, DEMO_NEW_RESIDENT_WEIGHT, DEMO_DIVERSITY_WEIGHT, DEMO_ORIGIN_DIVERSITY_WEIGHT]
    )
    df["CIPI"] = w_avg(
        ["Vacuum", "Protest", "Apathy", "Demo"],
        [VACUUM_WEIGHT, PROTEST_WEIGHT, APATHY_WEIGHT, DEMO_WEIGHT]
    )
    
    return df
