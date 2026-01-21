from pathlib import Path
from cipi_pipeline import build_master_table, compute_scores, OUTPUT_DIR

def generate_comprehensive_master():
    print("Building master table...")
    master = build_master_table()
    
    print("Computing scores...")
    scored = compute_scores(master)
    
    # Column definitions
    id_cols = ["district_code", "State", "District", "state_po", "state_key"]
    
    vacuum_cols = [
        # ICI (Independent Context Index) - replaces Cook PVI
        "ici_penalty", "ici_context", "district_margin", "state_margin",
        "national_deviation", "state_deviation", "score_ici",
        # Legacy PVI (still loaded for historical swing analysis)
        "PVI_value", "pvi_party", "inc_party", "pvi_crossover_flag",
        # Other Vacuum sub-scores
        "score_osborn",
        "votes_2024", "votes_2022", "gap_2024", "avg_gap_hist", "excess_gap", "score_dropoff",
        "swing_flip_count", "score_swing_flip", "swing_crossover_count", "score_swing_crossover", "SwingHistory"
    ]
    
    protest_cols = [
        "2024_republican_pct", "2024_democratic_pct", "2024_libertarian_pct", "2024_green_pct", "2024_independent_pct",
        "maverick_minor_pct", "minor_2020", "minor_2022", "minor_2024", "minor_hist_avg",
        "margin_2016", "margin_2022", "margin_2024", "two_party_margin_hist_avg", "score_maverick",
        "split_2012", "split_2016", "split_rate", "split_years", "score_split"
    ]
    
    apathy_cols = [
        "Citizen_Voting_Age_Population", "turnout_rate_2022", "score_midterm_apathy",
        "turnout_rate_2024", "score_pres_apathy",
        "state_registration_total", "score_registration"
    ]
    
    demo_cols = [
        "gen_18_54_share", "gen_55_plus_share", "score_gen_shift",
        "mob_pop1plus", "mob_same_house", "mob_diff_house", "mob_diff_state", "mob_abroad", "mob_move_share", "score_new_resident",
        "Total_Population", "White_Alone", "Black_or_African_American_Alone", "Hispanic_or_Latino",
        "race_white_share", "race_black_share", "race_hispanic_share", "race_other_share", "score_diversity",
        "place_total", "place_native_state", "place_diff_state", "place_pr_island_abroadparents", "place_foreign",
        "origin_native_share", "origin_us_other_share", "origin_foreign_share", "score_origin_diversity"
    ]
    
    agg_cols = ["Vacuum", "Protest", "Apathy", "Demo", "CIPI", "Profile"]
    
    all_cols = id_cols + vacuum_cols + protest_cols + apathy_cols + demo_cols + agg_cols
    final_cols = [c for c in all_cols if c in scored.columns]
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "cipi_comprehensive_master.csv"
    print(f"Saving comprehensive master table to {output_path}...")
    scored[final_cols].to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    generate_comprehensive_master()
