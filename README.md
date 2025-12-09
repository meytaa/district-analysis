# CIPI District Analysis

This project calculates the **Comprehensive Independent Potential Index (CIPI)** for US Congressional Districts. The CIPI identifies districts with high potential for new Comprehensive Independent based on four key dimensions: **Vacuum**, **Protest**, **Apathy**, and **Demographics**.

## Quick Start

To generate the comprehensive master table containing all raw data, intermediate calculations, and final scores:

```bash
python3 generate_cipi_master.py
```

This will create `output/cipi_comprehensive_master.csv`.

## Project Structure

- `cipi_pipeline.py`: Core logic for loading data, calculating variables, and computing scores.
- `generate_cipi_master.py`: Script to generate the superset CSV with all available data points.
- `generate_district_report.py`: Script to generate AI-powered qualitative reports for districts.
- `reports/`: Directory where generated district reports are saved.
- `all data/`: Directory containing raw input CSVs and Excel files.
- `output/`: Directory where generated CSVs are saved.

## Data Dictionary (Key Columns)

The master table (`cipi_comprehensive_master.csv`) contains over 100 columns. Here are the primary scores:

### 1. Vacuum (Political Competitiveness & Stability)

- `score_pvi`: Competitiveness based on Partisan Voting Index.
- `score_hegemony`: Lack of opposition in state legislature.
- `score_dropoff`: Drop in voter turnout during midterms.
- `SwingHistory`: History of party flipping and ticket splitting.

### 2. Protest (Anti-Establishment Sentiment)

- `score_maverick`: Closeness of elections (lower margins = higher score).
- `score_split`: Rate of split-ticket voting (Pres/House).

### 3. Apathy (Voter Disengagement)

- `score_midterm_apathy`: Low turnout in 2022 midterms.
- `score_pres_apathy`: Low turnout in 2024 presidential election.
- `score_registration`: Low voter registration rates.

### 4. Demo (Demographic Shift)

- `score_gen_shift`: High proportion of population aged 18-54.
- `score_new_resident`: High rate of new residents (mobility).
- `score_diversity`: Racial diversity.
- `score_origin_diversity`: Diversity of birthplaces.

### Aggregates

- `Vacuum`, `Protest`, `Apathy`, `Demo`: Weighted category scores.
- `CIPI`: Final weighted index.
