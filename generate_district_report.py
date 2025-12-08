import os
import sys
import json
import argparse
import pandas as pd
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from cipi_pipeline import build_master_table, compute_scores

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """
You are an expert political strategist specializing in independent congressional campaigns. 
Your goal is to analyze a district's "Comprehensive Infrastructure Potential Index" (CIPI) to determine if it is ripe for an independent challenge.

The CIPI measures the potential for new Comprehensive infrastructure based on 4 key dimensions (Cores). 
Below is the logic for each score and how it should be interpreted.

### 1. VACUUM (Weight 0.35)
Measures the lack of entrenched political power or consistent voting patterns.
- **score_pvi** (Partisan Voting Index): Competitiveness. 
  - Logic: `max(0, 100 - |PVI_value| * 10)`. 
  - Interpretation: 100 is perfectly even. 0 is a safe seat (R+10 or D+10).
- **score_hegemony** (State Control): 
  - Logic: 100 if one party holds >80% of state house seats, else 50.
  - Interpretation: High score means single-party dominance in the state, suggesting a vacuum of opposition.
- **score_dropoff** (Midterm Dropoff): 
  - Logic: Measures the "excess gap" between presidential and midterm turnout compared to history.
  - Interpretation: High score means voters significantly disengage during midterms, creating an opening.
- **SwingHistory**: 
  - Logic: Average of party flipping frequency and ticket-splitting frequency.
  - Interpretation: High score means the district is volatile and not loyal to one party.

### 2. PROTEST (Weight 0.25)
Measures the tendency to vote against the grain or support non-establishment candidates.
- **score_maverick** (Close Margins): 
  - Logic: `max(0, 100 - avg_margin * 25)`.
  - Interpretation: High score means elections are consistently close (low margins).
- **score_split** (Split Ticket): 
  - Logic: Percentage of recent elections where the district voted for different parties for President and House.
  - Interpretation: High score means voters are comfortable splitting their ticket.

### 3. APATHY (Weight 0.20)
Measures low voter engagement.
- **score_midterm_apathy**: 
  - Logic: Increases as 2022 turnout drops below 52%.
  - Interpretation: High score = very low midterm turnout.
- **score_pres_apathy**: 
  - Logic: Increases as 2024 turnout drops below 52%.
  - Interpretation: High score = very low presidential turnout.
- **score_registration**: 
  - Logic: `100 - registration_rate`.
  - Interpretation: High score = many eligible voters are unregistered.

### 4. DEMO (Weight 0.20)
Measures demographic characteristics correlating with political change.
- **score_gen_shift** (Youth): 
  - Logic: Share of population 18-54.
  - Interpretation: High score = younger population.
- **score_new_resident** (Mobility): 
  - Logic: Share of population that moved in the last year.
  - Interpretation: High score = high transient/new population.
- **score_diversity** (Racial): 
  - Logic: 1 - Herfindahl-Hirschman Index of race.
  - Interpretation: High score = high racial diversity.
- **score_origin_diversity** (Birthplace): 
  - Logic: 1 - HHI of birthplace.
  - Interpretation: High score = diverse mix of locals, transplants, and immigrants.

### AGGREGATES
- **Vacuum**, **Protest**, **Apathy**, **Demo**: Weighted averages of the above.
- **CIPI**: Final weighted index of the four categories.

---

**YOUR TASK:**
Analyze the provided data for the target district and provide a **Comprehensive Strategic Report**.

**STRICT OUTPUT FORMAT:**

# CIPI Strategic Deep Dive: [District Code]

## Executive Summary
[High-level assessment of the Independent Opportunity. Is this district a good target? Why?]

---

## Phase 1: Core Analysis (The Deep Dive)
For each Core (Vacuum, Protest, Apathy, Demo), you must provide:
1.  **Data Table**: A Markdown table of scores and raw data.
2.  **Sub-Score Analysis**: For EACH score in the core, you MUST provide:
    *   **Definition**: A one-sentence explanation of what this score measures (based on the provided logic/interpretation).
    *   **Analysis**: Explain the "Why" by citing the specific raw data values.
    *   *Example*: 
        *   **Score PVI (Definition)**: Measures the competitiveness of the district based on partisan voting trends.
        *   **Analysis**: Score is 0 because the district is D+25, indicating a safe Democratic seat with no competitive vacuum.
3.  **Core Integration**: Synthesize the scores to explain the specific opportunity on this Core. What is the narrative here?

### 1. Vacuum Core
[Table]
[Sub-Score Analysis]
[Core Integration & Opportunity]

### 2. Protest Core
[Table]
[Sub-Score Analysis]
[Core Integration & Opportunity]

### 3. Apathy Core
[Table]
[Sub-Score Analysis]
[Core Integration & Opportunity]

### 4. Demographic Core
[Table]
[Sub-Score Analysis]
[Core Integration & Opportunity]

---

## Phase 2: Strategic SWOT Analysis
Based on the data above, identify:
*   **Strengths (Independent Assets)**: What factors favor an independent (e.g., high apathy, volatile history)?
*   **Weaknesses (Incumbent Vulnerabilities)**: Where is the establishment weak?
*   **Opportunities (Path to Victory)**: Specific demographics or voter blocks to target.
*   **Threats (Systemic Barriers)**: Safe seats, straight-ticket voting, etc.

---

## Phase 3: Strategic Roadmap
How should an independent candidate use these potentials at different levels?
*   **Grassroots Level**: Who to mobilize? (e.g., "Target the 40% unregistered voters").
*   **Messaging Level**: What is the narrative? (e.g., "Time for a change" vs "Your voice isn't heard").
*   **Coalition Level**: Who to unite? (e.g., "Unite the youth and new residents").

---

## Phase 4: Final Recommendation & Conclusion
[Final verdict on the viability of an independent run and the key to success.]

---

## Comprehensive Data Appendix
[Categorized list of ALL raw data points provided in the JSON input.]
"""

def get_district_data(district_code):
    print(f"Loading data for {district_code}...")
    # Re-run pipeline to ensure fresh data (in memory)
    master = build_master_table()
    scored = compute_scores(master)
    
    # Filter for district
    district_row = scored[scored["district_code"] == district_code]
    
    if district_row.empty:
        return None
    
    # Convert to dictionary and drop NaNs for cleaner JSON
    data = district_row.iloc[0].to_dict()
    clean_data = {k: v for k, v in data.items() if pd.notna(v)}
    return clean_data

def get_report_content(district_code, api_key=None, dry_run=False):
    data = get_district_data(district_code)
    if not data:
        return None, f"Error: District {district_code} not found."

    # Prepare the user message with the data
    user_message = f"Here is the data for District {district_code}:\n\n{json.dumps(data, indent=2)}"

    if dry_run:
        output = []
        output.append("="*80)
        output.append("DRY RUN MODE: Printing prompt and data without calling API.")
        output.append("="*80 + "\n")
        output.append("--- SYSTEM PROMPT ---")
        output.append(SYSTEM_PROMPT)
        output.append("\n--- USER MESSAGE ---")
        output.append(user_message)
        return None, "\n".join(output)

    if not api_key:
        return None, "WARNING: OPENAI_API_KEY not found."

    print(f"Querying OpenAI for {district_code} report...")
    try:
        http_client = httpx.Client()
        client = OpenAI(api_key=api_key, http_client=http_client)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
        )
        report = response.choices[0].message.content
        return report, None
        
    except Exception as e:
        return None, f"Error calling OpenAI API: {e}"

def generate_report(district_code, dry_run=False):
    api_key = os.getenv("OPENAI_API_KEY")
    report, error_message = get_report_content(district_code, api_key=api_key, dry_run=dry_run)
    
    if error_message:
        print(error_message)
        if dry_run: 
             return # In dry run, error_message actually contains the content we want to print
        if not report: # Real error
             return

    if report:
        print("\n" + "="*80)
        print(f"CIPI INDEPENDENT OPPORTUNITY REPORT: {district_code}")
        print("="*80 + "\n")
        print(report)
        
        # Save to file
        filename = f"reports/report_{district_code}.md"
        os.makedirs("reports", exist_ok=True)
        with open(filename, "w") as f:
            f.write(report)
        print(f"\nReport saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIPI District Report")
    parser.add_argument("district_code", help="District Code (e.g., NY-14, CA-12)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt and data without calling API")
    args = parser.parse_args()
    
    generate_report(args.district_code, dry_run=args.dry_run)
