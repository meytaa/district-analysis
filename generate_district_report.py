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
Your goal is to analyze a district's "Comprehensive Independent Potential Index" (CIPI) to determine if it is ripe for an independent challenge.

The CIPI measures the potential for new Comprehensive Independent based on 4 key dimensions (Cores). 
Below is the logic for each score and how it should be interpreted.

### 1. VACUUM (Weight 0.35)
Measures the lack of entrenched political power or consistent voting patterns.
- **score_ici** (Independent Context Index): **
  - Logic: `100 - max(National_Deviation * 3, State_Deviation * 2)`.
  - Interpretation: Measures *Competitiveness*. High score = Active swing history. Low score = Safe seat.
- **score_osborn** (Osborn Strategy):
  - Logic: Percentage of state house seats held by the dominant party (e.g., 100% control = 100).
  - Interpretation: Measures *Structural Hegemony*. "How dominant is the party machine?"
- **THE OSBORN/ICI PARADOX (CRITICAL ANALYSIS POINT)**:
  - **"Lazy Giant" (High Osborn, Low ICI)**: The incumbent is dominant but untested. The voters are asleep. This is a *Latent Opportunity*.
  - **"Active Disruption" / Volatile Swing (High Osborn, High ICI)**: The machine is dominant, BUT the voters are revolting. This is a *Kinetic Opportunity* (Best of Both Worlds).
- **score_dropoff** (Midterm Dropoff):
  - Logic: `(Average_Midterm_Gap - 10) * 2.5`.
  - Interpretation: High score means massive absolute voter dropoff (~40-50%) in midterms, creating a "Vacuum" of disengaged voters to mobilize.
- **SwingHistory**:
  - Logic: `Sum(Flips * 50, Split_Tickets * 30)`.
  - Interpretation: High score means the district has PROVEN volatility. A score of 100 means it has flipped parties twice or flipped and split-ticketed recently.

### 2. PROTEST (Weight 0.25)
Measures the tendency to vote against the grain or support non-establishment candidates.
- **score_maverick** (Indecision & Protest): 
  - Logic: `(100 - avg_margin * 10) + (Minor_Party_Pct * 2)`.
  - Interpretation: High score signals deep indecision (Close Margins) or active refusal of major parties (High Third Party vote).
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
  - Logic: Share of population 18-44 (Gen Z + Millennials).
  - Interpretation: High score = younger population ("Future Coalition").
- **score_sogi** (SOGI - Lifestyle/Freedom):
  - Logic: Same-Sex Couple Rate * 20. (5% Rate = 100 Score).
  - Interpretation: High score indicates a strong "Lifestyle liberalism" or "Freedom Coalition" constituency. This is the **Dominant Demographic Factor** (High weighting).
- **score_new_resident** (Mobility): 
  - Logic: Share of population that moved in the last year.
  - Interpretation: High score = high transient/new population.
- **score_diversity** (Racial): 
  - Logic: 1 - Herfindahl-Hirschman Index of race.
  - Interpretation: High score = high racial diversity.
- **score_origin_diversity** (Birthplace): 
  - Logic: 1 - HHI of birthplace.
  - Interpretation: High score = diverse mix of locals, transplants, and immigrants.

### 5. STRATEGIC PROFILES (The "Brand")
The district has been assigned one of the following "Strategic Profiles" based on its dominant scores. You MUST explain this profile in the report.
- **`volatile_swing`**: High Vacuum/Protest. Structural instability.
  - *Strategy*: Attack the "Openness" of the seat. Run on competence & change.
- **`lazy_giant`**: High Osborn (Hegemony) + Low ICI (Competitiveness).
  - *Strategy*: Attack the "One-Party Corruption". The incumbent is asleep despite high latent power.
- **`sleeping_giant`**: High Apathy.
  - *Strategy*: Mobilize the non-voters. "Your voice isn't heard."
- **`freedom_coalition`**: High Demo (SOGI/Youth).
  - *Strategy*: Run on "Lifestyle Freedom" and modern values. The "Future Coalition."
- **`maverick_rebellion`**: High Protest (3rd Party/Margins).
  - *Strategy*: "Break the Duopoly." Capitalize on deep dissatisfaction with both parties.
- **`unawakened_future`**: High Apathy AND High Youth.
  - *Strategy*: "Rock the Vote." A massive latent youth vote that needs activation.
- **`cultural_wave`**: High New Residents/Diversity.
  - *Strategy*: "Welcome Neighbors." Unite the new residents who have no loyalty to the old establishment.
- **`balanced_general`**: No dominant core.
  - *Strategy*: Run a broad, moderate Independent campaign.

---

**YOUR TASK:**
Analyze the provided data for the target district and provide a **Comprehensive Strategic Report**.

**FORMATTING RULES (CRITICAL):**
1.  **DATA TABLES**: Always use standard Markdown Table syntax (e.g., `| Score | Value |`). Do NOT use list format for data tables.
2.  **ANALYSIS SECTIONS**: Use **clean vertical spacing**.
    *   Do NOT inline analysis (e.g., `* Meaning: X * Implication: Y`).
    *   Instead, use **Nested Lists** or **New Lines**:
        *   **Meaning**: ...
        *   **Implication**: ...
3.  **SEPARATION**: Ensure there is an empty line between every major point.

**STRICT OUTPUT FORMAT:**

# [Dynamic Professional Title]
(Create a compelling, professional title. Example: "Target Tier 1: The Sleeping Giant Opportunity in Texas-04")

## Executive Summary
**Strategic Profile**: [Profile Name]
**Strategy Definition**: [Explain what this profile means and why it fits this district based on the data.]
**Verdict**: [High-level assessment of viability.]
**Opportunity Elaboration**: [Deep dive into the specific opportunity. Explain the "Why Now".]

---

## Phase 1: Core Analysis (The Deep Dive)
For each Core (Vacuum, Protest, Apathy, Demo), provide:
1.  **Data Table**: Scores and raw data.
2.  **Detailed Analysis**: For EVERY score, you MUST use the following nested bullet format (Do not clump text):
    *   **[Score Name]**: [Value]
        *   **Meaning**: [Explanation of what the score measures]
        *   **Implication**: [What this means for the campaign strategy]
    *(Ensure there are new lines between each score)*
3.  **Synthesis**: How do these scores combine to create a vulnerability?

### 1. Vacuum Core
[Content]

### 2. Protest Core
[Content]

### 3. Apathy Core
[Content]

### 4. Demographic Core
[Content]

---

## Phase 2: Strategic SWOT Analysis
**Deeply analyze specific data points.** Do not use generic text.
*   **Strengths (Independent Assets)**: (e.g., "The 40% midterm dropoff gives us a massive pool of 200k voters to activate...")
*   **Weaknesses (Incumbent Vulnerabilities)**: (e.g., "The incumbent's party has 100% control (Osborn 100) but only won by 5 points...")
*   **Opportunities (Specific Targets)**: (e.g., "Target the High SOGI neighborhoods using a Personal Freedom alignment...")
*   **Threats (Systemic Barriers)**: (e.g., "Low State-Level deviation suggests a rigid partisan lock...")

---

## Phase 3: Strategic Roadmap
[Detailed actionable steps for Grassroots, Messaging, and Coalition building.]

---

## Phase 4: Final Recommendation & Conclusion
[Final verdict.]

---

## Phase 5: Technical Methodology Appendix
(Briefly summarize the model architecture for the reader)
1.  **Strategic Profile Used**: Re-state the profile and its specific weights (e.g., "Sleeping Giant: Apathy 50%, Demo 25%...").
2.  **Score Formulas**: Briefly explain the key formulas derived from the CIPI Technical Architecture (e.g., "ICI = 100 - Deviations", "SOGI = Rate * 20").
3.  **Why This Matters**: Explain that these scores are weighted to identify *systemic* vulnerability, not just partisan lean.

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
    
    # Exclude legacy columns to prevent confusion
    legacy_cols = ["score_pvi", "score_hegemony", "score_incumbent", "majority_pct", "excess_gap"]
    
    clean_data = {
        k: v for k, v in data.items() 
        if pd.notna(v) and k not in legacy_cols
    }
    return clean_data

def get_report_content(district_code, api_key=None, dry_run=False, chart_image_path=None):
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
        
        # Inject chart if provided
        if chart_image_path:
            # We insert it after the title (Line 1 usually) or just prepend
            # Assuming report starts with "# CIPI Strategic Deep Dive: ..."
            # We'll place it after the header or in Executive Summary?
            # Safest is to append to the end of the Executive Summary or just prepend.
            # User wants it in PDF.
            # Let's insert it right after the main H1 title.
            lines = report.split('\n', 1)
            if len(lines) > 0 and lines[0].startswith('# '):
                report = f"{lines[0]}\n\n![Score Breakdown]({chart_image_path})\n\n{lines[1]}"
            else:
                report = f"![Score Breakdown]({chart_image_path})\n\n{report}"
                
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
