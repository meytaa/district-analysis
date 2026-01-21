# CIPI Technical Architecture & Logic Reference (Expanded)

This document serves as the "Master Specification" for the **Comprehensive Independent Potential Index (CIPI)**. It details the mathematical architecture, strategic logic, and weighting systems used to identify independent opportunities in US House Elections.

---

## 1. The Core Philosophy

The CIPI rejects the traditional "Partisan Lean" (Cook PVI) model. Instead, it quantifies **Systemic Vulnerability**.

- **Traditional Model**: "Is this district Red or Blue?"
- **CIPI Model**: "Is this district _functionally broken_? Is it asleep? Is it voting for a system it hates?"

We measure this via 4 vectors (Cores): **Vacuum** (Opportunity), **Protest** (Rejection), **Apathy** (Disengagement), and **Demo** (Future).

---

## 2. Dynamic Weighting Profiles (The "8 Strategies")

The system classifies every district into one of **8 Strategic Profiles**. This classification fundamentally alters how the final score is calculated, rewarding districts that "Specialize" in a specific path to victory.

### The Profile Matrix & Weights

_Weights sum to 1.0 (100%). High weights (>40%) indicate the primary vector of attack._

| Profile Name             | **Vacuum** | **Protest** | **Apathy** | **Demo** | Strategic Narrative                                                                                                                                  |
| :----------------------- | :--------- | :---------- | :--------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`lazy_giant`**         | **45%**    | 25%         | 20%        | 10%      | **The Hegemon**. A district controlled by one party (Osborn > 75) that has grown complacent (Low ICI). Strategy: "Anti-Corruption".                  |
| **`sleeping_giant`**     | 15%        | 10%         | **50%**    | 25%      | **The Non-Voter**. Participation is abysmal. The winner is determined by who stays home. Strategy: "Wake the Giant" (Mobilization).                  |
| **`freedom_coalition`**  | 10%        | 20%         | 20%        | **50%**  | **The Modernist**. High concentration of SOGI/Youth. Values align with Personal Freedom, not binary dogma. Strategy: "Lifestyle Liability".          |
| **`maverick_rebellion`** | 20%        | **50%**     | 15%        | 15%      | **The Protestor**. History of high 3rd party vote and thin margins. Voters are actively searching for an alternative. Strategy: "Break the Duopoly". |
| **`cultural_wave`**      | 15%        | 15%         | 15%        | **55%**  | **The Newcomer**. High migration and diversity. Voters have no generational loyalty to the local establishment. Strategy: "Welcome Neighbors".       |
| **`unawakened_future`**  | 10%        | 10%         | **40%**    | **40%**  | **The Youth Void**. High Apathy + High Youth Share. The future majority is present but silent. Strategy: "Rock the Vote".                            |
| **`volatile_swing`**     | **50%**    | 30%         | 10%        | 10%      | **The Open Seat**. High structural instability (Open Seat or Competitive PVI). Strategy: "Change vs Status Quo".                                     |
| **`balanced_general`**   | 25%        | 25%         | 25%        | 25%      | **The Average**. No distinct features. Requires a broad spectrum Independent campaign.                                                               |

---

## 3. Profile Assignment Logic ("The Waterfall")

Districts are not assigned profiles randomly. We use a **Hierarchical Waterfall** with "Spike Detection".
_Logic priority ensures the most actionable features are captured first._

1.  **Priority 1: Lazy Giant** (`score_osborn > 75` AND `score_ici < 40`)
    - _Why?_ Monopolies are the most distinct structural flaw.
2.  **Priority 2: Sleeping Giant** (`Apathy > 80`)
    - _Why?_ If 100% of the opportunity is Apathy, we must categorize it here even if other scores are low.
3.  **Priority 3: Freedom Coalition** (`SOGI > 80`)
    - _Why?_ A massive cultural base overrides other factors.
4.  **Priority 4: Maverick Rebellion** (`Maverick > 30`)
    - _Why?_ Proven 3rd party activity is rare and valuable.
5.  **Priority 5: Cultural Wave** (`New Resident > 70` OR `Origin > 70`)
6.  **Priority 6: Unawakened Future** (`Apathy > 50` AND `Youth > 60`)
7.  **Priority 7: Volatile Swing** (`ICI > 50` OR `Vacuum > 55`)
8.  **Default**: Balanced General.

---

## 4. Score Formulas & Calibration Constants

### A. VACUUM CORE (Structural Opportunity)

- **`score_ici` (Independent Context Index)**
  - **(REPLACES COOK PVI)**: We removed the traditional Partisan Voting Index.
  - **Formula**: `100 - max(National_Deviation * 3, State_Deviation * 2)`
  - **Why \*3?**: Standard deviations are small (e.g., 10-15%). Multiplying by 3 expands this to a 0-100 scale where >30% deviation (Safe Seat) results in a 0 Score.
- **`score_osborn` (Hegemony Index)**
  - **Formula**: `% State Lower House Seats held by Dominant Party`.
  - **Meaning**: Proxy for local machine strength. 100% = Total Lockout.
- **The ICI/Osborn Paradox Logic (Critical)**:
  - **Lazy Giant (High Osborn / Low ICI)**: The incumbent is dominant, but the independent scene is dead. The opportunity is **Latent**.
  - **Active Disruption / Volatile Swing (High Osborn / High ICI)**: The incumbent is dominant, AND the independent scene is active. The opportunity is **Kinetic** (Best Case Scenario).
- **`score_dropoff` (Midterm Vacuum)**
  - **Formula**: `(Midterm_Dropoff_Rate - 10) * 2.5`
  - **Why \*2.5?**: Dropoff ranges from 10% to 50%. Subtracting 10 (base) and multiplying by 2.5 maps this to a 0-100 score (e.g. 50% dropoff -> Score 100).

### B. PROTEST CORE (Refusal)

- **`score_maverick` (Indecision)**
  - **Formula**: `(100 - Margin * 10) + (3rd_Party_Rate * 2)`
  - **Calibration**: We penalize large margins heavily (*10). We reward proven 3rd party voters (*2).
- **`score_split` (Ticket Splitting)**
  - **Formula**: Direct %.
  - **Meaning**: % of elections where District voted differently for Pres vs House.

### C. APATHY CORE (Disengagement)

- **`score_midterm_apathy`**
  - **Threshold**: If Turnout < 52%, Score increases exponentially.
  - **Spike**: If Turnout < 35%, Score = 100.
- **`score_registration`**
  - **Formula**: `100 - (Registered / CVAP)`
  - **Meaning**: The % of eligible adults who aren't on the rolls.

### D. DEMO CORE (Future Coalition)

- **`score_sogi` (Lifestyle Freedom)**
  - **Formula**: `Same_Sex_Couple_Rate * 20`
  - **Calibration**: A 5% rate is "High" nationally. Multiplying by 20 converts 5% -> Score 100. This ensures this small but critical demographic drives the score.
- **`score_gen_shift`**
  - **Formula**: `Share of Pop (18-44)`.
- **`score_new_resident`**
  - **Formula**: `Share of Pop Moved in Last Year`.
  - **Direct**: Higher mobility = Lower incumbent loyalty.

---

## 5. Why These Constants Matter

Every constant (x2.5, x20, x3) is a **Strategic Opinion** embedded in math:

1.  **SOGI x20**: We believe cultural liberalism is 20x more potent than raw numbers suggest as a signal for independent viability.
2.  **Maverick Threshold (30)**: We believe any district with distinct 3rd party history is a rare "Unicorn" that must be flagged immediately.
3.  **Apathy Weight (50%)**: In a Sleeping Giant district, we believe mobilization is _half the battle_, rendering other factors secondary.
