# Project Implementation Review: "Brett's Directives"

This document summarizes the status of the requested features, identifying what was successfully implemented and what was blocked by data/technical limitations.

## 1. Vacuum Core Refinement (The "Osborn" Factor)

### 🟢 "Hegemony" Replacement

- **Request (Brett's Directive)**: Move away from "Hegemony" (pure dominance) and focus on the "Lazy Giant" / "Osborn" dynamic—districts where one party is dominant but lazy/uncontested, creating a vacuum for an independent.
- **Status**: **Fully Implemented**.
- **Methodology**:
  - **New Component**: Added `score_osborn` (Weight 0.5) to the Vacuum Core.
  - **Cook PVI Removed**: Replaced entirely by `score_ici` and `score_osborn`.
  - **Logic**: This specifically targets districts with **High Partisan Lean (>75%)** but **Low Independent Context (<40%)**. It rewards the _absence_ of competition in a stronghold.
  - **Profile integration**: This logic also triggers the **"Lazy Giant"** strategic profile.
  - **The "Paradox" Logic**:
    - We explicitly look for an inverse relationship: **High Osborn Score** (>75) and **Low ICI Score** (<40).
    - _Why?_ Because if ICI is already high, independent candidates are _already_ active (not "Lazy"). The "Lazy Giant" specifically identifies areas where the opportunity is huge but **latent/untapped**.
  - **The "Active Disruption" Scenario (High Osborn + High ICI)**:
    - _Question_: "If ICI is high AND Osborn is high, do we have both opportunities?"
    - _Answer_: **YES.** This bypasses "Lazy Giant" and triggers **"Volatile Swing"**.
    - _Distinction_:
      - **Lazy Giant** = **Latent Opportunity** (Structural weakness, but sleeping).
      - **Volatile Swing** = **Kinetic Opportunity** (Structural weakness + Active Revolt).
    - _Result_: A High Osborn + High ICI district receives a massive CIPI score, correctly identifying it as a top-tier target.

## 2. Demographic Core Improvements

### 🟢 SOGI (Lifestyle Freedom)

- **Request**: Include LGBTQ+ data as a strong proxy for "Lifestyle Liberalism" and non-traditional voting blocks.
- **Status**: **Fully Implemented**.
- **Methodology**:
  - **Data Source**: ACS 1-Year Estimates (Same-Sex Coupled Households).
  - **Logic**: Applied a `x20` scalar (Weight 2.0) to make this a dominant factor in the "Freedom Coalition" score.
  - **Result**: Districts with high lifestyle liberalism are now explicitly flagged.

### 🔴 Migration Origin Flows ("Where are they from?")

- **Request**: Identify the specific origin states of new residents (e.g., "5% from California") to target messaging.
- **Status**: **Not Implemented (Data Limitation)**.
- **The Blocker**:
  - **Granularity**: Census API does not provide "District-to-State" migration flows directly.
  - **Calculation Barrier**: Constructing this requires joining "Tract-to-PUMA" and "Tract-to-CD" spatial files. The required Crosswalk files (2020 Census) were not downloadable via script, and the environment lacks `geopandas` for raw spatial calculation.
- **Current State**: We implemented a "Mobility Score" (`score_new_resident`) which tracks **Volume** of movers, but not their **Source**.

---

## 3. Strategic Profiles & Weighting

### 🟢 Profile Expansion (4 -> 8 Types)

- **Request**: "Increase the profiles" and ensure distinct scenarios (like 100% Apathy) are not missed.
- **Status**: **Fully Implemented**.
- **Methodology**:
  - Expanded the strategic roster to 8 Profiles:
    1.  `volatile_swing` (Open Seat)
    2.  `lazy_giant` (Hegemon)
    3.  `sleeping_giant` (Apathy)
    4.  `freedom_coalition` (SOGI)
    5.  `maverick_rebellion` (3rd Party)
    6.  `unawakened_future` (Youth+Apathy)
    7.  `cultural_wave` (Migration)
    8.  `balanced_general` (Fallback)

### 🟢 Spike Detection Logic

- **Request**: Ensure a district with "100% Apathy" falls into the Apathy profile, even if other scores drag down the average.
- **Status**: **Fully Implemented**.
- **Methodology**: Added a **"Spike Check" Waterfall**.
  - _Logic_: `If Apathy > 80: Force Profile = Sleeping Giant`.
  - This logic overrides the weighted average, ensuring outliers determine strategy.

### 🟢 Dynamic Weighting

- **Request**: Weights must strictly reflect the profile (e.g., Sleeping Giant relies on Apathy).
- **Status**: **Fully Implemented**.
- **Methodology**:
  - Each of the 8 profiles now triggers a custom weighting matrix.
  - **Specialist Profiles** (like Sleeping Giant) allocate **50% of the total CIPI Weight** to their dominant core.

---

## 4. Reporting & Visualization

### 🟢 Technical Appendix & Formatting

- **Request**: Reports must allow for clear reading ("Enters"/Line Braks) and explain the math.
- **Status**: **Fully Implemented**.
- **Methodology**:
  - **Formatting**: Injected strict "Nested List" rules into the prompt to fix PDF readability.
  - **Transparency**: Added a "Technical Methodology Appendix" to every report that cites the specific formula and weights used.
