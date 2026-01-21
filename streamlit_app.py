import streamlit as st
import pandas as pd
import os
import markdown
from xhtml2pdf import pisa
from io import BytesIO
from dotenv import load_dotenv
from cipi_pipeline import build_master_table, compute_scores, PROFILES, PROFILE_THRESHOLDS
import plotly.graph_objects as go
from generate_district_report import get_report_content

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="CIPI District Analysis",
    page_icon="assets/ic-logo-black.png",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .report-container {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

def create_pdf(markdown_content):
    # Convert Markdown to HTML with tables extension
    html_content = markdown.markdown(markdown_content, extensions=['tables'])
    
    # Add styling
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, sans-serif; font-size: 12px; }}
            h1 {{ color: #333333; }}
            h2 {{ color: #444444; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            h3 {{ color: #666666; }}
            p {{ line-height: 1.5; }}
            
            /* Table Styling */
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f4f4f4; color: #333; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #fafafa; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(styled_html, dest=pdf_buffer)
    
    if pisa_status.err:
        return None
    
    return pdf_buffer.getvalue()

# @st.cache_data (Disabled to force refresh during development)
def load_data():
    """Builds master table and computes scores. Cache invalidated for Osborn Strategy."""
    with st.spinner("Loading and processing data..."):
        master = build_master_table()
        scored = compute_scores(master)
    return scored

def check_login():
    """Simple login system using Streamlit session state and secrets."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if st.session_state.authenticated:
        return True
        
    # Login Page Layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.image("assets/logo.svg", width=300)
        st.markdown("### Access Restricted")
        st.markdown("Please log in to access the District Analysis Dashboard.")
    
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                    # Check against secrets
                authenticated = False
                
                # 1. Multi-user support (preferred)
                if "passwords" in st.secrets:
                    if username in st.secrets["passwords"] and password == st.secrets["passwords"][username]:
                        authenticated = True
                
                # 2. Single-user fallback (legacy)
                elif "credentials" in st.secrets:
                    if (username == st.secrets["credentials"]["username"] and 
                        password == st.secrets["credentials"]["password"]):
                        authenticated = True
                
                # 3. Default fallback (only if no secrets configured)
                elif username == "admin" and password == "admin":
                     authenticated = True

                if authenticated:
                    st.session_state.authenticated = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
                
    return False

def page_rankings():
    st.image("assets/logo.svg", width=400)
    st.title("District Rankings")
    st.markdown("Select a row to generate a strategic report.")
    
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Configure columns
    display_cols = [
        "district_code", "State", "CIPI", "Vacuum", "Protest", "Apathy", "Demo"
    ]
    valid_cols = [c for c in display_cols if c in df.columns]
    
    # Prepare Dataframe
    df_display = df[valid_cols].set_index("district_code").sort_values("CIPI", ascending=False)
    
    # Styling Function
    def highlight_tiers(row):
        score = row['CIPI']
        if score >= 65: color = 'rgba(255, 215, 0, 0.15)' # Gold Tint
        elif score >= 60: color = 'rgba(192, 192, 192, 0.15)' # Silver Tint
        elif score >= 50: color = 'rgba(205, 127, 50, 0.15)' # Bronze Tint
        else: return [''] * len(row)
        return [f'background-color: {color}'] * len(row)

    # Apply Style
    styled_df = df_display.style.apply(highlight_tiers, axis=1).format("{:.1f}", subset=["CIPI", "Vacuum", "Protest", "Apathy", "Demo"])
    
    # Display Dataframe
    event = st.dataframe(
        styled_df,
        use_container_width=True,
        height=800,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Handle Selection
    if len(event.selection.rows) > 0:
        selected_row_index = event.selection.rows[0]
        displayed_df = df[valid_cols].set_index("district_code")
        target_code = displayed_df.index[selected_row_index]
        
        # Set target and redirect
        st.session_state.target_district = target_code
        st.switch_page(st.session_state.pages_map["Deep Dive"])


def page_deep_dive():
    st.image("assets/logo.svg", width=400)
    st.title("Strategic Deep Dive")
    st.markdown("Generate comprehensive strategic reports based on the **Comprehensive Independent Potential Index (CIPI)**.")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar controls
    with st.sidebar:
        st.header("Selection")
        
        # Consoliated Target Logic
        if "target_district" in st.session_state and st.session_state.target_district:
            target = st.session_state.target_district
            row = df[df["district_code"] == target]
            if not row.empty:
                # Set State
                target_state = row.iloc[0]["State"]
                st.session_state["deep_dive_state_selector"] = target_state
                
                # Set District Label
                # We need to reconstruct the label: "District X (CODE)"
                # But creating the exact label requires the districts_map logic.
                # So we defer district setting slightly? 
                # No, we can pre-calculate or just let the map logic below handle it?
                # Actually, simpler: Set State here. Let the map build. Then set District Key.
                # But we act *before* the widgets are rendered.
                
                # Check persistence of state first.
                pass

        # 1. Select State
        states = sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("Select State", states, key="deep_dive_state_selector")
        
        # 2. Select District
        state_data = df[df['State'] == selected_state]
        
        districts_map = {
            f"District {row['District']} ({row['district_code']})": row['district_code']
            for _, row in state_data.iterrows()
            if pd.notna(row['district_code'])
        }
        
        # Calculate District Index if target needs to be set
        dist_keys = list(districts_map.keys())
        
        # Initialize selector state if target provided
        if "target_district" in st.session_state and st.session_state.target_district:
            target = st.session_state.target_district
            # Find the label that maps to this code
            for key, val in districts_map.items():
                if val == target:
                    st.session_state["deep_dive_district_selector"] = key
                    break
            # Clear target after consuming
            del st.session_state.target_district

        selected_district_label = st.selectbox(
            "Select District", 
            options=dist_keys, 
            key="deep_dive_district_selector"
        )
        selected_district_code = districts_map[selected_district_label] if selected_district_label else None
        
        st.divider()
        
        # API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
            elif "passwords" in st.secrets and "OPENAI_API_KEY" in st.secrets["passwords"]:
                api_key = st.secrets["passwords"]["OPENAI_API_KEY"]
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here if not set in .env")
        
        generate_btn = st.button("Generate Strategic Report", type="primary")

    # Main Content
    if selected_district_code:
        # Metrics
        district_row = df[df['district_code'] == selected_district_code].iloc[0]
        st.subheader(f"District Overview: {selected_district_code}")
        
        cols = st.columns(5)
        cols[0].metric("CIPI Score", f"{district_row.get('CIPI', 0):.1f}")
        cols[1].metric("Vacuum", f"{district_row.get('Vacuum', 0):.1f}")
        cols[2].metric("Protest", f"{district_row.get('Protest', 0):.1f}")
        cols[3].metric("Apathy", f"{district_row.get('Apathy', 0):.1f}")
        cols[4].metric("Demo", f"{district_row.get('Demo', 0):.1f}")
        
        # Calculate Tier
        cipi_score = district_row.get('CIPI', 0) or 0
        if cipi_score >= 65:
            tier = 1
            tier_color = "#FFD700"  # Gold
            tier_desc = "Top Target (Elite)"
        elif cipi_score >= 60:
            tier = 2
            tier_color = "#C0C0C0"  # Silver
            tier_desc = "Strong Potential"
        elif cipi_score >= 50:
            tier = 3
            tier_color = "#CD7F32"  # Bronze
            tier_desc = "Watch List"
        else:
            tier = 4
            tier_color = "#95A5A6"  # Gray
            tier_desc = "Lower Priority (Ignored)"
        
        # Display Profile and Weights
        profile_name = district_row.get('Profile', 'balanced_general')
        profile_display = profile_name.replace('_', ' ').title()
        weights = PROFILES.get(profile_name, PROFILES['balanced_general'])
        
        # Profile badge with weights
        profile_colors = {
            'volatile_swing': '#FF6B6B',
            'lazy_giant': '#9B59B6', # Purple
            'sleeping_giant': '#4ECDC4', 
            'freedom_coalition': '#45B7D1',
            'maverick_rebellion': '#C0392B', # Dark Red
            'unawakened_future': '#16A085',  # Teal/Green
            'cultural_wave': '#E67E22',      # Orange
            'balanced_general': '#95A5A6'
        }
        badge_color = profile_colors.get(profile_name, '#95A5A6')
        
        # Two-column layout for Tier and Profile
        tier_col, profile_col = st.columns(2)
        
        with tier_col:
            st.markdown(f"""
            <div style="margin-top: 10px; padding: 10px; background-color: {tier_color}20; border-left: 4px solid {tier_color}; border-radius: 4px;">
                <span style="font-weight: bold; font-size: 1.2em; color: {tier_color};">Tier {tier}</span>
                <span style="font-size: 0.9em; color: #666; margin-left: 10px;">{tier_desc}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with profile_col:
            st.markdown(f"""
            <div style="margin-top: 10px; padding: 10px; background-color: {badge_color}20; border-left: 4px solid {badge_color}; border-radius: 4px;">
                <span style="font-weight: bold; color: {badge_color};">Profile: {profile_display}</span>
                <br>
                <span style="font-size: 0.8em; color: #666;">
                    V:{int(weights['Vacuum']*100)}% · P:{int(weights['Protest']*100)}% · A:{int(weights['Apathy']*100)}% · D:{int(weights['Demo']*100)}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # Score Visualization
        st.markdown("##### Score Breakdown")
        
        # Define sub-scores for each core group
        score_groups = {
            "Vacuum": {
                "scores": ["score_ici", "score_osborn", "score_dropoff", "SwingHistory"],
                "labels": ["ICI", "Osborn", "Dropoff", "Swing History"],
                "color": "#3498db"
            },
            "Protest": {
                "scores": ["score_maverick", "score_split"],
                "labels": ["Maverick", "Split Ticket"],
                "color": "#e74c3c"
            },
            "Apathy": {
                "scores": ["score_midterm_apathy", "score_pres_apathy", "score_registration"],
                "labels": ["Midterm", "Presidential", "Registration"],
                "color": "#2ecc71"
            },
            "Demo": {
                "scores": ["score_gen_shift", "score_new_resident", "score_sogi", "score_diversity", "score_origin_diversity"],
                "labels": ["Gen Shift", "New Resident", "SOGI (Lifestyle)", "Diversity", "Origin Div"],
                "color": "#9b59b6"
            }
        }
        
        cipi_score = district_row.get('CIPI', 0) or 0
        
        # Build x-axis labels and values for grouped bar chart
        x_labels = []
        y_values = []
        colors = []
        
        for core, config in score_groups.items():
            for score_col, label in zip(config["scores"], config["labels"]):
                x_labels.append(label)
                val = district_row.get(score_col, 0)
                y_values.append(0 if pd.isna(val) else val)
                colors.append(config["color"])
        
        # Create figure
        fig = go.Figure()
        
        # Sub-score bars
        fig.add_trace(go.Bar(
            x=x_labels,
            y=y_values,
            marker_color=colors,
            text=[f'{v:.0f}' for v in y_values],
            textposition='outside',
            name='Sub-scores'
        ))
        
        # Calculate center position for each group to place line chart points
        group_centers = []
        group_scores = []
        group_colors = []
        bar_idx = 0
        for core, config in score_groups.items():
            num_bars = len(config["scores"])
            center_idx = bar_idx + (num_bars - 1) / 2  # Center of the group
            group_centers.append(x_labels[bar_idx + num_bars // 2])  # Use middle bar's label
            group_scores.append(district_row.get(core, 0) or 0)
            group_colors.append(config["color"])
            bar_idx += num_bars
        
        # Add line chart connecting the 4 group scores
        fig.add_trace(go.Scatter(
            x=group_centers,
            y=group_scores,
            mode='lines+markers+text',
            line=dict(color='#333333', width=3),
            marker=dict(size=12, color=group_colors, line=dict(width=2, color='white')),
            text=[f'{s:.1f}' for s in group_scores],
            textposition='top center',
            textfont=dict(size=12, color='#333333'),
            showlegend=False,
            name='Group Scores'
        ))
        
        # CIPI horizontal line (prominent)
        fig.add_hline(
            y=cipi_score, 
            line_dash="dash", 
            line_color="#f39c12",
            line_width=3,
            annotation_text=f"CIPI: {cipi_score:.1f}",
            annotation_position="right",
            annotation_font=dict(size=14, color="#f39c12")
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=60, t=30, b=120),
            showlegend=False,
            yaxis=dict(range=[0, 105], title="Score"),
            xaxis=dict(title="", tickangle=-90)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add legend for core colors
        legend_html = """
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 0.85em; margin-top: -10px;">
            <span><span style="color: #3498db;">●</span> Vacuum</span>
            <span><span style="color: #e74c3c;">●</span> Protest</span>
            <span><span style="color: #2ecc71;">●</span> Apathy</span>
            <span><span style="color: #9b59b6;">●</span> Demo</span>
            <span><span style="color: #f39c12;">- -</span> CIPI</span>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        
        st.divider()

        # Report Generation Logic
        if generate_btn:
            if not api_key:
                st.error("Please provide an OpenAI API Key.")
            else:
                with st.spinner(f"Analyzing {selected_district_code}... This may take a minute."):
                    # Save chart image for PDF inclusion
                    os.makedirs("reports", exist_ok=True)
                    chart_path = os.path.abspath(f"reports/scores_{selected_district_code}.png")
                    try:
                        # Use scale=2 for better quality
                        fig.write_image(chart_path, scale=2)
                    except Exception as e:
                        st.warning(f"Could not save chart image for PDF: {e}")
                        chart_path = None
                        
                    report_content, error = get_report_content(selected_district_code, api_key=api_key, chart_image_path=chart_path)
                    if error:
                        st.error(error)
                    else:
                        # Save to session state to persist across reruns (downloads)
                        st.session_state['report_content'] = report_content
                        st.session_state['report_district'] = selected_district_code
                        st.success(f"Report generated for {selected_district_code}!")

        # Display Report (Persistent)
        if st.session_state.get('report_district') == selected_district_code and st.session_state.get('report_content'):
            report_content = st.session_state['report_content']
            with st.container():
                st.markdown(report_content)
                st.markdown("---")
                
                # Prepare downloads
                # Generate PDF only when needed (or could cache this too, but it's fast enough usually)
                pdf_data = create_pdf(report_content)
                
                d_col1, d_col2 = st.columns(2)
                
                with d_col1:
                    st.download_button(
                        label="Download MD",
                        data=report_content,
                        file_name=f"cipi_report_{selected_district_code}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with d_col2:
                    if pdf_data:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_data,
                            file_name=f"cipi_report_{selected_district_code}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.warning("PDF generation failed.")
        elif not generate_btn:
            st.info("Click 'Generate Strategic Report' to create a deep-dive analysis using AI.")


def main():
    if check_login():
        # Define Pages
        deep_dive = st.Page(page_deep_dive, title="Strategic Deep Dive")
        rankings = st.Page(page_rankings, title="District Rankings")
        
        # Store in session state for cross-page navigation
        st.session_state.pages_map = {
            "Deep Dive": deep_dive,
            "Rankings": rankings
        }

        # Setup Navigation
        pg = st.navigation({
            "Menu": [deep_dive, rankings]
        })

        pg.run()

        # Logout Button in Sidebar (displayed below navigation AND page content)
        with st.sidebar:
            st.divider()
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.rerun()

if __name__ == "__main__":
    main()
