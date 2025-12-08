import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from cipi_pipeline import build_master_table, compute_scores
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

@st.cache_data
def load_data():
    """Builds master table and computes scores, cached for performance."""
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
    
    # Display Dataframe
    event = st.dataframe(
        df[valid_cols].set_index("district_code"),
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
    st.markdown("Generate comprehensive strategic reports based on the **Civic Infrastructure Potential Index (CIPI)**.")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Determine defaults
    default_state_ix = 0
    default_dist_ix = 0
    
    # Logic to pre-fill if coming from explorer
    if "target_district" in st.session_state and st.session_state.target_district:
        target = st.session_state.target_district
        row = df[df["district_code"] == target]
        if not row.empty:
            target_state = row.iloc[0]["State"]
            states_list = sorted(df['State'].unique().tolist())
            if target_state in states_list:
                default_state_ix = states_list.index(target_state)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Selection")
        
        # 1. Select State
        states = sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("Select State", states, index=default_state_ix)
        
        # 2. Select District
        state_data = df[df['State'] == selected_state]
        
        districts_map = {
            f"District {row['District']} ({row['district_code']})": row['district_code']
            for _, row in state_data.iterrows()
            if pd.notna(row['district_code'])
        }
        
        # Calculate District Index if target needs to be set
        dist_keys = list(districts_map.keys())
        if "target_district" in st.session_state and st.session_state.target_district:
                target = st.session_state.target_district
                for i, key in enumerate(dist_keys):
                    if districts_map[key] == target:
                        default_dist_ix = i
                        break
                # Clear target
                del st.session_state.target_district

        selected_district_label = st.selectbox("Select District", options=dist_keys, index=default_dist_ix)
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
        
        st.divider()

        # Report Logic
        if generate_btn:
            if not api_key:
                st.error("Please provide an OpenAI API Key.")
            else:
                with st.spinner(f"Analyzing {selected_district_code}... This may take a minute."):
                    report_content, error = get_report_content(selected_district_code, api_key=api_key)
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Report generated for {selected_district_code}!")
                        with st.container():
                            st.markdown(report_content)
                            st.markdown("---")
                            st.download_button(
                                label="Download Report as Markdown",
                                data=report_content,
                                file_name=f"cipi_report_{selected_district_code}.md",
                                mime="text/markdown"
                            )
        else:
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
