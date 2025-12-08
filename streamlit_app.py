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

def run_dashboard():
    st.image("assets/logo.svg", width=600)
    st.title("CIPI District Strategic Analysis")
    st.markdown("Generate comprehensive strategic reports for congressional districts based on the **Civic Infrastructure Potential Index (CIPI)**.")

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar for controls
    with st.sidebar:
        st.header("District Selection")
        
        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
            
        st.divider()
        
        # 1. Select State
        # Extract unique states from the dataframe
        # Assuming 'state' column exists or we need to derive it from 'state_po' or 'State'
        # Looking at cipi_pipeline, 'State' and 'state_po' are available.
        
        states = sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("Select State", states)
        
        # 2. Select District
        # Filter for the selected state
        state_data = df[df['State'] == selected_state]
        
        # Get district codes or numbers
        # We want to show district numbers but use district_code for the generation
        # logic in generate_district_report uses 'district_code' like 'NY-14'
        
        # Create a display list (District Number) -> Value list (District Code)
        # Assuming district_code format is "STATE-DISTRICT" e.g. "NY-14"
        
        districts_map = {
            f"District {row['District']} ({row['district_code']})": row['district_code']
            for _, row in state_data.iterrows()
            if pd.notna(row['district_code'])
        }
        
        selected_district_label = st.selectbox("Select District", options=districts_map.keys())
        selected_district_code = districts_map[selected_district_label]
        
        st.divider()
        
        # API Key handling
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]

        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here if not set in .env")
        
        generate_btn = st.button("Generate Strategic Report", type="primary")

    # Main Content Area
    if generate_btn:
        if not selected_district_code:
            st.warning("Please select a district.")
        elif not api_key:
            st.error("Please provide an OpenAI API Key.")
        else:
            with st.spinner(f"Analyzing {selected_district_code}... This may take a minute."):
                report_content, error = get_report_content(selected_district_code, api_key=api_key)
                
                if error:
                    st.error(error)
                else:
                    st.success(f"Report generated for {selected_district_code}!")
                    with st.container():
                        st.markdown("---")
                        st.markdown(report_content)
                        st.markdown("---")
                        
                        # Download button
                        st.download_button(
                            label="Download Report as Markdown",
                            data=report_content,
                            file_name=f"cipi_report_{selected_district_code}.md",
                            mime="text/markdown"
                        )

    elif selected_district_code:
        # Show some quick stats when report is not yet generated
        st.info(f"Ready to analyze **{selected_district_code}**.")
        
        # Optional: Show a preview of the data stats
        district_row = df[df['district_code'] == selected_district_code].iloc[0]
        
        cols = st.columns(5)
        cols[0].metric("CIPI Score", f"{district_row.get('CIPI', 0):.1f}")
        cols[1].metric("Vacuum", f"{district_row.get('Vacuum', 0):.1f}")
        cols[2].metric("Protest", f"{district_row.get('Protest', 0):.1f}")
        cols[3].metric("Apathy", f"{district_row.get('Apathy', 0):.1f}")
        cols[4].metric("Demo", f"{district_row.get('Demo', 0):.1f}")

def main():
    if check_login():
        run_dashboard()

if __name__ == "__main__":
    main()
