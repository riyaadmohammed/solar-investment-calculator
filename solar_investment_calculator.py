# ============================================================================
# Solar Investment Calculator for Trinidad and Tobago
# with Machine Learning Predictions (Full PV Model)
# Streamlit Web Application - With API Fallback
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import folium
from streamlit_folium import folium_static, st_folium
import warnings
import os
import glob
import joblib
import sys
from pathlib import Path
warnings.filterwarnings('ignore')


def safe_console_print(*values, sep=" ", end="\n", **_kwargs):
    """Print safely even when the Windows console uses a limited code page."""
    message = sep.join(str(value) for value in values)
    output = sys.stdout
    encoding = getattr(output, "encoding", None) or "utf-8"
    safe_message = message.encode(encoding, errors="backslashreplace").decode(
        encoding, errors="replace"
    )
    output.write(safe_message + end)


# Route all diagnostic prints, including dynamic API column names, through the
# encoding-safe writer without changing Streamlit's browser-rendered Unicode.
print = safe_console_print

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Solar Investment Calculator - Trinidad & Tobago",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 10px;
        color: #0c5460;
    }
    .loading-box {
        background-color: #e8f4f8;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    /* ============================================ */
    /* TAB STYLING - BALANCED FOR RESULTS WINDOW */
    /* ============================================ */
    
    /* Tab container - fits all tabs without scrolling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background-color: #f8f9fa;
        padding: 8px 8px 0 8px !important;
        border-radius: 8px 8px 0 0;
        flex-wrap: nowrap !important;
        overflow: visible !important;
        white-space: nowrap !important;
        justify-content: center !important;
        width: 100% !important;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Individual tab styling - well-proportioned */
    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        white-space: nowrap !important;
        background-color: #f8f9fa;
        border-radius: 6px 6px 0px 0px;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
        color: #000000 !important;
        font-weight: 500;
        border: 1px solid #d0d0d0;
        border-bottom: none;
        flex-shrink: 1 !important;
        flex-grow: 0 !important;
        min-width: auto !important;
        font-size: 13px !important;
        transition: all 0.2s ease;
        margin: 0 2px;
        line-height: 1.3;
        letter-spacing: 0.3px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8f0f8;
        color: #000000 !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: 1px solid #1f77b4;
        border-bottom: none;
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background-color: #1a6aa0 !important;
        color: #ffffff !important;
        transform: translateY(-2px);
    }
    
    /* Ensure tab text is always visible */
    .stTabs [role="tab"] {
        color: #000000 !important;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        color: #ffffff !important;
    }
    
    /* Hide scrollbar on desktop */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }
    
    /* ============================================ */
    /* RESPONSIVE BREAKPOINTS */
    /* ============================================ */
    
    /* Large desktops - more spacing */
    @media screen and (min-width: 1400px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 14px !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
            height: 48px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
        }
    }
    
    /* Standard desktops and laptops */
    @media screen and (min-width: 1025px) and (max-width: 1399px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 13px !important;
            padding-left: 16px !important;
            padding-right: 16px !important;
            height: 44px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
        }
    }
    
    /* Tablets and small laptops */
    @media screen and (max-width: 1024px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 12px !important;
            padding-left: 12px !important;
            padding-right: 12px !important;
            height: 40px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px !important;
            padding: 6px 6px 0 6px !important;
        }
    }
    
    /* Mobile phones - enable scrolling */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        
        /* Enable scrolling on mobile */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            overflow-y: hidden !important;
            justify-content: flex-start !important;
            -webkit-overflow-scrolling: touch;
            padding: 4px 4px 0 4px !important;
            border-bottom: 2px solid #e0e0e0;
        }
        
        /* Show scrollbar on mobile */
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: block !important;
            height: 4px;
        }
        
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
            background: #e0e0e0;
            border-radius: 4px;
        }
        
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
            background: #1f77b4;
            border-radius: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 11px !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
            height: 36px !important;
            flex-shrink: 0 !important;
        }
        
        /* Stack metric cards on mobile */
        .metric-card {
            padding: 10px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.2rem;
        }
        .metric-label {
            font-size: 0.75rem;
        }
        
        /* Adjust button sizes */
        .stButton > button {
            font-size: 14px;
            padding: 8px 16px;
        }
        
        /* Adjust input sizes */
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            font-size: 14px !important;
            padding: 8px !important;
        }
    }
    
    /* Small mobile phones */
    @media screen and (max-width: 480px) {
        .main-header {
            font-size: 1.2rem;
        }
        .sub-header {
            font-size: 0.8rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 10px !important;
            padding-left: 8px !important;
            padding-right: 8px !important;
            height: 32px !important;
        }
        
        .metric-value {
            font-size: 1rem;
        }
    }
    
    /* ============================================ */
    /* DARK MODE SUPPORT */
    /* ============================================ */
    
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2d2d2d;
            border-bottom: 2px solid #444444;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d;
            color: #ffffff !important;
            border-color: #444444;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3d3d3d;
            color: #ffffff !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4 !important;
            color: #ffffff !important;
        }
        
        .stTabs [role="tab"] {
            color: #ffffff !important;
        }
        
        .stTabs [role="tab"][aria-selected="true"] {
            color: #ffffff !important;
        }
        
        .metric-card {
            background-color: #2d2d2d;
        }
        .metric-label {
            color: #adb5bd;
        }
        
        /* Dark mode scrollbar */
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
            background: #3d3d3d;
        }
    }
    
    /* ============================================ */
    /* ADDITIONAL IMPROVEMENTS */
    /* ============================================ */
    
    /* Make sure charts are responsive */
    .js-plotly-plot {
        width: 100% !important;
    }
    
    /* Improve table scrolling on mobile */
    .stDataFrame {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
    }
    
    /* Better spacing for mobile */
    @media screen and (max-width: 768px) {
        .row-widget {
            flex-wrap: wrap !important;
        }
        .row-widget > div {
            flex: 1 1 100% !important;
            margin-bottom: 8px;
        }
        /* Reduce padding on mobile */
        .stApp {
            padding: 8px !important;
        }
        /* Make sure sidebar is accessible */
        .css-1d391kg {
            padding: 1rem !important;
        }
    }
    
    /* Add a subtle indicator for the active tab */
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 20%;
        width: 60%;
        height: 3px;
        background-color: #ffffff;
        border-radius: 2px;
    }
    
    /* Make tabs more visually appealing */
    .stTabs [data-baseweb="tab"] {
        position: relative;
        cursor: pointer;
        user-select: none;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.markdown('<div class="main-header">☀️ Solar Investment Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Trinidad & Tobago - ML-Powered PV Performance & Financial Analysis</div>', unsafe_allow_html=True)

# ============================================================================
# Sidebar - User Inputs
# ============================================================================

with st.sidebar:
    st.markdown("### 📍 System Location")
    
    # Trinidad and Tobago location options
    location_options = {
        "Port of Spain": {"lat": 10.6667, "lon": -61.5167},
        "San Fernando": {"lat": 10.2833, "lon": -61.4667},
        "Arima": {"lat": 10.6333, "lon": -61.2833},
        "Chaguanas": {"lat": 10.5167, "lon": -61.4167},
        "Point Fortin": {"lat": 10.1667, "lon": -61.6833},
        "Scarborough (Tobago)": {"lat": 11.1833, "lon": -60.7333},
        "Crown Point (Tobago)": {"lat": 11.15, "lon": -60.8333},
        "Custom Location": {"lat": None, "lon": None}
    }
    
    selected_location = st.selectbox("Select Location", list(location_options.keys()))
    st.caption("Select **Custom Location** to enter coordinates or choose a point on the map.")

    # A map click is applied before the coordinate widgets are created on the next
    # rerun; Streamlit does not allow widget state to be changed after instantiation.
    pending_coordinates = st.session_state.pop("pending_custom_coordinates", None)
    if pending_coordinates is not None:
        st.session_state["custom_lat_input"] = pending_coordinates["lat"]
        st.session_state["custom_lon_input"] = pending_coordinates["lng"]
    
    if selected_location == "Custom Location":
        col1, col2 = st.columns(2)
        with col1:
            custom_lat = st.number_input(
                "Latitude", value=10.5, format="%.4f", step=0.01,
                key="custom_lat_input"
            )
        with col2:
            custom_lon = st.number_input(
                "Longitude", value=-61.5, format="%.4f", step=0.01,
                key="custom_lon_input"
            )
        lat = custom_lat
        lon = custom_lon
    else:
        lat = location_options[selected_location]["lat"]
        lon = location_options[selected_location]["lon"]
    
    # Show location on map
    st.markdown("### 🗺️ Location Map")
    m = folium.Map(location=[lat, lon], zoom_start=10, control_scale=True)
    folium.Marker(
        [lat, lon],
        popup=selected_location,
        icon=folium.Icon(color='red', icon='sun')
    ).add_to(m)
    sidebar_map = st_folium(
        m,
        key="sidebar_location_map",
        height=250,
        use_container_width=True,
        returned_objects=["last_clicked"],
    )
    if selected_location == "Custom Location" and sidebar_map.get("last_clicked"):
        clicked = sidebar_map["last_clicked"]
        if (abs(clicked["lat"] - lat) > 0.00005
                or abs(clicked["lng"] - lon) > 0.00005):
            st.session_state["pending_custom_coordinates"] = clicked
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚡ System Specifications")
    
    system_capacity = st.number_input(
        "System Capacity (kWp)",
        min_value=1.0,
        max_value=1000.0,
        value=5.0,
        step=0.5,
        help="Total DC capacity of your solar panels"
    )
    
    tilt = st.slider(
        "Panel Tilt Angle (degrees)",
        min_value=0,
        max_value=90,
        value=10,
        help="Angle of solar panels from horizontal"
    )
    
    azimuth = st.slider(
        "Panel Azimuth (degrees from North)",
        min_value=0,
        max_value=360,
        value=180,
        help="0° = North, 90° = East, 180° = South, 270° = West"
    )
    
    st.markdown("---")
    panel_coverage_factor = st.slider(
        "Panel Coverage Factor", 0.05, 1.00, 1.00, 0.01,
        help="Fraction of stated DC capacity represented by installed and active panels."
    )

    st.markdown("### 💰 Financial Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        capex = st.number_input(
            "CAPEX ($TTD)",
            min_value=1000,
            max_value=10000000,
            value=50000,
            step=1000,
            help="Initial system cost"
        )
    with col2:
        o_and_m = st.number_input(
            "O&M ($TTD/year)",
            min_value=0,
            max_value=100000,
            value=1000,
            step=100,
            help="Annual operations and maintenance"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        electricity_rate = st.number_input(
            "Electricity Rate ($TTD/kWh)",
            min_value=0.01,
            max_value=5.0,
            value=0.40,
            step=0.01,
            help="Current electricity price"
        )
    with col4:
        discount_rate = st.number_input(
            "Discount Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
            help="Required rate of return"
        ) / 100
    
    st.markdown("---")
    
    with st.expander("Advanced financial assumptions"):
        electricity_escalation_rate = st.number_input(
            "Electricity escalation (%/year)", 0.0, 20.0, 3.0, 0.5
        ) / 100
        om_inflation_rate = st.number_input(
            "O&M inflation (%/year)", 0.0, 20.0, 2.5, 0.5
        ) / 100
        tax_rate = st.number_input("Income tax rate (%)", 0.0, 60.0, 25.0, 1.0) / 100
        itc_rate = st.number_input("Investment tax credit (%)", 0.0, 100.0, 0.0, 1.0) / 100
        system_lifetime = st.number_input("Project life (years)", 5, 40, 25, 1)
        annual_degradation_rate = st.number_input(
            "Annual degradation (%)", 0.0, 5.0, 0.5, 0.1
        ) / 100

    # Use ML Predictions toggle
    use_ml = st.checkbox(
        "🧠 Use Machine Learning Predictions",
        value=True,
        help="Enable ML models for improved weather predictions (may take longer)"
    )
    
    st.markdown("---")
    calculate = st.button("🚀 Calculate", type="primary", width="stretch")

# ============================================================================
# Trinidad & Tobago Climate Defaults (Fallback when APIs fail)
# ============================================================================

TT_CLIMATE_DEFAULTS = {
    'Temperature at 2 Meters (C)': 28.0,
    'T2M': 28.0,
    'Max Temperature at 2 Meters (C)': 30.0,
    'T2M_MAX': 30.0,
    'Min Temperature at 2 Meters (C)': 24.0,
    'T2M_MIN': 24.0,
    'Relative Humidity at 2 Meters (%)': 75.0,
    'RH2M': 75.0,
    'Wind Speed at 10 Meters (m/s)': 5.0,
    'WS10M': 5.0,
    'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)': 5.5,
    'ALLSKY_SFC_SW_DWN': 5.5,
    'Precipitation Corrected (mm/day)': 0.0,
    'PRECTOTCORR': 0.0,
    'T2MDEW': 22.0,
    'T2MWET': 24.0,
    'WS10M_MAX': 7.0,
    'WS10M_MIN': 2.0,
    'WD10M': 180.0,
    'ALLSKY_SFC_UV_INDEX': 8.0,
    'ALLSKY_KT': 0.5,
    'PS': 101.3,
    'IMERG_PRECTOT': 0.0,
    'sunshine_duration': 43200,  # 12 hours
    'daylight_duration': 43200,
    'temperature_2m_mean': 28.0,
    'temperature_2m_max': 30.0,
    'temperature_2m_min': 24.0,
    'precipitation_sum': 0.0,
    'shortwave_radiation_sum': 5.5,
}

# ============================================================================
# API Client Classes with Fallback
# ============================================================================

# ============================================================================
# NASA POWER API Client - WORKING VERSION (from test_nasa_api.py)
# ============================================================================

class NASAPOWERClient:
    """
    Client for NASA Prediction Of Worldwide Energy Resources API
    Documentation: https://power.larc.nasa.gov/docs/services/api/
    """
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Parameter mapping - POWER codes to readable names
    PARAMETERS = {
        # Temperature parameters 
        'T2M': 'Temperature at 2 Meters (C)',
        'T2M_MAX': 'Max Temperature at 2 Meters (C)',
        'T2M_MIN': 'Min Temperature at 2 Meters (C)',
        'T2MDEW': 'Dew Point at 2 Meters (C)',
        'T2MWET': 'Wet Bulb Temperature at 2 Meters (C)',
        
        # Humidity parameters
        'RH2M': 'Relative Humidity at 2 Meters (%)',
        
        # Wind parameters
        'WS10M': 'Wind Speed at 10 Meters (m/s)',
        'WS10M_MAX': 'Max Wind Speed at 10 Meters (m/s)',
        'WS10M_MIN': 'Min Wind Speed at 10 Meters (m/s)',
        'WD10M': 'Wind Direction at 10 Meters (Degrees)',
        
        # Solar radiation parameters
        'ALLSKY_SFC_SW_DWN': 'All Sky Surface Shortwave Downward Irradiance (kW-hr/m²/day)',
        'CLRSKY_SFC_SW_DWN': 'Clear Sky Surface Shortwave Downward Irradiance (kW-hr/m²/day)',
        'ALLSKY_SFC_UV_INDEX': 'All Sky UV Index',
        'ALLSKY_KT': 'Insolation Clearness Index',
        'CLOUD_AMT': 'Total Cloud Cover (fraction)',
        
        # Precipitation
        'PRECTOTCORR': 'Precipitation Corrected (mm/day)',
        'IMERG_PRECTOT': 'Total Precipitation (mm/day)',
        
        # Pressure
        'PS': 'Surface Pressure (kPa)',
    }
    
    def __init__(self, latitude, longitude, community='RE'):
        """
        Initialize with your location
        community: 'AG' for agriculture, 'SB' for sustainable buildings, 'RE' for renewable energy
        """
        self.latitude = latitude
        self.longitude = longitude
        self.community = community

    DEFAULT_PV_PARAMETERS = [
        'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS10M', 
        'ALLSKY_SFC_SW_DWN', 'PRECTOTCORR'
    ]
    
    def fetch_daily_data(self, start_date, end_date, parameters=None):
        """
        Fetch daily data from NASA POWER
        
        Parameters:
        -----------
        start_date : str (YYYYMMDD) or datetime
        end_date : str (YYYYMMDD) or datetime
        parameters : list of POWER parameter codes (None = use PV defaults)
        
        Returns:
        --------
        pandas.DataFrame with daily data
        """
        # Format dates
        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y%m%d')
        else:
            start_str = str(start_date).replace('-', '')
            
        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y%m%d')
        else:
            end_str = str(end_date).replace('-', '')
        
        # Use default PV parameters if none specified
        if parameters is None:
            parameters = self.DEFAULT_PV_PARAMETERS
        
        # Build request URL
        param_str = ','.join(parameters)
        url = (f"{self.BASE_URL}?"
               f"parameters={param_str}&"
               f"community={self.community}&"
               f"longitude={self.longitude}&"
               f"latitude={self.latitude}&"
               f"start={start_str}&"
               f"end={end_str}&"
               f"format=JSON")
        
        print("      Requesting NASA POWER data...")
        print(f"      URL: {url}")
        
        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                data = response.json()
                print("      NASA POWER data received")
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"      NASA POWER request failed: {e}")
                    return pd.DataFrame()
                print(f"      Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
        
        # Parse response
        if 'properties' not in data or 'parameter' not in data['properties']:
            print("      Unexpected NASA POWER API response format")
            return pd.DataFrame()
        
        # Convert to DataFrame
        dfs = []
        for param, values in data['properties']['parameter'].items():
            param_df = pd.DataFrame(list(values.items()), columns=['date', param])
            param_df['date'] = pd.to_datetime(param_df['date'])
            dfs.append(param_df.set_index('date'))
        
        # Combine all parameters
        result = pd.concat(dfs, axis=1)
        
        # Preserve both API codes and the NASA_ names used by the final models.
        raw_nasa = result.copy()

        # Rename columns to readable names
        result.rename(columns={k: self.PARAMETERS.get(k, k) for k in result.columns}, inplace=True)

        training_names = {
            'ALLSKY_SFC_SW_DWN': 'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)',
            'CLRSKY_SFC_SW_DWN': 'Clear Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)',
            'T2M': 'Temperature at 2 Meters (C)',
            'T2M_MAX': 'Max Temperature at 2 Meters (C)',
            'T2M_MIN': 'Min Temperature at 2 Meters (C)',
            'T2MDEW': 'Dew Point at 2 Meters (C)',
            'T2MWET': 'Wet Bulb Temperature at 2 Meters (C)',
            'RH2M': 'Relative Humidity at 2 Meters (%)',
            'WS10M': 'Wind Speed at 10 Meters (m/s)',
            'WS10M_MAX': 'Max Wind Speed at 10 Meters (m/s)',
            'WS10M_MIN': 'Min Wind Speed at 10 Meters (m/s)',
            'WD10M': 'Wind Direction at 10 Meters (Degrees)',
            'ALLSKY_SFC_UV_INDEX': 'All Sky UV Index',
            'ALLSKY_KT': 'Insolation Clearness Index',
            'PRECTOTCORR': 'Precipitation Corrected (mm/day)',
            'PS': 'Surface Pressure (kPa)',
            'CLOUD_AMT': 'Total Cloud Cover (fraction)',
        }
        for raw_column in raw_nasa.columns:
            readable_column = self.PARAMETERS.get(raw_column, raw_column)
            result[raw_column] = raw_nasa[raw_column]
            result[f"NASA_{readable_column}"] = raw_nasa[raw_column]
            if raw_column in training_names:
                result[f"NASA_{training_names[raw_column]}"] = raw_nasa[raw_column]
        
        # Fill any missing values with Trinidad & Tobago defaults
        for col in result.columns:
            if col in TT_CLIMATE_DEFAULTS:
                result[col] = result[col].fillna(TT_CLIMATE_DEFAULTS[col])
        
        return result
    
    def create_fallback_data(self, start_date, end_date):
        """Create fallback data using Trinidad & Tobago climate defaults"""
        if isinstance(start_date, datetime):
            start = start_date
        else:
            start = pd.to_datetime(start_date)
        
        if isinstance(end_date, datetime):
            end = end_date
        else:
            end = pd.to_datetime(end_date)
        
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Create DataFrame with defaults
        data = pd.DataFrame(index=dates)
        
        for param, default_value in TT_CLIMATE_DEFAULTS.items():
            # Only add parameters that are in our mapping
            if param in self.PARAMETERS.values() or param in self.PARAMETERS.keys():
                col_name = param
                # If it's a key, map to readable name
                if param in self.PARAMETERS:
                    col_name = self.PARAMETERS[param]
                data[col_name] = default_value + np.random.normal(0, 0.1 * default_value, len(dates))
        
        st.info(f"📊 Using Trinidad & Tobago climate defaults for {len(data)} days")
        return data


# ---------- Open-Meteo Client Class ----------
class OpenMeteoClient:
    """
    Client for Open-Meteo API - free weather forecast and historical data 
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameter groupings for better organization
    PARAMETER_GROUPS = {
        'Temperature': [
            'temperature_2m_max',
            'temperature_2m_min',
            'temperature_2m_mean',
            'apparent_temperature_max',
            'apparent_temperature_min',
            'apparent_temperature_mean'
        ],
        'Sun & Radiation': [
            'sunrise',
            'sunset',
            'daylight_duration',
            'sunshine_duration',
            'shortwave_radiation_sum'
        ],
        'Precipitation': [
            'precipitation_sum',
            'precipitation_probability_max',
            'showers_sum',
            'rain_sum',
            'snowfall_sum',
            'precipitation_hours'
        ],
        'Wind': [
            'wind_speed_10m_max',
            'wind_gusts_10m_max',
            'wind_direction_10m_dominant'
        ],
        'Other': [
            'et0_fao_evapotranspiration',
            'relative_humidity_2m_max',
            'relative_humidity_2m_mean',
            'relative_humidity_2m_min',
            'cloud_cover_max',
            'cloud_cover_mean',
            'cloud_cover_min'
        ]
    }
    
    # Human-readable names for parameters
    PARAMETER_NAMES = {
        'temperature_2m_max': 'Max Temperature (°C)',
        'temperature_2m_min': 'Min Temperature (°C)',
        'temperature_2m_mean': 'Mean Temperature (°C)',
        'apparent_temperature_max': 'Max Apparent Temperature (°C)',
        'apparent_temperature_min': 'Min Apparent Temperature (°C)',
        'apparent_temperature_mean': 'Mean Apparent Temperature (°C)',
        'sunrise': 'Sunrise (Time)',
        'sunset': 'Sunset (Time)',
        'daylight_duration': 'Daylight Duration (s)',
        'sunshine_duration': 'Sunshine Duration (s)',
        'precipitation_sum': 'Total Precipitation (mm)',
        'precipitation_probability_max': 'Max Precipitation Probability (%)',
        'showers_sum': 'Total Showers (mm)',
        'rain_sum': 'Total Rain (mm)',
        'snowfall_sum': 'Total Snowfall (cm)',
        'precipitation_hours': 'Precipitation Hours (h)',
        'wind_speed_10m_max': 'Max Wind Speed (km/h)',
        'wind_gusts_10m_max': 'Max Wind Gusts (km/h)',
        'wind_direction_10m_dominant': 'Dominant Wind Direction (°)',
        'shortwave_radiation_sum': 'Shortwave Radiation (MJ/m²)',
        'et0_fao_evapotranspiration': 'ET₀ Evapotranspiration (mm)'
    }
    
    # Columns that contain time/string data (not numeric)
    NON_NUMERIC_COLUMNS = ['sunrise', 'sunset']
    
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
    
    def fetch_historical_data(self, start_date, end_date, parameters=None):
        """
        Fetch historical weather data from Open-Meteo
        
        Parameters:
        -----------
        start_date : str (YYYY-MM-DD)
        end_date : str (YYYY-MM-DD)
        parameters : list of parameter codes (None = all available)
        
        Returns:
        --------
        pandas.DataFrame with daily data
        """
        
        # Default to all parameters if none specified
        if parameters is None:
            parameters = []
            for group in self.PARAMETER_GROUPS.values():
                parameters.extend(group)
        
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': parameters,
            'timezone': 'auto',
            'format': 'json'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Open-Meteo request failed: {e}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        if 'daily' not in data:
            st.error("Unexpected Open-Meteo API response format")
            return pd.DataFrame()
            
        df = pd.DataFrame(data['daily'])
        df['date'] = pd.to_datetime(df['time'])
        df = df.drop('time', axis=1)
        df.set_index('date', inplace=True)

        # Preserve the raw API names and add the OM_ names used during training.
        raw_openmeteo = df.copy()

        # Rename columns to readable names
        df.rename(columns={k: self.PARAMETER_NAMES.get(k, k) for k in df.columns}, inplace=True)

        for raw_column in raw_openmeteo.columns:
            if raw_column not in df.columns:
                df[raw_column] = raw_openmeteo[raw_column]
            df[f"OM_{raw_column}"] = raw_openmeteo[raw_column]
        
        # Handle any missing values
        df = df.replace(-999, np.nan)
        
        # Parse time columns (sunrise/sunset) to extract hour as numeric value
        # This allows them to be used in analysis while preserving the original format
        for col in df.columns:
            if 'Time' in col or 'Sunrise' in col or 'Sunset' in col:
                # Convert time strings to numeric hours since midnight
                df[f'{col} (Hours)'] = pd.to_datetime(df[col], errors='coerce').dt.hour + \
                                       pd.to_datetime(df[col], errors='coerce').dt.minute / 60
        
        return df
    
    def create_fallback_data(self, start_date, end_date):
        """Create fallback data using Trinidad & Tobago climate defaults"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = pd.DataFrame(index=dates)
        
        for param, default_value in TT_CLIMATE_DEFAULTS.items():
            if param in ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
                         'precipitation_sum', 'shortwave_radiation_sum', 'sunshine_duration',
                         'daylight_duration', 'wind_speed_10m_max']:
                data[param] = default_value + np.random.normal(0, 0.05 * default_value, len(dates))
        
        st.info(f"📊 Using Trinidad & Tobago climate defaults for {len(data)} days")
        return data

# ============================================================================
# Full PV Model Classes (From the reference)
# ============================================================================

class CellTemperatureModel:
    """PV cell temperature model using NOCT method"""
    
    def __init__(self, noct=45, reference_irradiance=800, reference_temp=20):
        self.noct = noct
        self.ref_irradiance = reference_irradiance
        self.ref_temp = reference_temp
    
    def calculate_cell_temperature(self, ambient_temp, irradiance, wind_speed=None):
        ambient_temp = np.array(ambient_temp)
        irradiance = np.array(irradiance)
        
        if np.mean(irradiance) < 10:
            irradiance_wm2 = irradiance * 1000 / 12
        else:
            irradiance_wm2 = irradiance
        
        cell_temp = ambient_temp + (self.noct - self.ref_temp) * (irradiance_wm2 / self.ref_irradiance)
        
        if wind_speed is not None:
            wind_speed = np.array(wind_speed)
            wind_factor = np.maximum(0.5, 1 - 0.05 * wind_speed)
            cell_temp = self.ref_temp + (cell_temp - self.ref_temp) * wind_factor
        
        return cell_temp
    
    def calculate_power_loss(self, cell_temp, temp_coefficient=-0.004):
        cell_temp = np.array(cell_temp)
        temp_delta = cell_temp - 25
        power_factor = 1 + temp_coefficient * temp_delta
        return np.maximum(0.5, power_factor)


class SolarGeometry:
    """Calculate solar geometry and plane-of-array irradiance"""
    
    def __init__(self, latitude, longitude, elevation_m, tilt, azimuth):
        self.latitude = np.radians(latitude)
        self.longitude = longitude
        self.elevation_m = elevation_m
        self.tilt = np.radians(tilt)
        self.azimuth = np.radians(azimuth)
        
    def solar_declination(self, day_of_year):
        return 23.44 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    
    def hour_angle(self, solar_time_hours):
        return np.radians(15 * (solar_time_hours - 12))
    
    def solar_altitude(self, latitude, declination, hour_angle):
        sin_alt = (np.sin(latitude) * np.sin(declination) + 
                   np.cos(latitude) * np.cos(declination) * np.cos(hour_angle))
        return np.arcsin(np.clip(sin_alt, -1, 1))
    
    def solar_azimuth(self, latitude, declination, hour_angle, altitude):
        cos_az = ((np.sin(declination) - np.sin(latitude) * np.sin(altitude)) / 
                  (np.cos(latitude) * np.cos(altitude)))
        cos_az = np.clip(cos_az, -1, 1)
        az = np.arccos(cos_az)
        if hour_angle > 0:
            az = 2 * np.pi - az
        return az
    
    def incidence_angle(self, latitude, declination, hour_angle):
        altitude = self.solar_altitude(latitude, declination, hour_angle)
        solar_az = self.solar_azimuth(latitude, declination, hour_angle, altitude)
        
        panel_az_math = np.pi/2 - self.azimuth
        
        panel_normal = np.array([
            np.sin(self.tilt) * np.sin(panel_az_math),
            np.sin(self.tilt) * np.cos(panel_az_math),
            np.cos(self.tilt)
        ])
        
        sun_dir = np.array([
            np.cos(altitude) * np.sin(solar_az),
            np.cos(altitude) * np.cos(solar_az),
            np.sin(altitude)
        ])
        
        cos_incidence = np.clip(np.dot(panel_normal, sun_dir), -1, 1)
        return np.arccos(cos_incidence)
    
    def poa_irradiance(self, dni, dhi, ghi, date):
        day_of_year = date.timetuple().tm_yday
        declination = np.radians(self.solar_declination(day_of_year))
        
        hour_angle = 0
        latitude = self.latitude
        altitude = self.solar_altitude(latitude, declination, hour_angle)
        incidence = self.incidence_angle(latitude, declination, hour_angle)
        
        solar_zenith = np.pi/2 - altitude
        
        if np.abs(np.cos(incidence)) > 1e-6 and np.abs(np.cos(solar_zenith)) > 1e-6:
            if incidence < np.pi/2 and solar_zenith < np.pi/2:
                beam_component = dni * np.cos(incidence) / np.cos(solar_zenith)
                beam_component = max(0, beam_component)
            else:
                beam_component = 0
        else:
            beam_component = 0
        
        diffuse_component = dhi * (1 + np.cos(self.tilt)) / 2
        
        albedo = 0.2
        ground_component = ghi * albedo * (1 - np.cos(self.tilt)) / 2
        
        poa = beam_component + diffuse_component + ground_component
        return max(0, poa)
    
    def calculate_tilt_factor(self, date=None, use_annual_average=False):
        if use_annual_average:
            tilt_factor = 1 + 0.2 * np.cos(self.tilt - 0.5)
            return np.clip(tilt_factor, 0.5, 1.3)
        
        if date is None:
            date = datetime.now()
        
        day_of_year = date.timetuple().tm_yday
        declination = np.radians(self.solar_declination(day_of_year))
        
        alt_rad = np.arcsin(
            np.sin(self.latitude) * np.sin(declination) + 
            np.cos(self.latitude) * np.cos(declination)
        )
        
        if alt_rad <= 0.01:
            return 0
        
        panel_az_from_south = np.pi - self.azimuth
        
        cos_incidence = (np.sin(alt_rad) * np.cos(self.tilt) + 
                        np.cos(alt_rad) * np.sin(self.tilt) * np.cos(panel_az_from_south))
        cos_incidence = np.clip(cos_incidence, 0.01, 1)
        
        beam_tilt_factor = cos_incidence / np.sin(alt_rad)
        diffuse_factor = (1 + np.cos(self.tilt)) / 2
        ground_factor = 0.2 * (1 - np.cos(self.tilt)) / 2
        
        tilt_factor = 0.7 * beam_tilt_factor + 0.3 * diffuse_factor + ground_factor
        return np.clip(tilt_factor, 0.3, 1.8)


class SoilingModel:
    """Dynamic soiling model with dust day integration"""
    
    def __init__(self, accumulation_rate=0.003, cleaning_threshold=5.0, max_soiling=0.20,
                 dust_enhancement_factor=2.0, dust_decay_days=3):
        self.accumulation_rate = accumulation_rate
        self.cleaning_threshold = cleaning_threshold
        self.max_soiling = max_soiling
        self.dust_enhancement_factor = dust_enhancement_factor
        self.dust_decay_days = dust_decay_days
    
    def calculate_dust_influence(self, dust_days_array, dust_prob_array=None):
        n_days = len(dust_days_array)
        dust_factor = np.ones(n_days)
        days_since_dust = np.inf * np.ones(n_days)
        last_dust = -np.inf
        
        for i in range(n_days):
            if dust_days_array[i] == 1:
                last_dust = i
                days_since_dust[i] = 0
            else:
                days_since_dust[i] = i - last_dust if last_dust >= 0 else np.inf
        
        for i in range(n_days):
            if dust_days_array[i] == 1:
                if dust_prob_array is not None:
                    enhancement = 1 + (self.dust_enhancement_factor - 1) * dust_prob_array[i]
                else:
                    enhancement = self.dust_enhancement_factor
                dust_factor[i] = enhancement
            elif days_since_dust[i] <= self.dust_decay_days:
                decay_factor = 1 - (days_since_dust[i] / self.dust_decay_days) * 0.5
                dust_factor[i] = 1 + (self.dust_enhancement_factor - 1) * decay_factor
        
        return dust_factor
    
    def simulate_soiling(self, daily_rainfall, dust_days_array, dust_prob_array=None, 
                         dry_spell_sensitivity=False, seasonal_enhancement=False, 
                         dates=None):
        daily_rainfall = np.array(daily_rainfall)
        dust_days_array = np.array(dust_days_array)
        n_days = len(daily_rainfall)
        
        soiling = np.zeros(n_days)
        dry_spell_counter = 0
        
        dust_influence = self.calculate_dust_influence(dust_days_array, dust_prob_array)
        
        if seasonal_enhancement and dates is not None:
            seasonal_multiplier = np.ones(n_days)
            for i, date in enumerate(dates):
                month = pd.to_datetime(date).month
                if 3 <= month <= 8:
                    seasonal_multiplier[i] = 1.3
            dust_influence = dust_influence * seasonal_multiplier
        
        for i in range(n_days):
            if daily_rainfall[i] >= self.cleaning_threshold:
                soiling[i] = 0
                dry_spell_counter = 0
            else:
                if dry_spell_sensitivity:
                    spell_factor = 1 + 0.1 * np.sqrt(dry_spell_counter)
                else:
                    spell_factor = 1
                
                daily_accumulation = self.accumulation_rate * spell_factor * dust_influence[i]
                
                if i > 0:
                    soiling[i] = min(soiling[i-1] + daily_accumulation, self.max_soiling)
                else:
                    soiling[i] = min(daily_accumulation, self.max_soiling)
                
                dry_spell_counter += 1
        
        return 1 - soiling, dust_influence


class DegradationModel:
    """Climate-adjusted PV degradation model"""
    
    def __init__(self, baseline_degradation=0.005, baseline_uncertainty=0.002,
                 temp_sensitivity=0.05, humidity_sensitivity=0.01, 
                 rainfall_sensitivity=0.005):
        self.baseline_degradation = baseline_degradation
        self.baseline_uncertainty = baseline_uncertainty
        self.temp_sensitivity = temp_sensitivity
        self.humidity_sensitivity = humidity_sensitivity
        self.rainfall_sensitivity = rainfall_sensitivity
        
        self.ref_temp = 25
        self.ref_humidity = 60
        self.ref_rainfall = 1500
    
    def calculate_climate_factors(self, temperature, humidity=None, rainfall=None):
        temperature = np.array(temperature)
        
        temp_factor = np.exp(self.temp_sensitivity * (temperature - self.ref_temp))
        
        if humidity is not None:
            humidity = np.array(humidity)
            humidity_factor = 1 + self.humidity_sensitivity * (humidity - self.ref_humidity) / 10
        else:
            humidity_factor = 1.0
        
        if rainfall is not None:
            rainfall = np.array(rainfall)
            annual_rainfall = rainfall * 365
            rainfall_factor = 1 + self.rainfall_sensitivity * (annual_rainfall - self.ref_rainfall) / 100
        else:
            rainfall_factor = 1.0
        
        climate_factor = temp_factor * humidity_factor * rainfall_factor
        return climate_factor
    
    def calculate_degradation_factor(self, temperature, system_age_years, 
                                      humidity=None, rainfall=None,
                                      degradation_rate=None):
        climate_factors = self.calculate_climate_factors(temperature, humidity, rainfall)
        mean_climate_factor = np.mean(climate_factors)
        
        if degradation_rate is not None:
            rate = degradation_rate
        else:
            rate = self.baseline_degradation * mean_climate_factor
        
        degradation_factor = np.exp(-rate * system_age_years)
        degradation_factor = np.clip(degradation_factor, 0.7, 1.0)
        
        return degradation_factor, mean_climate_factor


class PVPerformanceModel:
    """
    Integrated PV performance model combining:
    - Cell Temperature Model (NOCT method)
    - Solar Geometry Model (POA irradiance calculation)
    - Soiling Model (dust-enhanced soiling)
    - Degradation Model (climate-adjusted degradation)
    """
    
    def __init__(self, rated_power_kw, cell_temp_model, solar_geometry_model,
                 soiling_model, degradation_model, inverter_efficiency=0.96, 
                 system_loss=0.10, panel_coverage_factor=1.0):
        self.rated_power_kw = rated_power_kw
        self.panel_coverage_factor = float(panel_coverage_factor)
        if not 0 < self.panel_coverage_factor <= 1:
            raise ValueError("panel_coverage_factor must be greater than 0 and at most 1")
        self.effective_rated_power_kw = self.rated_power_kw * self.panel_coverage_factor
        self.cell_temp_model = cell_temp_model
        self.solar_geometry = solar_geometry_model
        self.soiling_model = soiling_model
        self.degradation_model = degradation_model
        self.inverter_efficiency = inverter_efficiency
        self.system_loss = system_loss
    
    def calculate_daily_energy(self, ambient_temp, irradiance, wind_speed=None, 
                               rainfall=None, dust_days=None, dust_prob=None,
                               dates=None, sunshine_duration=None, 
                               humidity=None, system_age_years=0,
                               degradation_rate=None, dni=None, dhi=None):
        ambient_temp = np.array(ambient_temp)
        irradiance = np.array(irradiance)
        n_days = len(irradiance)
        
        # Step 1: Calculate POA Irradiance
        poa_kwh_m2 = np.zeros(n_days)
        tilt_factors = np.zeros(n_days)
        
        for i in range(n_days):
            date = None
            if dates is not None:
                date = pd.to_datetime(dates[i])
            else:
                date = datetime.now()
            
            if dni is not None and dhi is not None:
                poa_wm2 = self.solar_geometry.poa_irradiance(
                    dni[i], dhi[i], irradiance[i], date
                )
                poa_kwh_m2[i] = poa_wm2 * 12 / 1000
                tilt_factors[i] = poa_kwh_m2[i] / irradiance[i] if irradiance[i] > 0 else 1
            else:
                tilt_factor = self.solar_geometry.calculate_tilt_factor(date)
                tilt_factors[i] = tilt_factor
                poa_kwh_m2[i] = irradiance[i] * tilt_factor
        
        # Step 2: Cell Temperature
        if sunshine_duration is not None:
            sunshine_hours = np.array(sunshine_duration) / 3600
            sunshine_hours = np.maximum(sunshine_hours, 0.1)
        else:
            sunshine_hours = np.ones(n_days) * 8
        
        avg_irradiance_wm2 = (irradiance * 1000) / sunshine_hours
        avg_irradiance_wm2 = np.nan_to_num(avg_irradiance_wm2, nan=0)
        
        cell_temp = self.cell_temp_model.calculate_cell_temperature(
            ambient_temp, irradiance, wind_speed
        )
        temp_factor = self.cell_temp_model.calculate_power_loss(cell_temp)
        
        # Step 3: Soiling
        soiling_factor = np.ones(n_days)
        dust_influence = np.ones(n_days)
        
        if rainfall is not None:
            if dust_days is not None:
                soiling_factor, dust_influence = self.soiling_model.simulate_soiling(
                    rainfall, dust_days, dust_prob, 
                    dry_spell_sensitivity=True, 
                    seasonal_enhancement=True,
                    dates=dates
                )
            else:
                soiling_factor, dust_influence = self.soiling_model.simulate_soiling(
                    rainfall, np.zeros(n_days), None
                )
        
        # Step 4: Degradation
        if system_age_years > 0:
            degradation_factor, climate_factor = self.degradation_model.calculate_degradation_factor(
                ambient_temp, system_age_years, humidity, rainfall, degradation_rate
            )
        else:
            degradation_factor = 1.0
            climate_factor = 1.0
        
        # Step 5: Energy Calculation
        energy_dc = self.effective_rated_power_kw * poa_kwh_m2
        
        energy_ac = (energy_dc * 
                    temp_factor * 
                    soiling_factor * 
                    degradation_factor * 
                    self.inverter_efficiency * 
                    (1 - self.system_loss))
        
        energy_ac = np.maximum(0, energy_ac)
        max_energy = self.effective_rated_power_kw * sunshine_hours * 0.9
        energy_ac = np.minimum(energy_ac, max_energy)
        
        return {
            'energy_kwh': energy_ac,
            'cell_temp_c': cell_temp,
            'temp_factor': temp_factor,
            'soiling_factor': soiling_factor,
            'dust_influence': dust_influence,
            'degradation_factor': degradation_factor,
            'climate_factor': climate_factor,
            'tilt_factor': tilt_factors,
            'poa_irradiance': poa_kwh_m2,
            'energy_dc': energy_dc
        }

# ============================================================================
# ML Prediction Functions with Proper NaN Handling
# ============================================================================

# ============================================================================
# Updated ML Prediction Functions with Two-Stage Rainfall Model
# ============================================================================

def load_ml_models():
    """Load the saved ML models - Updated for two-stage rainfall"""
    models = {}
    model_dir = 'saved_models'
    
    if not os.path.exists(model_dir):
        st.warning(f"Model directory '{model_dir}' not found")
        return None
    
    try:
        # Model patterns - including separate classifier and regressor for rainfall
        model_patterns = {
            'pred_max': ('Daily_Max_C_model_*.joblib', 'Daily_Max_C_scaler_*.joblib'),
            'pred_min': ('Daily_Min_C_model_*.joblib', 'Daily_Min_C_scaler_*.joblib'),
            'pred_rh': ('Mean_RH_%_model_*.joblib', 'Mean_RH_%_scaler_*.joblib'),
            'pred_wind': ('Avg_hourly_Wind_speed_KTS_model_*.joblib', 'Avg_hourly_Wind_speed_KTS_scaler_*.joblib'),
            'pred_dust': ('dust_days_classifier_*.joblib', 'dust_days_scaler_*.joblib'),
            # Two-stage rainfall models
            'rain_classifier': ('rainfall_classifier_*.joblib', 'rainfall_scaler_*.joblib'),
            'rain_regressor': ('rainfall_regressor_*.joblib', 'rainfall_scaler_*.joblib'),
        }
        
        loaded_models = []
        
        for key, (model_pattern, scaler_pattern) in model_patterns.items():
            model_files = glob.glob(os.path.join(model_dir, model_pattern))
            scaler_files = glob.glob(os.path.join(model_dir, scaler_pattern))
            
            if model_files and scaler_files:
                models[f'{key}_model'] = joblib.load(sorted(model_files)[-1])
                models[f'{key}_scaler'] = joblib.load(sorted(scaler_files)[-1])
                loaded_models.append(key)
                
                # Store feature info
                if hasattr(models[f'{key}_scaler'], 'feature_names_in_'):
                    models[f'{key}_features'] = list(models[f'{key}_scaler'].feature_names_in_)
        
        if loaded_models:
            st.info(f"✅ Loaded ML models: {', '.join(loaded_models)}")
            return models
        else:
            st.warning("No ML models found in saved_models directory")
            return None
        
    except Exception as e:
        st.error(f"Error loading ML models: {str(e)}")
        return None


def get_feature_value(row, feature_name):
    """
    Safely get a feature value from a row, handling various column name formats
    """
    # Check if feature exists directly
    if feature_name in row.index:
        value = row[feature_name]
        if pd.notna(value):
            return float(value)
    
    # Try alternative names
    alt_names = {
        'Temperature at 2 Meters (C)': ['T2M', 'temperature_2m_mean', 'temp_air'],
        'Max Temperature at 2 Meters (C)': ['T2M_MAX', 'temperature_2m_max', 'temp_max'],
        'Min Temperature at 2 Meters (C)': ['T2M_MIN', 'temperature_2m_min', 'temp_min'],
        'Relative Humidity at 2 Meters (%)': ['RH2M', 'humidity'],
        'Wind Speed at 10 Meters (m/s)': ['WS10M', 'wind_speed_10m_max', 'wind_speed'],
        'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)': ['ALLSKY_SFC_SW_DWN', 'shortwave_radiation_sum', 'irradiance'],
        'Precipitation Corrected (mm/day)': ['PRECTOTCORR', 'precipitation_sum', 'precipitation'],
        'T2MDEW': ['T2MDEW'],
        'T2MWET': ['T2MWET'],
        'WS10M_MAX': ['WS10M_MAX'],
        'WS10M_MIN': ['WS10M_MIN'],
        'WD10M': ['WD10M'],
        'ALLSKY_SFC_UV_INDEX': ['ALLSKY_SFC_UV_INDEX'],
        'ALLSKY_KT': ['ALLSKY_KT'],
        'PS': ['PS'],
        'IMERG_PRECTOT': ['IMERG_PRECTOT'],
        'sunshine_duration': ['sunshine_duration'],
        'daylight_duration': ['daylight_duration'],
    }
    
    # Check if feature has alternative names
    for alt_list in alt_names.values():
        if feature_name in alt_list:
            for alt in alt_list:
                if alt in row.index and pd.notna(row[alt]):
                    return float(row[alt])
    
    # Return None if not found
    return None


def get_default_value(feature_name):
    """Get reasonable default values for features"""
    defaults = {
        'Temperature at 2 Meters (C)': 28.0,
        'T2M': 28.0,
        'Max Temperature at 2 Meters (C)': 30.0,
        'T2M_MAX': 30.0,
        'Min Temperature at 2 Meters (C)': 24.0,
        'T2M_MIN': 24.0,
        'Relative Humidity at 2 Meters (%)': 75.0,
        'RH2M': 75.0,
        'Wind Speed at 10 Meters (m/s)': 5.0,
        'WS10M': 5.0,
        'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)': 5.5,
        'ALLSKY_SFC_SW_DWN': 5.5,
        'Precipitation Corrected (mm/day)': 0.0,
        'PRECTOTCORR': 0.0,
        'T2MDEW': 22.0,
        'T2MWET': 24.0,
        'WS10M_MAX': 7.0,
        'WS10M_MIN': 2.0,
        'WD10M': 180.0,
        'ALLSKY_SFC_UV_INDEX': 8.0,
        'ALLSKY_KT': 0.5,
        'PS': 101.3,
        'IMERG_PRECTOT': 0.0,
        'sunshine_duration': 43200,  # 12 hours
        'daylight_duration': 43200,
    }
    
    # Try exact match
    if feature_name in defaults:
        return defaults[feature_name]
    
    # Try partial match
    for key in defaults:
        if key.lower() in feature_name.lower() or feature_name.lower() in key.lower():
            return defaults[key]
    
    return 0.0


def predict_with_ml(models, row, expected_features):
    """
    Make predictions using loaded ML models with proper NaN handling.
    This now uses the two-stage rainfall model when available.
    """
    results = {}
    
    try:
        # Extract features with proper NaN handling
        feature_vector = []
        for feat in expected_features:
            value = get_feature_value(row, feat)
            
            if value is None:
                value = get_default_value(feat)
            
            feature_vector.append(value)
        
        # Convert to numpy array and check for NaN
        X = np.array(feature_vector).reshape(1, -1)
        
        # Check for NaN values and replace with defaults if any remain
        if np.any(np.isnan(X)):
            for i in range(X.shape[1]):
                if np.isnan(X[0, i]):
                    X[0, i] = get_default_value(expected_features[i])
        
        # Scale features
        if 'pred_max_scaler' in models:
            X_scaled = models['pred_max_scaler'].transform(X)
        else:
            X_scaled = X
        
        # Predict each variable
        if 'pred_max_model' in models:
            results['Pred_Max'] = models['pred_max_model'].predict(X_scaled)[0]
        
        if 'pred_min_model' in models:
            results['Pred_Min'] = models['pred_min_model'].predict(X_scaled)[0]
        
        if 'pred_rh_model' in models:
            results['Pred_RH'] = models['pred_rh_model'].predict(X_scaled)[0]
        
        if 'pred_wind_model' in models:
            results['Pred_Wind'] = models['pred_wind_model'].predict(X_scaled)[0]
        
        if 'pred_dust_model' in models:
            if hasattr(models['pred_dust_model'], 'predict_proba'):
                proba = models['pred_dust_model'].predict_proba(X_scaled)[0]
                pred_class = models['pred_dust_model'].predict(X_scaled)[0]
                results['Pred_Dust'] = pred_class
                results['Dust_Prob'] = proba[1]
            else:
                results['Pred_Dust'] = models['pred_dust_model'].predict(X_scaled)[0]
        
        # ====================================================================
        # TWO-STAGE RAINFALL PREDICTION
        # ====================================================================
        # Check if we have both classifier and regressor for rainfall
        if 'rain_classifier_model' in models and 'rain_regressor_model' in models:
            try:
                # Use the same scaled features for rainfall prediction
                # Step 1: Predict rain occurrence
                rain_classifier = models['rain_classifier_model']
                rain_proba = rain_classifier.predict_proba(X_scaled)[0]
                rain_pred_class = rain_classifier.predict(X_scaled)[0]
                
                results['Rain_Prob'] = rain_proba[1]  # Probability of rain (class 1)
                
                # Step 2: If rain is predicted, predict the amount
                if rain_pred_class == 1:
                    rain_regressor = models['rain_regressor_model']
                    rain_amount = rain_regressor.predict(X_scaled)[0]
                    results['Pred_Rain'] = max(0, rain_amount)  # Clip negative values
                else:
                    results['Pred_Rain'] = 0.0
                    
            except Exception as e:
                st.warning(f"Two-stage rainfall prediction error: {str(e)}")
                # Fallback to simple regressor if available
                if 'pred_rain_model' in models:
                    rain_amount = models['pred_rain_model'].predict(X_scaled)[0]
                    results['Pred_Rain'] = max(0, rain_amount)
                else:
                    results['Pred_Rain'] = 0.0
        else:
            # Fallback to simple regressor if two-stage not available
            if 'pred_rain_model' in models:
                try:
                    rain_amount = models['pred_rain_model'].predict(X_scaled)[0]
                    results['Pred_Rain'] = max(0, rain_amount)
                except Exception as e:
                    st.warning(f"Rainfall prediction error: {str(e)}")
                    results['Pred_Rain'] = 0.0
            else:
                results['Pred_Rain'] = 0.0
        
    except Exception as e:
        st.warning(f"ML prediction error: {str(e)}")
    
    return results


def apply_ml_predictions(weather_df, models):
    """Apply ML predictions to weather data with proper NaN handling"""
    if models is None:
        return weather_df
    
    # Get expected features from the first available scaler
    expected_features = None
    for key in ['pred_max_scaler', 'pred_min_scaler', 'pred_rh_scaler', 'pred_wind_scaler']:
        if key in models and hasattr(models[key], 'feature_names_in_'):
            expected_features = list(models[key].feature_names_in_)
            break
    
    if expected_features is None:
        st.warning("Could not determine expected features from scalers")
        return weather_df
    
    st.info(f"🧠 ML models expecting {len(expected_features)} features")
    
    # Initialize prediction columns with NaN
    pred_cols = ['Pred_Max', 'Pred_Min', 'Pred_RH', 'Pred_Wind', 'Pred_Dust', 'Dust_Prob', 'Pred_Rain', 'Rain_Prob']
    for col in pred_cols:
        if col not in weather_df.columns:
            weather_df[col] = np.nan
    
    # Apply predictions row by row
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(weather_df)
    for idx, (date, row) in enumerate(weather_df.iterrows()):
        status_text.text(f"Processing ML predictions... {date.strftime('%Y-%m-%d')} ({idx+1}/{total_rows})")
        
        pred_results = predict_with_ml(models, row, expected_features)
        
        for key, value in pred_results.items():
            if key in weather_df.columns:
                weather_df.at[date, key] = value
        
        progress_bar.progress((idx + 1) / total_rows)
    
    progress_bar.empty()
    status_text.empty()
    
    return weather_df


# ============================================================================
# Alternative: Use the original two-stage prediction function directly
# ============================================================================

def predict_rainfall_two_stage(row, models):
    """
    Special function for two-stage rainfall model:
    1. Classifier predicts if rain will occur
    2. Regressor predicts the amount if rain is predicted
    
    This is called from within predict_with_ml for each row.
    """
    try:
        # Check if models exist
        if 'rain_classifier_model' not in models or 'rain_regressor_model' not in models:
            return None, "Two-stage rainfall models not loaded"
        
        # Get expected features from scaler
        if 'rain_classifier_features' in models:
            expected_features = models['rain_classifier_features']
        elif hasattr(models['rain_classifier_model'], 'feature_names_in_'):
            expected_features = list(models['rain_classifier_model'].feature_names_in_)
        else:
            # Try to get from scaler
            if 'rain_classifier_scaler' in models and hasattr(models['rain_classifier_scaler'], 'feature_names_in_'):
                expected_features = list(models['rain_classifier_scaler'].feature_names_in_)
            else:
                # Use the same features as other models
                return None, "Could not determine expected features for rainfall model"
        
        # Extract features
        feature_vector = []
        for feat in expected_features:
            value = get_feature_value(row, feat)
            if value is None:
                value = get_default_value(feat)
            feature_vector.append(value)
        
        # Scale features
        X = np.array(feature_vector).reshape(1, -1)
        
        if 'rain_classifier_scaler' in models:
            X_scaled = models['rain_classifier_scaler'].transform(X)
        else:
            X_scaled = X
        
        # Step 1: Predict rain occurrence
        classifier = models['rain_classifier_model']
        rain_prob = classifier.predict_proba(X_scaled)[0][1]
        rain_pred = classifier.predict(X_scaled)[0]
        
        # Step 2: If rain is predicted, predict the amount
        if rain_pred == 1:
            regressor = models['rain_regressor_model']
            amount = regressor.predict(X_scaled)[0]
            amount = max(0, amount)  # Clip negative values
        else:
            amount = 0
        
        return amount, rain_prob
        
    except Exception as e:
        return None, f"Error in two-stage rainfall prediction: {str(e)}"


# ============================================================================
# Updated apply_ml_predictions with explicit two-stage rainfall
# ============================================================================

def apply_ml_predictions_v2(weather_df, models):
    """
    Apply ML predictions to weather data with explicit two-stage rainfall handling
    """
    if models is None:
        return weather_df
    
    # Get expected features from the first available scaler
    expected_features = None
    for key in ['pred_max_scaler', 'pred_min_scaler', 'pred_rh_scaler', 'pred_wind_scaler']:
        if key in models and hasattr(models[key], 'feature_names_in_'):
            expected_features = list(models[key].feature_names_in_)
            break
    
    if expected_features is None:
        st.warning("Could not determine expected features from scalers")
        return weather_df
    
    st.info(f"🧠 ML models expecting {len(expected_features)} features")
    
    # Initialize prediction columns with NaN
    pred_cols = ['Pred_Max', 'Pred_Min', 'Pred_RH', 'Pred_Wind', 'Pred_Dust', 'Dust_Prob', 'Pred_Rain', 'Rain_Prob']
    for col in pred_cols:
        if col not in weather_df.columns:
            weather_df[col] = np.nan
    
    # Apply predictions row by row
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(weather_df)
    for idx, (date, row) in enumerate(weather_df.iterrows()):
        status_text.text(f"Processing ML predictions... {date.strftime('%Y-%m-%d')} ({idx+1}/{total_rows})")
        
        # Predict all variables including two-stage rainfall
        pred_results = predict_with_ml(models, row, expected_features)
        
        # Also try the explicit two-stage rainfall prediction
        if 'rain_classifier_model' in models and 'rain_regressor_model' in models:
            rain_amount, rain_prob = predict_rainfall_two_stage(row, models)
            if rain_amount is not None:
                pred_results['Pred_Rain'] = rain_amount
                pred_results['Rain_Prob'] = rain_prob
        
        for key, value in pred_results.items():
            if key in weather_df.columns:
                weather_df.at[date, key] = value
        
        progress_bar.progress((idx + 1) / total_rows)
    
    progress_bar.empty()
    status_text.empty()
    
    return weather_df

# ============================================================================
# Financial Model
# ============================================================================

def calculate_financial_metrics(annual_energy, capex, o_and_m, electricity_rate, 
                                discount_rate, system_lifetime=25, 
                                degradation_rate=0.005, itc_rate=0.0, tax_rate=0.25):
    """Calculate financial metrics for PV system"""
    
    years = np.arange(1, system_lifetime + 1)
    
    # Energy production with degradation
    energy_production = annual_energy * (1 - degradation_rate) ** (years - 1)
    
    # Revenue from electricity sales
    revenue = energy_production * electricity_rate
    
    # O&M costs
    o_m_costs = np.ones(system_lifetime) * o_and_m
    
    # Cash flows before tax
    cashflow_bt = revenue - o_m_costs
    
    # Simple depreciation (straight line over 10 years)
    depreciation = np.zeros(system_lifetime)
    depreciation[:10] = capex / 10
    
    # Taxes
    ebt = cashflow_bt - depreciation
    taxes = np.maximum(0, ebt * tax_rate)
    
    # Cash flow after tax
    cashflow_at = cashflow_bt - taxes
    
    # Initial investment with ITC
    initial_investment = capex * (1 - itc_rate)
    cashflow_at[0] += capex * itc_rate
    
    # Discounted cash flows
    discount_factors = 1 / (1 + discount_rate) ** years
    discounted_cashflows = cashflow_at * discount_factors
    
    # NPV
    npv = -initial_investment + np.sum(discounted_cashflows)
    
    # Payback Period
    cumulative_cashflow = 0
    payback_period = None
    
    for i, cf in enumerate(cashflow_at):
        cumulative_cashflow += cf
        if payback_period is None and cumulative_cashflow >= initial_investment:
            payback_period = i + 1
    
    # LCOE
    crf = (discount_rate * (1 + discount_rate) ** system_lifetime) / ((1 + discount_rate) ** system_lifetime - 1)
    pv_energy = np.sum(energy_production / (1 + discount_rate) ** years)
    lcoe = (initial_investment * crf) / pv_energy
    
    return {
        'years': years,
        'energy_production': energy_production,
        'revenue': revenue,
        'cashflow_at': cashflow_at,
        'npv': npv,
        'payback_period': payback_period,
        'lcoe': lcoe,
        'total_energy': np.sum(energy_production),
        'total_revenue': np.sum(revenue),
        'net_profit': np.sum(cashflow_at) - initial_investment,
        'annualized_return': (np.sum(cashflow_at) - initial_investment) / system_lifetime
    }

# ========================================================================
# Display Results Function
# ========================================================================

def display_results():
    """Display the results from session state"""
    
    # Retrieve from session state
    weather_df = st.session_state.weather_df
    nasa_df = st.session_state.nasa_df
    openmeteo_df = st.session_state.openmeteo_df
    annual_energy_sat = st.session_state.annual_energy_sat
    financial_sat = st.session_state.financial_sat
    use_ml = st.session_state.use_ml
    
    if use_ml and 'ml_energy_kwh' in weather_df.columns:
        annual_energy_ml = st.session_state.annual_energy_ml
        financial_ml = st.session_state.financial_ml
    
    # [Continue with all the display code from your original file...]
    # This is where all the tabs and charts go
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Summary Metrics - Show both Satellite and ML if available
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Annual Energy (Satellite)</div>
            <div class="metric-value">{annual_energy_sat:,.0f} kWh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Annual Energy (ML)</div>
                <div class="metric-value" style="color: #ff7f0e;">{annual_energy_ml:,.0f} kWh</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Annual Revenue</div>
                <div class="metric-value">${annual_energy_sat * electricity_rate:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ML Improvement</div>
                <div class="metric-value" style="color: #2ca02c;">+{((annual_energy_ml - annual_energy_sat) / annual_energy_sat * 100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Average</div>
                <div class="metric-value">{annual_energy_sat/365:.0f} kWh/day</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CO₂ Offset</div>
            <div class="metric-value">{annual_energy_sat * 0.5 / 1000:,.1f} tons</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Persist the intended tab across the full rerun caused by financial controls.
    results_tab_labels = [
        "📈 Energy Production",
        "💰 Financial Analysis",
        "🌤️ Weather Data",
        "📋 Detailed Results",
        "🧠 ML vs Satellite",
        "📋 Nasa Results",
        "📋 OpenMeteo Results"
    ]
    financial_tab_label = results_tab_labels[1]

    def keep_financial_tab_active():
        st.session_state["results_default_tab"] = financial_tab_label

    default_results_tab = st.session_state.pop(
        "results_default_tab", results_tab_labels[0]
    )
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        results_tab_labels, default=default_results_tab
    )
    
    with tab1:
        st.markdown("### 📈 Daily Energy Production")
        
        # Time series plot with both satellite and ML
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=weather_df.index,
            y=weather_df['satellite_energy_kwh'],
            mode='lines',
            name='Satellite',
            line=dict(color='#1f77b4', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            fig.add_trace(go.Scatter(
                x=weather_df.index,
                y=weather_df['ml_energy_kwh'],
                mode='lines',
                name='ML Predictions',
                line=dict(color='#ff7f0e', width=1.5)
            ))
        
        fig.add_hline(
            y=weather_df['satellite_energy_kwh'].mean(),
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Satellite Avg: {weather_df['satellite_energy_kwh'].mean():.1f} kWh/day",
            annotation_position="bottom right"
        )
        
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            fig.add_hline(
                y=weather_df['ml_energy_kwh'].mean(),
                line_dash="dash",
                line_color="orange",
                annotation_text=f"ML Avg: {weather_df['ml_energy_kwh'].mean():.1f} kWh/day",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title=f"Daily Energy Production - {selected_location}",
            xaxis_title="Date",
            yaxis_title="Energy (kWh/day)",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
        
        # Monthly average comparison
        st.markdown("### 📊 Monthly Average Production Comparison")
        
        monthly_sat = weather_df.groupby(weather_df.index.month)['satellite_energy_kwh'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=month_names,
            y=monthly_sat.values,
            name='Satellite',
            marker_color='#1f77b4',
            text=[f"{v:.1f}" for v in monthly_sat.values],
            textposition='outside'
        ))
        
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            monthly_ml = weather_df.groupby(weather_df.index.month)['ml_energy_kwh'].mean()
            fig2.add_trace(go.Bar(
                x=month_names,
                y=monthly_ml.values,
                name='ML Predictions',
                marker_color='#ff7f0e',
                text=[f"{v:.1f}" for v in monthly_ml.values],
                textposition='outside'
            ))
        
        fig2.update_layout(
            title="Monthly Average Daily Production",
            xaxis_title="Month",
            yaxis_title="Energy (kWh/day)",
            height=350,
            template="plotly_white",
            barmode='group'
        )
        
        st.plotly_chart(fig2)
    
    with tab2:
        st.markdown("### 💰 Financial Analysis")
        
        # Select which model to show - Using a key to prevent re-runs
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            financial_model = st.radio(
                "Select Financial Model",
                ["Satellite", "ML Predictions"],
                horizontal=True,
                key="financial_model_radio",
                on_change=keep_financial_tab_active,
            )
            
            if financial_model == "ML Predictions":
                financial_results = financial_ml
                energy_label = "ML"
            else:
                financial_results = financial_sat
                energy_label = "Satellite"
        else:
            financial_results = financial_sat
            energy_label = "Satellite"
        
        # Cashflow chart
        fig3 = go.Figure()
        
        fig3.add_trace(go.Bar(
            x=financial_results['years'],
            y=financial_results['cashflow_at'],
            name=f'{energy_label} Cashflow',
            marker_color='#2ecc71' if any(financial_results['cashflow_at'] > 0) else '#e74c3c'
        ))
        
        fig3.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig3.update_layout(
            title=f"Annual Cashflow After Tax ({energy_label})",
            xaxis_title="Year",
            yaxis_title="Cashflow ($TTD)",
            height=350,
            template="plotly_white"
        )
        
        st.plotly_chart(fig3)
        
        # Cumulative cashflow
        cumulative_cf = np.cumsum(financial_results['project_cashflows'])
        cumulative_years = np.arange(len(cumulative_cf))
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=cumulative_years,
            y=cumulative_cf,
            mode='lines',
            name=f'{energy_label} Cumulative',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        
        fig4.add_hline(y=0, line_dash="dash", line_color="red")
        
        if financial_results['payback_period']:
            fig4.add_vline(
                x=financial_results['payback_period'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"Payback: Year {financial_results['payback_period']:.0f}",
                annotation_position="top right"
            )
        
        fig4.update_layout(
            title=f"Cumulative Cashflow ({energy_label})",
            xaxis_title="Year",
            yaxis_title="Cumulative Cashflow ($TTD)",
            height=350,
            template="plotly_white"
        )
        
        st.plotly_chart(fig4)
        
        # Financial metrics table
        st.markdown("### 📊 Financial Metrics Summary")
        
        metrics_data = {
            "Metric": [
                "Initial Investment (CAPEX)",
                "Effective Investment (after ITC)",
                "Annual Energy Production",
                "Annual Revenue",
                "Lifetime Energy Production",
                "Lifetime Revenue",
                "Lifetime O&M Costs",
                "Net Lifetime Profit",
                "Annualized Return",
                "Payback Period",
                "Discounted Payback Period",
                "Internal Rate of Return (IRR)",
                "LCOE",
                "NPV"
            ],
            "Value": [
                f"${capex:,.2f}",
                f"${financial_results['net_initial_investment']:,.2f}",
                f"{financial_results['energy_production'][0]:,.0f} kWh/year",
                f"${financial_results['revenue'][0]:,.2f}/year",
                f"{financial_results['total_energy']:,.0f} kWh",
                f"${financial_results['total_revenue']:,.2f}",
                f"${financial_results['total_om']:,.2f}",
                f"${financial_results['net_profit']:,.2f}",
                f"${financial_results['annualized_return']:,.2f}/year",
                f"{financial_results['payback_period']:.1f} years" if financial_results['payback_period'] is not None else f">{len(financial_results['years'])} years",
                f"{financial_results['discounted_payback_period']:.1f} years" if financial_results['discounted_payback_period'] is not None else f">{len(financial_results['years'])} years",
                f"{financial_results['irr'] * 100:.2f}%" if np.isfinite(financial_results['irr']) else "Not available",
                f"${financial_results['lcoe']:.3f}/kWh",
                f"${financial_results['npv']:,.2f}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, hide_index=True, width="stretch")
        
        with tab3:
            st.markdown("### 🌤️ Weather Data")
            
            # Weather summary
            col1, col2, col3 = st.columns(3)
            
            # Temperature
            temp_display = 'Temperature at 2 Meters (C)' if 'Temperature at 2 Meters (C)' in weather_df.columns else 'T2M'
            if temp_display in weather_df.columns:
                with col1:
                    st.metric(
                        "Average Temperature",
                        f"{weather_df[temp_display].mean():.1f}°C",
                        f"Min: {weather_df[temp_display].min():.1f}°C, Max: {weather_df[temp_display].max():.1f}°C"
                    )
            
            # Irradiance
            irrad_display = 'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)' if 'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)' in weather_df.columns else 'ALLSKY_SFC_SW_DWN'
            if irrad_display in weather_df.columns:
                with col2:
                    st.metric(
                        "Average Irradiance",
                        f"{weather_df[irrad_display].mean():.2f} kWh/m²/day",
                        f"Min: {weather_df[irrad_display].min():.2f}, Max: {weather_df[irrad_display].max():.2f}"
                    )
            
            # Wind speed
            wind_display = 'Wind Speed at 10 Meters (m/s)' if 'Wind Speed at 10 Meters (m/s)' in weather_df.columns else 'WS10M'
            if wind_display in weather_df.columns:
                with col3:
                    st.metric(
                        "Average Wind Speed",
                        f"{weather_df[wind_display].mean():.1f} m/s",
                        f"Min: {weather_df[wind_display].min():.1f}, Max: {weather_df[wind_display].max():.1f}"
                    )
            
            # Weather data table
            st.markdown("### 📋 Weather Data (Last 30 Days)")
            
            display_cols = []
            for col in ['Temperature at 2 Meters (C)', 'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)', 
                       'Wind Speed at 10 Meters (m/s)', 'satellite_energy_kwh']:
                if col in weather_df.columns:
                    display_cols.append(col)
            
            if use_ml and 'ml_energy_kwh' in weather_df.columns:
                display_cols.append('ml_energy_kwh')
            
            display_df = weather_df.tail(30)[display_cols].copy()
            
            st.dataframe(display_df.round(2), width="stretch")
        
        with tab4:
            st.markdown("### 📋 Detailed Results")
            
            # Combine all results
            detailed_df = weather_df.copy()
            
            # Select columns for display
            display_cols = []
            for col in ['Temperature at 2 Meters (C)', 'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)', 
                       'Wind Speed at 10 Meters (m/s)', 'satellite_energy_kwh', 'satellite_cell_temp_c', 
                       'satellite_temp_factor', 'satellite_soiling_factor']:
                if col in detailed_df.columns:
                    display_cols.append(col)
            
            if use_ml and 'ml_energy_kwh' in detailed_df.columns:
                for col in ['ml_energy_kwh', 'ml_cell_temp_c', 'ml_temp_factor', 'ml_soiling_factor']:
                    if col in detailed_df.columns:
                        display_cols.append(col)
            
            st.dataframe(detailed_df[display_cols].round(2), width="stretch")
            
            # Download button
            csv = detailed_df.to_csv()
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"solar_calculator_results_{selected_location.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with tab5:
            st.markdown("### 🧠 ML vs Satellite Comparison")
            
            if use_ml and 'ml_energy_kwh' in weather_df.columns:
                # Comparison metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Annual Energy Difference",
                        f"{annual_energy_ml - annual_energy_sat:,.0f} kWh",
                        delta=f"{((annual_energy_ml - annual_energy_sat) / annual_energy_sat * 100):.1f}%",
                        delta_color="normal" if annual_energy_ml > annual_energy_sat else "inverse"
                    )
                
                with col2:
                    st.metric(
                        "Daily Average Difference",
                        f"{(annual_energy_ml - annual_energy_sat) / 365:,.1f} kWh/day"
                    )
                
                with col3:
                    st.metric(
                        "NPV Difference",
                        f"${financial_ml['npv'] - financial_sat['npv']:,.2f}",
                        delta="Improved" if financial_ml['npv'] > financial_sat['npv'] else "Worse",
                        delta_color="normal" if financial_ml['npv'] > financial_sat['npv'] else "inverse"
                    )
                
                # Scatter plot
                fig5 = go.Figure()
                
                fig5.add_trace(go.Scatter(
                    x=weather_df['satellite_energy_kwh'],
                    y=weather_df['ml_energy_kwh'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='#1f77b4',
                        opacity=0.6,
                        line=dict(width=1, color='darkblue')
                    ),
                    name='ML vs Satellite'
                ))
                
                # Add diagonal line
                max_val = max(weather_df['satellite_energy_kwh'].max(), weather_df['ml_energy_kwh'].max())
                fig5.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Perfect Agreement',
                    line=dict(dash='dash', color='red')
                ))
                
                fig5.update_layout(
                    title="ML Predictions vs Satellite Data",
                    xaxis_title="Satellite Energy (kWh/day)",
                    yaxis_title="ML Energy (kWh/day)",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig5)
                
                # Error statistics
                st.markdown("### 📊 Error Statistics")
                
                error = weather_df['ml_energy_kwh'] - weather_df['satellite_energy_kwh']
                abs_error = np.abs(error)
                rel_error = (abs_error / (weather_df['satellite_energy_kwh'] + 0.01)) * 100
                
                error_stats = {
                    "Metric": [
                        "Mean Absolute Error (MAE)",
                        "Root Mean Square Error (RMSE)",
                        "Mean Bias Error (MBE)",
                        "Mean Absolute Percentage Error (MAPE)",
                        "Correlation Coefficient"
                    ],
                    "Value": [
                        f"{abs_error.mean():.3f} kWh/day",
                        f"{np.sqrt((error**2).mean()):.3f} kWh/day",
                        f"{error.mean():+.3f} kWh/day",
                        f"{rel_error.mean():.1f}%",
                        f"{weather_df['satellite_energy_kwh'].corr(weather_df['ml_energy_kwh']):.4f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(error_stats), hide_index=True, width="stretch")
                
            else:
                st.info("ML predictions were not enabled or models were not available. Enable ML in the sidebar to see comparison.")
        
        # Investment recommendation
        st.markdown("---")
        st.markdown("### 💡 Investment Recommendation")
        
        # Follow the user's selection in the Financial Analysis radio control.
        # `financial_results` and `energy_label` are assigned by that control above.
        selected_financial = financial_results
        model_name = energy_label
        selected_energy = (
            annual_energy_ml
            if model_name == "ML" and use_ml and 'ml_energy_kwh' in weather_df.columns
            else annual_energy_sat
        )
        
        if (selected_financial['npv'] > 0
                and selected_financial['payback_period'] is not None
                and selected_financial['payback_period'] < 10):
            st.success(f"""
            ✅ **This solar investment is financially viable!** (Based on {model_name} model)
            
            - **NPV**: ${selected_financial['npv']:,.2f} (Positive)
            - **Payback Period**: {selected_financial['payback_period']:.1f} years
            - **LCOE**: ${selected_financial['lcoe']:.3f}/kWh (Compare to current rate: ${electricity_rate:.2f}/kWh)
            
            The system will generate ${selected_energy * electricity_rate:,.2f} in annual savings.
            """)
        elif selected_financial['npv'] > 0:
            st.warning(f"""
            ⚠️ **Marginal investment - proceed with caution** (Based on {model_name} model)
            
            - **NPV**: ${selected_financial['npv']:,.2f} (Positive but low)
            - **Payback Period**: {selected_financial['payback_period']:.1f} years
            - **LCOE**: ${selected_financial['lcoe']:.3f}/kWh (Compare to current rate: ${electricity_rate:.2f}/kWh)
            
            Consider larger system size or lower costs to improve returns.
            """)
        else:
            st.error(f"""
            ❌ **This solar investment is not financially viable** (Based on {model_name} model)
            
            - **NPV**: ${selected_financial['npv']:,.2f} (Negative)
            - **LCOE**: ${selected_financial['lcoe']:.3f}/kWh (Compare to current rate: ${electricity_rate:.2f}/kWh)
            
            Try adjusting system size, reducing costs, or increasing electricity rate assumption.
            """)

        with tab6:
            st.markdown("### 📋 Nasa Results")
            
            # Combine all results
            detailed_df = nasa_df.copy()
            
            # Select columns for display
            display_cols = []
            for col in ['Temperature at 2 Meters (C)', 'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)', 
                       'Wind Speed at 10 Meters (m/s)', 'satellite_energy_kwh', 'satellite_cell_temp_c', 
                       'satellite_temp_factor', 'satellite_soiling_factor']:
                if col in detailed_df.columns:
                    display_cols.append(col)
            
            if use_ml and 'ml_energy_kwh' in detailed_df.columns:
                for col in ['ml_energy_kwh', 'ml_cell_temp_c', 'ml_temp_factor', 'ml_soiling_factor']:
                    if col in detailed_df.columns:
                        display_cols.append(col)
            
            st.dataframe(detailed_df[display_cols].round(2), width="stretch")
            
            # Download button
            csv = detailed_df.to_csv()
            st.download_button(
                label="📥 Download Nasa Results as CSV",
                data=csv,
                file_name=f"solar_calculator_results_{selected_location.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        with tab7:
            st.markdown("### 📋 OpenMeteo Results")
            
            # Combine all results
            detailed_df = openmeteo_df.copy()
            
            # Select columns for display
            display_cols = []
            for col in ['Max Temperature (°C)','Min Temperature (°C)','Mean Temperature (°C)','Max Apparent Temperature (°C)','Min Apparent Temperature (°C)','Mean Apparent Temperature (°C)',
                        'Sunrise (Time)','Sunset (Time)','Daylight Duration (s)','Sunshine Duration (s)','Shortwave Radiation (MJ/m²)','Total Precipitation (mm)',
                        'Max Precipitation Probability (%)','Total Showers (mm)','Total Rain (mm)','Total Snowfall (cm)','Precipitation Hours (h)','Max Wind Speed (km/h)',
                        'Max Wind Gusts (km/h)','Dominant Wind Direction (°)','ET₀ Evapotranspiration (mm)','Sunrise (Time) (Hours)','Sunset (Time) (Hours)']:
                if col in detailed_df.columns:
                    display_cols.append(col)
            
            st.dataframe(detailed_df[display_cols].round(2), width="stretch")
            
            # Download button
            csv = detailed_df.to_csv()
            st.download_button(
                label="📥 Download OpenMeteo Results as CSV",
                data=csv,
                file_name=f"solar_calculator_results_{selected_location.replace(' ', '_')}.csv",
                mime="text/csv"
            )


# ============================================================================
# Finalized Notebook Models and Corrected Financial Analysis
# ============================================================================

FINAL_MODEL_DIR = Path(__file__).resolve().parent / "regression_outputs" / "final_models"
FINAL_MODEL_FILES = {
    "predicted_max_temperature_c": "maximum_temperature_final_refit_bundle.joblib",
    "predicted_min_temperature_c": "minimum_temperature_final_refit_bundle.joblib",
    "predicted_relative_humidity_pct": "relative_humidity_final_refit_bundle.joblib",
    "predicted_wind_speed_kts": "wind_speed_final_refit_bundle.joblib",
}


@st.cache_resource(show_spinner=False)
def load_final_model_bundles(model_dir=str(FINAL_MODEL_DIR)):
    model_path = Path(model_dir)
    required = list(FINAL_MODEL_FILES.values()) + [
        "daily_rainfall_final_refit_bundle.joblib",
        "dust_days_final_refit_bundle.joblib",
    ]
    missing = [filename for filename in required if not (model_path / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing final model bundles: {missing}")
    bundles = {
        output: joblib.load(model_path / filename)
        for output, filename in FINAL_MODEL_FILES.items()
    }
    bundles["rainfall"] = joblib.load(model_path / "daily_rainfall_final_refit_bundle.joblib")
    bundles["dust"] = joblib.load(model_path / "dust_days_final_refit_bundle.joblib")
    return bundles


def _calendar_feature(dates, feature_name):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    month = dates.month.to_numpy()
    day_of_year = dates.dayofyear.to_numpy()
    mappings = {
        "Year": dates.year.to_numpy(), "Month": month, "Day": dates.day.to_numpy(),
        "DayOfYear": day_of_year, "Day_of_Year": day_of_year,
        "Quarter": dates.quarter.to_numpy(), "WeekOfYear": dates.isocalendar().week.to_numpy(dtype=float),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "day_of_year_sin_2": np.sin(4 * np.pi * day_of_year / 365.25),
        "day_of_year_cos_2": np.cos(4 * np.pi * day_of_year / 365.25),
        "time_index_years": (dates - pd.Timestamp("2015-01-01")).days.to_numpy() / 365.25,
        "wet_season": ((month >= 6) & (month <= 12)).astype(int),
    }
    return mappings.get(feature_name)


def prepare_final_model_features(weather_df, bundles):
    source = weather_df.copy().sort_index()
    dates = pd.to_datetime(source.index)
    required = sorted({feature for bundle in bundles.values() for feature in bundle["features"]})
    prepared = pd.DataFrame(index=source.index)
    fallback_features = []
    for feature in required:
        candidates = [feature]
        if feature.startswith("NASA_"):
            candidates.append(feature[5:])
        if feature.startswith("OM_"):
            candidates.append(feature[3:])
        available = next((candidate for candidate in candidates if candidate in source.columns), None)
        if available is not None:
            prepared[feature] = pd.to_numeric(source[available], errors="coerce")
            continue
        calendar = _calendar_feature(dates, feature)
        if calendar is not None:
            prepared[feature] = calendar
            continue
        prepared[feature] = [
            get_feature_value(row, feature) if get_feature_value(row, feature) is not None
            else get_default_value(feature)
            for _, row in source.iterrows()
        ]
        fallback_features.append(feature)
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.ffill().bfill()
    for column in prepared.columns[prepared.isna().any()]:
        prepared[column] = prepared[column].fillna(get_default_value(column))
    return prepared, fallback_features


def predict_final_weather_models_app(weather_df, bundles=None):
    bundles = bundles or load_final_model_bundles()
    features, fallback_features = prepare_final_model_features(weather_df, bundles)
    output = pd.DataFrame(index=weather_df.sort_index().index)
    for output_column in FINAL_MODEL_FILES:
        bundle = bundles[output_column]
        output[output_column] = bundle["model"].predict(features[bundle["features"]])

    rain_bundle = bundles["rainfall"]
    rain_x = features[rain_bundle["features"]]
    rain_probability = rain_bundle["occurrence_model"].predict_proba(rain_x)[:, 1]
    transformed_amount = rain_bundle["amount_model"].predict(rain_x)
    transformation = rain_bundle["amount_target_transformation"]
    if transformation == "raw":
        positive_amount = np.clip(transformed_amount, 0.0, None)
    elif transformation == "sqrt":
        positive_amount = np.square(np.clip(transformed_amount, 0.0, None))
    elif transformation == "log1p":
        positive_amount = np.clip(np.expm1(transformed_amount), 0.0, None)
    else:
        raise ValueError(f"Unsupported rainfall transformation: {transformation}")
    policy = "".join(c for c in str(rain_bundle["combination_policy"]).lower() if c.isalnum())
    threshold = float(rain_bundle["occurrence_reporting_threshold"])
    scale = float(rain_bundle.get("amount_scale", 1.0))
    if policy == "probabilityweightedexpectedamount":
        rainfall = rain_probability * positive_amount * scale
    elif policy == "thresholdgatedamount":
        rainfall = (rain_probability >= threshold) * positive_amount * scale
    else:
        raise ValueError(f"Unsupported rainfall policy: {rain_bundle['combination_policy']}")
    output["rain_occurrence_probability"] = rain_probability
    output["predicted_rain_day"] = (rain_probability >= threshold).astype(int)
    output["predicted_daily_rainfall_mm"] = rainfall

    dust_bundle = bundles["dust"]
    dust_probability = dust_bundle["model"].predict_proba(features[dust_bundle["features"]])[:, 1]
    dust_threshold = float(dust_bundle["classification_threshold"])
    output["dust_day_probability"] = dust_probability
    output["predicted_dust_day"] = (dust_probability >= dust_threshold).astype(int)
    return output, fallback_features


def calculate_irr(cashflows):
    cashflows = np.asarray(cashflows, dtype=float)
    periods = np.arange(len(cashflows), dtype=float)
    def value(rate):
        return np.sum(cashflows / np.power(1.0 + rate, periods))
    grid = np.concatenate([np.linspace(-0.99, -0.01, 100), np.linspace(0, 1, 201), np.linspace(1.05, 10, 180)])
    values = np.array([value(rate) for rate in grid])
    for left, right, left_value, right_value in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if left_value == 0:
            return left
        if np.sign(left_value) != np.sign(right_value):
            low, high, low_value = left, right, left_value
            for _ in range(100):
                midpoint = (low + high) / 2
                midpoint_value = value(midpoint)
                if abs(midpoint_value) < 1e-8:
                    return midpoint
                if np.sign(low_value) != np.sign(midpoint_value):
                    high = midpoint
                else:
                    low, low_value = midpoint, midpoint_value
            return (low + high) / 2
    return np.nan


def _fractional_payback(cumulative, annual):
    reached = np.flatnonzero(np.asarray(cumulative) >= 0)
    if not len(reached):
        return None
    index = int(reached[0])
    if index == 0:
        return 0.0
    previous = cumulative[index - 1]
    recovery = annual[index]
    return (index - 1) + np.clip(-previous / recovery, 0, 1) if recovery > 0 else float(index)


def calculate_financial_metrics(annual_energy, capex, o_and_m, electricity_rate,
                                discount_rate, system_lifetime=25,
                                degradation_rate=0.005, itc_rate=0.0, tax_rate=0.25,
                                electricity_escalation_rate=0.03,
                                om_inflation_rate=0.025):
    """Corrected notebook DCF: one ITC application, MACRS, escalations and discounted LCOE."""
    years = np.arange(1, int(system_lifetime) + 1)
    energy = annual_energy * np.power(1 - degradation_rate, years - 1)
    prices = electricity_rate * np.power(1 + electricity_escalation_rate, years - 1)
    revenue = energy * prices
    om = o_and_m * np.power(1 + om_inflation_rate, years - 1)
    itc_value = capex * itc_rate
    year_zero = -capex + itc_value
    macrs = np.array([0.20, 0.32, 0.192, 0.115, 0.115, 0.058])
    depreciation = np.zeros(len(years))
    count = min(len(years), len(macrs))
    depreciation[:count] = capex * (1 - 0.5 * itc_rate) * macrs[:count]
    taxable_income = revenue - om - depreciation
    taxes = np.maximum(taxable_income, 0) * tax_rate
    operating_cashflow = revenue - om - taxes
    project_cashflows = np.concatenate([[year_zero], operating_cashflow])
    discount_factors = np.power(1 + discount_rate, -years)
    discounted = np.concatenate([[year_zero], operating_cashflow * discount_factors])
    cumulative = np.cumsum(project_cashflows)
    cumulative_discounted = np.cumsum(discounted)
    pv_energy = np.sum(energy * discount_factors)
    pv_om = np.sum(om * discount_factors)
    lcoe = (capex - itc_value + pv_om) / pv_energy
    return {
        "years": years, "energy_production": energy, "revenue": revenue,
        "cashflow_at": operating_cashflow, "project_cashflows": project_cashflows,
        "discounted_cashflows": discounted, "npv": float(discounted.sum()),
        "irr": calculate_irr(project_cashflows),
        "payback_period": _fractional_payback(cumulative, project_cashflows),
        "discounted_payback_period": _fractional_payback(cumulative_discounted, discounted),
        "lcoe": lcoe, "unsubsidized_lcoe": (capex + pv_om) / pv_energy,
        "total_energy": energy.sum(), "total_revenue": revenue.sum(), "total_om": om.sum(),
        "net_profit": cumulative[-1], "annualized_return": cumulative[-1] / len(years),
        "itc_value": itc_value, "net_initial_investment": -year_zero,
    }

# ============================================================================
# Main Calculation Logic
# ============================================================================

if calculate:
    # Store calculation flag in session state
    st.session_state.calculated = True
    
    with st.spinner("Fetching weather data and calculating results..."):
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch NASA POWER data
        status_text.text("📡 Fetching NASA POWER data...")
        progress_bar.progress(10)
        
        end_date = datetime.now() - timedelta(days=31)  # Use data up to 31 days ago to avoid incomplete recent data
        start_date = end_date - timedelta(days=364)  # Last 365 days
        #start_date = end_date - timedelta(days=3)  # Last 4 days
        
        # Initialize the client with the working format
        nasa_client = NASAPOWERClient(lat, lon, community='RE')
        
        # Fetch PV-specific parameters
        nasa_df = nasa_client.fetch_daily_data(start_date, end_date, parameters=nasa_client.PARAMETERS)
        
        # Check if we got data
        if nasa_df.empty:
            st.warning("⚠️ NASA POWER API returned no data. Using Trinidad & Tobago climate defaults.")
            # Create fallback data
            nasa_df = nasa_client.create_fallback_data(start_date, end_date)
        else:
            st.success(f"✅ NASA POWER data loaded: {len(nasa_df)} days")
        
        progress_bar.progress(30)
        
        # Step 2: Fetch Open-Meteo data (as backup for sunshine duration)
        status_text.text("📡 Fetching Open-Meteo data...")
        
        openmeteo_client = OpenMeteoClient(lat, lon)
        openmeteo_df = openmeteo_client.fetch_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if openmeteo_df.empty:
            st.warning("🌴 Open-Meteo API unavailable. Using Trinidad & Tobago climate defaults.")
            openmeteo_df = openmeteo_client.create_fallback_data(start_date, end_date)
        else:
            st.success(f"✅ Open-Meteo data loaded: {len(openmeteo_df)} days")
        
        progress_bar.progress(50)

        # Step 3: Combine data - PRESERVE ORIGINAL COLUMN NAMES
        status_text.text("🔄 Processing weather data...")
        
        # Start with NASA data - use the original data as is
        if not nasa_df.empty:
            weather_df = nasa_df.copy(deep=True)
            print(f"   NASA data shape: {weather_df.shape}")
            print(f"   NASA columns: {list(weather_df.columns)}")
        else:
            weather_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
            print("   No NASA data available")
        
        # Add Open-Meteo data (preserving original column names)
        if not openmeteo_df.empty:
            for col in openmeteo_df.columns:
                if col not in weather_df.columns:
                    # Add the new column
                    weather_df[col] = openmeteo_df[col]
                    print(f"   Added column: {col}")
                else:
                    # If column exists, only fill NaN values where NASA has no data
                    nan_count_before = weather_df[col].isna().sum()
                    weather_df[col] = weather_df[col].fillna(openmeteo_df[col])
                    nan_count_after = weather_df[col].isna().sum()
                    if nan_count_before != nan_count_after:
                        print(f"   Filled {nan_count_before - nan_count_after} NaN values in {col}")
        
        print(f"   Combined columns: {list(weather_df.columns)}")
        
        # Don't reindex if we already have all dates
        # Instead, just ensure the index is sorted
        if not weather_df.empty:
            weather_df = weather_df.sort_index()
            print(f"   Weather df date range: {weather_df.index.min()} to {weather_df.index.max()}")
        
        print(f"   Final weather_df shape: {weather_df.shape}")
        print(f"   Final columns: {list(weather_df.columns)}")
        
        # Show the combined data sample
        with st.expander("🔍 Combined Weather Data Sample"):
            st.dataframe(weather_df.head(10))
        
        progress_bar.progress(60)
        
        # Step 4: Initialize PV Models
        status_text.text("☀️ Initializing PV performance models...")
        
        # Initialize models
        cell_temp_model = CellTemperatureModel(noct=45, reference_irradiance=800, reference_temp=20)
        solar_geometry = SolarGeometry(
            latitude=lat,
            longitude=lon,
            elevation_m=10,
            tilt=tilt,
            azimuth=azimuth
        )
        soiling_model = SoilingModel(accumulation_rate=0.003, cleaning_threshold=5.0, max_soiling=0.20)
        degradation_model = DegradationModel(
            baseline_degradation=0.005,
            baseline_uncertainty=0.002,
            temp_sensitivity=0.05,
            humidity_sensitivity=0.01,
            rainfall_sensitivity=0.005
        )
        
        pv_model = PVPerformanceModel(
            rated_power_kw=system_capacity,
            cell_temp_model=cell_temp_model,
            solar_geometry_model=solar_geometry,
            soiling_model=soiling_model,
            degradation_model=degradation_model,
            inverter_efficiency=0.96,
            system_loss=0.10,
            panel_coverage_factor=panel_coverage_factor
        )
        
        progress_bar.progress(65)
        
        # Step 5: Run PV Model with Satellite Data (Original NASA POWER columns)
        status_text.text("☀️ Calculating PV performance (Satellite data)...")
        
        # Extract inputs from original NASA POWER columns
        # Temperature - try various column names
        temp_cols = ['Temperature at 2 Meters (C)', 'T2M', 'temperature_2m_mean']
        temp_found = None
        for col in temp_cols:
            if col in weather_df.columns:
                temp_found = col
                break
        
        if temp_found:
            ambient_temp = weather_df[temp_found].values
        else:
            ambient_temp = np.ones(len(weather_df)) * 28
        
        # Irradiance
        irrad_cols = ['All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)', 
                      'ALLSKY_SFC_SW_DWN', 'shortwave_radiation_sum']
        irrad_found = None
        for col in irrad_cols:
            if col in weather_df.columns:
                irrad_found = col
                break
        
        if irrad_found:
            irradiance = weather_df[irrad_found].values
            # If irradiance is in W/m², convert to kWh/m²/day
            if np.mean(irradiance) > 100:
                irradiance = irradiance / 1000
        else:
            irradiance = np.ones(len(weather_df)) * 5.5
        
        # Wind speed
        wind_cols = ['Wind Speed at 10 Meters (m/s)', 'WS10M', 'wind_speed_10m_max']
        wind_found = None
        for col in wind_cols:
            if col in weather_df.columns:
                wind_found = col
                break
        
        if wind_found:
            wind_speed = weather_df[wind_found].values
        else:
            wind_speed = np.ones(len(weather_df)) * 5
        
        # Precipitation
        rain_cols = ['Precipitation Corrected (mm/day)', 'PRECTOTCORR', 'precipitation_sum']
        rain_found = None
        for col in rain_cols:
            if col in weather_df.columns:
                rain_found = col
                break
        
        if rain_found:
            rainfall = weather_df[rain_found].values
        else:
            rainfall = np.zeros(len(weather_df))
        
        # Humidity
        humidity_cols = ['Relative Humidity at 2 Meters (%)', 'RH2M']
        humidity_found = None
        for col in humidity_cols:
            if col in weather_df.columns:
                humidity_found = col
                break
        
        if humidity_found:
            humidity = weather_df[humidity_found].values
        else:
            humidity = np.ones(len(weather_df)) * 75
        
        # Sunshine duration
        sun_cols = ['sunshine_duration']
        sun_found = None
        for col in sun_cols:
            if col in weather_df.columns:
                sun_found = col
                break
        
        if sun_found:
            sunshine_duration = weather_df[sun_found].values
        else:
            sunshine_duration = np.ones(len(weather_df)) * 43200  # 12 hours
        
        # Ensure no NaN values
        ambient_temp = np.nan_to_num(ambient_temp, nan=28.0)
        irradiance = np.nan_to_num(irradiance, nan=5.5)
        wind_speed = np.nan_to_num(wind_speed, nan=5.0)
        rainfall = np.nan_to_num(rainfall, nan=0.0)
        humidity = np.nan_to_num(humidity, nan=75.0)
        sunshine_duration = np.nan_to_num(sunshine_duration, nan=43200)
        
        dates = weather_df.index.values
        
        # Run model with satellite data
        results_satellite = pv_model.calculate_daily_energy(
            ambient_temp=ambient_temp,
            irradiance=irradiance,
            wind_speed=wind_speed,
            rainfall=rainfall,
            dust_days=np.zeros(len(weather_df)),  # No dust data from satellite
            dates=dates,
            sunshine_duration=sunshine_duration,
            humidity=humidity,
            system_age_years=0
        )
        
        # Store satellite results
        weather_df['satellite_energy_kwh'] = results_satellite['energy_kwh']
        weather_df['satellite_cell_temp_c'] = results_satellite['cell_temp_c']
        weather_df['satellite_temp_factor'] = results_satellite['temp_factor']
        weather_df['satellite_soiling_factor'] = results_satellite['soiling_factor']
        
        progress_bar.progress(75)
        
        # Step 6: Apply ML Predictions if enabled
        if use_ml:
            status_text.text("🧠 Loading ML models and making predictions...")
            
            # Load finalized full-refit bundles and predict the complete year in one call.
            try:
                final_predictions, fallback_features = predict_final_weather_models_app(weather_df)
                weather_df = weather_df.join(final_predictions)
                if fallback_features:
                    st.warning(
                        f"{len(fallback_features)} trained features were unavailable from the APIs and "
                        "used documented climate defaults. Review the feature diagnostics below."
                    )
                status_text.text("Calculating PV performance from finalized ML predictions...")
                results_ml = pv_model.calculate_daily_energy(
                    ambient_temp=weather_df["predicted_max_temperature_c"].to_numpy(float),
                    irradiance=irradiance,
                    wind_speed=weather_df["predicted_wind_speed_kts"].to_numpy(float) * 0.514444,
                    rainfall=weather_df["predicted_daily_rainfall_mm"].to_numpy(float),
                    dust_days=weather_df["predicted_dust_day"].to_numpy(int),
                    dust_prob=weather_df["dust_day_probability"].to_numpy(float),
                    dates=dates,
                    sunshine_duration=sunshine_duration,
                    humidity=weather_df["predicted_relative_humidity_pct"].to_numpy(float),
                    system_age_years=0,
                )
                weather_df["ml_energy_kwh"] = results_ml["energy_kwh"]
                weather_df["ml_cell_temp_c"] = results_ml["cell_temp_c"]
                weather_df["ml_temp_factor"] = results_ml["temp_factor"]
                weather_df["ml_soiling_factor"] = results_ml["soiling_factor"]
                with st.expander("Final model diagnostics"):
                    st.write("Model directory:", str(FINAL_MODEL_DIR))
                    st.write("Fallback/default features:", fallback_features or "None")
                    st.dataframe(final_predictions.tail().round(3))
                progress_bar.progress(90)
            except Exception as error:
                st.error(f"Finalized ML inference failed: {error}")
                st.info("Satellite pathway remains available; no legacy model substitution was made.")
                use_ml = False

        # Step 8: Calculate financial metrics
        status_text.text("💰 Calculating financial metrics...")
        
        # Satellite financial metrics
        annual_energy_sat = weather_df['satellite_energy_kwh'].sum()
        financial_sat = calculate_financial_metrics(
            annual_energy=annual_energy_sat,
            capex=capex,
            o_and_m=o_and_m,
            electricity_rate=electricity_rate,
            discount_rate=discount_rate,
            system_lifetime=system_lifetime,
            degradation_rate=annual_degradation_rate,
            itc_rate=itc_rate,
            tax_rate=tax_rate,
            electricity_escalation_rate=electricity_escalation_rate,
            om_inflation_rate=om_inflation_rate
        )
        
        # ML financial metrics
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            annual_energy_ml = weather_df['ml_energy_kwh'].sum()
            financial_ml = calculate_financial_metrics(
                annual_energy=annual_energy_ml,
                capex=capex,
                o_and_m=o_and_m,
                electricity_rate=electricity_rate,
                discount_rate=discount_rate,
                system_lifetime=system_lifetime,
                degradation_rate=annual_degradation_rate,
                itc_rate=itc_rate,
                tax_rate=tax_rate,
                electricity_escalation_rate=electricity_escalation_rate,
                om_inflation_rate=om_inflation_rate
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Calculation complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # ========================================================================
        # Store ALL results in session state for persistence
        # ========================================================================
        
        st.session_state.weather_df = weather_df
        st.session_state.nasa_df = nasa_df
        st.session_state.openmeteo_df = openmeteo_df
        st.session_state.annual_energy_sat = annual_energy_sat
        st.session_state.financial_sat = financial_sat
        st.session_state.use_ml = use_ml
        
        if use_ml and 'ml_energy_kwh' in weather_df.columns:
            st.session_state.annual_energy_ml = annual_energy_ml
            st.session_state.financial_ml = financial_ml
        
        st.session_state.calculated = True
        
        # ========================================================================
        # Display Results
        # ========================================================================
        
        # Render the persistent map/results branch on a clean rerun. Without
        # this, the first calculation bypasses the overview map until another
        # widget (such as the financial-model radio) triggers a rerun.
        st.rerun()



else:
    # Show default information when no calculation has been run
    st.markdown("---")
    st.markdown("""
    ### ☀️ Welcome to the Solar Investment Calculator!
    
    This tool helps you estimate the performance and financial returns of a solar PV system in Trinidad and Tobago.
    
    **How it works:**
    1. 📍 **Select your location** from the sidebar (or enter custom coordinates)
    2. ⚡ **Enter your system specifications** (capacity, tilt, azimuth)
    3. 💰 **Input your financial parameters** (costs, electricity rate, discount rate)
    4. 🚀 **Click 'Calculate'** to see the results
    
    **What you'll get:**
    - 📈 Daily energy production estimates (Satellite + ML predictions)
    - 💰 Financial metrics (NPV, payback period, LCOE)
    - 🌤️ Weather data for your location
    - 📋 Detailed results and downloadable CSV
    - 🧠 ML vs Satellite comparison (when ML enabled)
    
    Data is pulled from NASA POWER and Open-Meteo APIs for the last 365 days.
    """)
    
    # Sample location on map
    st.markdown("### 🗺️ Trinidad & Tobago Map")
    m_full = folium.Map(location=[10.5, -61.3], zoom_start=8, control_scale=True)
    
    # Add markers for major locations
    for name, coords in location_options.items():
        if coords["lat"] is not None:
            folium.Marker(
                [coords["lat"], coords["lon"]],
                popup=name,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m_full)
    
    st_folium(
        m_full,
        key="overview_location_map",
        height=430,
        use_container_width=True,
        returned_objects=[],
    )

# ========================================================================
# Check if calculation has been done before
# ========================================================================

# Initialize session state
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False

    # If calculation has been done, display results without re-fetching
    if st.session_state.calculated:
        display_results()

# ============================================================================
# Footer
# ============================================================================

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
        <p>Data sourced from NASA POWER and Open-Meteo APIs</p>
        <p>ML Models trained on TTMS data and validated against PVDAQ data</p>
        <p>© 2026 Solar Investment Calculator - Trinidad & Tobago</p>
    </div>
    """, unsafe_allow_html=True)
