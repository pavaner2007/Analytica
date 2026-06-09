import streamlit as st
import streamlit.components.v1 as components



st.set_page_config(

    page_title="Intelligent Data Suite",

    layout="wide",

    page_icon="�"

)



import pandas as pd

import numpy as np

from sklearn.impute import KNNImputer

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import xgboost as xgb

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sns

import matplotlib.pyplot as plt

import pickle

import json

from datetime import datetime

import time

import io

from scipy import stats

import base64



# Custom CSS for advanced modern styling

st.markdown("""

<style>

    /* Import Google Fonts */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    

    * {

        font-family: 'Inter', sans-serif;

    }

    

    /* Theme Colors */

    :root {

        --primary: #6366f1;

        --primary-dark: #4f46e5;

        --secondary: #8b5cf6;

        --accent: #ec4899;

        --success: #10b981;

        --warning: #f59e0b;

        --error: #ef4444;

        --bg-gradient-start: #667eea;

        --bg-gradient-end: #764ba2;

    }

    

    /* Main background with modern gradient */

    .main {

        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);

        background-attachment: fixed;

    }

    

    /* Glassmorphism sidebar */

    [data-testid="stSidebar"] {

        background: rgba(255, 255, 255, 0.1);

        backdrop-filter: blur(20px);

        -webkit-backdrop-filter: blur(20px);

        border-right: 1px solid rgba(255, 255, 255, 0.2);

        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.1);

    }

    

    [data-testid="stSidebar"] > div:first-child {

        background: transparent;

    }

    

    /* Modern Button Styling */

    div.stButton > button {

        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);

        color: white;

        border-radius: 12px;

        border: none;

        padding: 12px 28px;

        font-weight: 600;

        font-size: 14px;

        letter-spacing: 0.3px;

        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);

        position: relative;

        overflow: hidden;

    }

    

    div.stButton > button:hover {

        transform: translateY(-2px);

        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);

    }

    

    div.stButton > button:active {

        transform: translateY(0);

    }

    

    /* Glassmorphism Cards */

    .glass-card {

        background: rgba(255, 255, 255, 0.95);

        backdrop-filter: blur(10px);

        -webkit-backdrop-filter: blur(10px);

        border-radius: 20px;

        border: 1px solid rgba(255, 255, 255, 0.3);

        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);

        padding: 24px;

        margin: 16px 0;

    }

    

    /* Enhanced Metrics */

    [data-testid="stMetricValue"] {

        font-size: 32px;

        font-weight: 700;

        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

    }

    

    [data-testid="stMetricLabel"] {

        font-size: 14px;

        font-weight: 600;

        color: #64748b;

        text-transform: uppercase;

        letter-spacing: 0.5px;

    }

    

    /* Modern Headers */

    h1 {

        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        font-weight: 800;

        text-align: center;

        padding: 24px 0;

        animation: fadeInDown 0.8s ease-out;

        letter-spacing: -1px;

    }

    

    h2 {

        color: var(--primary);

        font-weight: 700;

        margin-top: 32px;

        margin-bottom: 16px;

    }

    

    h3 {

        color: var(--primary-dark);

        font-weight: 600;

    }

    

    /* Enhanced File Uploader */

    [data-testid="stFileUploader"] {

        background: rgba(255, 255, 255, 0.95);

        backdrop-filter: blur(10px);

        border-radius: 20px;

        padding: 32px;

        border: 2px dashed rgba(99, 102, 241, 0.3);

        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);

        transition: all 0.3s ease;

    }

    

    [data-testid="stFileUploader"]:hover {

        border-color: var(--primary);

        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);

    }

    

    /* Modern DataFrame */

    .dataframe {

        border-radius: 16px;

        overflow: hidden;

        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);

        border: 1px solid rgba(99, 102, 241, 0.1);

    }

    

    /* Enhanced Input Fields */

    .stTextInput > div > div > input,

    .stSelectbox > div > div > select,

    .stMultiSelect > div > div {

        border-radius: 12px;

        border: 2px solid rgba(99, 102, 241, 0.2);

        padding: 12px 16px;

        font-size: 14px;

        transition: all 0.3s ease;

        background: rgba(255, 255, 255, 0.9);

        color: #1e293b !important;

        font-weight: 500;

    }

    

    .stTextInput > div > div > input::placeholder {

        color: #94a3b8;

    }

    

    .stTextInput > div > div > input:focus,

    .stSelectbox > div > div > select:focus {

        border-color: var(--primary);

        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);

        outline: none;

        color: #0f172a !important;

    }

    

    /* Modern Radio Buttons */

    .stRadio > label {

        background: rgba(255, 255, 255, 0.7);

        backdrop-filter: blur(10px);

        padding: 12px 16px;

        border-radius: 12px;

        margin: 6px 0;

        border: 1px solid rgba(255, 255, 255, 0.3);

        transition: all 0.3s ease;

        cursor: pointer;

    }

    

    .stRadio > label:hover {

        background: rgba(255, 255, 255, 0.9);

        transform: translateX(4px);

    }

    

    /* Alert Styling */

    .stSuccess, .stInfo, .stWarning, .stError {

        border-radius: 12px;

        padding: 16px 20px;

        animation: slideInRight 0.4s ease-out;

        backdrop-filter: blur(10px);

    }

    

    /* Advanced Animations */

    @keyframes fadeInDown {

        from {

            opacity: 0;

            transform: translateY(-20px);

        }

        to {

            opacity: 1;

            transform: translateY(0);

        }

    }

    

    @keyframes slideInRight {

        from {

            opacity: 0;

            transform: translateX(30px);

        }

        to {

            opacity: 1;

            transform: translateX(0);

        }

    }

    

    @keyframes pulse {

        0%, 100% {

            opacity: 1;

        }

        50% {

            opacity: 0.8;

        }

    }

    

    /* Progress Bar */

    .stProgress > div > div > div > div {

        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);

        border-radius: 8px;

    }

    

    /* Tabs Enhancement */

    .stTabs [data-baseweb="tab-list"] {

        gap: 12px;

        background: rgba(255, 255, 255, 0.5);

        padding: 8px;

        border-radius: 16px;

    }

    

    .stTabs [data-baseweb="tab"] {

        border-radius: 12px;

        padding: 12px 24px;

        background: transparent;

        font-weight: 600;

        transition: all 0.3s ease;

    }

    

    .stTabs [aria-selected="true"] {

        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);

        color: white !important;

        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);

    }

    

    /* Scrollbar Styling */

    ::-webkit-scrollbar {

        width: 10px;

        height: 10px;

    }

    

    ::-webkit-scrollbar-track {

        background: rgba(255, 255, 255, 0.1);

        border-radius: 10px;

    }

    

    ::-webkit-scrollbar-thumb {

        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);

        border-radius: 10px;

    }

    

    ::-webkit-scrollbar-thumb:hover {

        background: var(--primary-dark);

    }

    

    /* Download Button Enhancement */

    .stDownloadButton > button {

        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);

        border-radius: 12px;

        font-weight: 600;

        padding: 12px 24px;

        transition: all 0.3s ease;

    }

    

    .stDownloadButton > button:hover {

        transform: translateY(-2px);

        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);

    }

</style>

""", unsafe_allow_html=True)



# voice features

voice_enabled = False

try:

    from gtts import gTTS # tts

    import pyttsx3

    import speech_recognition as sr

    voice_enabled = True

except:

    pass



# Session State Initialization (MUST be before navigation)

if 'df' not in st.session_state:

    st.session_state.df = None

if 'df_filtered' not in st.session_state:

    st.session_state.df_filtered = None

if 'logs' not in st.session_state:

    st.session_state.logs = []

if 'query_input' not in st.session_state:

    st.session_state.query_input = ""

if 'theme' not in st.session_state:

    st.session_state.theme = "Default"

if 'ml_models' not in st.session_state:

    st.session_state.ml_models = {}

if 'transformations' not in st.session_state:

    st.session_state.transformations = []

if 'current_step' not in st.session_state:

    st.session_state.current_step = "Home Dashboard"



# Sidebar Navigation

st.sidebar.markdown("""

<div style='text-align: center; padding: 24px 0 16px 0;'>

    <h2 style='color: white; font-size: 1.6em; font-weight: 700; text-shadow: 0 2px 8px rgba(0,0,0,0.2);'>

        Navigation

    </h2>

</div>

""", unsafe_allow_html=True)



# List of navigation options

nav_options = [

    "Home Dashboard",

    "Upload & Assessment",

    "Data Cleaning",

    "Filtering & Selection",

    "Visualizations",

    "ML Predictions",

    "Reports & Export",

    "Session Management",

    "Audit Trail"

]



# Get the index of current step

try:

    current_index = nav_options.index(st.session_state.current_step)

except:

    current_index = 0

    st.session_state.current_step = nav_options[0]



step = st.sidebar.radio("Navigation", nav_options, 

                        label_visibility="collapsed", 

                        index=current_index)



# Update current step

st.session_state.current_step = step



# Sidebar theme selector

st.sidebar.markdown("---")

st.sidebar.markdown("""

<div style='background: rgba(255,255,255,0.15); padding: 16px; border-radius: 12px; backdrop-filter: blur(10px);'>

    <p style='color: white; text-align: center; margin: 0; font-weight: 600; font-size: 14px;'>

        QUICK STATS

    </p>

</div>

""", unsafe_allow_html=True)



theme_choice = st.sidebar.selectbox("Theme", ["Default", "Light", "Dark"], index=["Default", "Light", "Dark"].index(st.session_state.theme))

st.session_state.theme = theme_choice



st.sidebar.markdown("---")

st.sidebar.markdown("### Quick Navigation")

if st.sidebar.button("🏠 Home"):

    st.session_state.current_step = "Home Dashboard"

    st.rerun()

if st.sidebar.button("📤 Upload"):

    st.session_state.current_step = "Upload & Assessment"

    st.rerun()

if st.sidebar.button("🧹 Clean"):

    st.session_state.current_step = "Data Cleaning"

    st.rerun()

if st.sidebar.button("📈 Visualize"):

    st.session_state.current_step = "Visualizations"

    st.rerun()



if st.session_state.df is not None:

    col_a, col_b = st.sidebar.columns(2)

    with col_a:

        st.metric("Rows", f"{st.session_state.df.shape[0]:,}")

        st.metric("Actions", len(st.session_state.logs))

    with col_b:

        st.metric("Columns", st.session_state.df.shape[1])

        if st.session_state.ml_models:

            st.metric("Models", len(st.session_state.ml_models))

else:

    st.sidebar.info("📊 Upload data to see stats")



# Helper Functions

def log_action(action, details=""):

    # log actions

    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

             "action": action, "details": details}

    st.session_state.logs.append(entry)

    with open("audit_log.json","a") as f:

        f.write(json.dumps(entry)+"\n")

        



def load_sample_data():

    # Interactive sample dataset for fast exploration

    sample = pd.DataFrame({

        "Category": np.random.choice(["A", "B", "C"], 80),

        "Sales": np.random.randint(120, 480, 80),

        "Profit": np.round(np.random.uniform(10, 120, 80), 2),

        "Region": np.random.choice(["North", "South", "East", "West"], 80),

        "Date": pd.date_range(start="2025-01-01", periods=80, freq="D")

    })

    return sample





def calculate_quality(df):

    # calculate data quality

    total_cols = df.shape[1]

    total_rows = df.shape[0]

    

    missing_pct = df.isnull().sum().sum() / (total_rows*total_cols) * 100

    duplicate_pct = df.duplicated().sum() / total_rows * 100



    numeric_cols = df.select_dtypes(include=np.number)

    if not numeric_cols.empty:

        z = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std(ddof=0))

        outlier_pct = (z > 3).sum().sum() / np.prod(numeric_cols.shape) * 100

    else:

        outlier_pct = 0



    total_quality = 100 - (missing_pct*0.5 + duplicate_pct*0.2 + outlier_pct*0.3)

    return round(total_quality,2), round(missing_pct,2), round(duplicate_pct,2), round(outlier_pct,2)



def speak_text(text):

    if voice_enabled:

        engine = pyttsx3.init()

        engine.say(text)

        engine.runAndWait()



def remove_outliers(df, columns, method='iqr'):

    # remove outliers

    df_clean = df.copy()

    for col in columns:

        if method == 'iqr':

            Q1 = df_clean[col].quantile(0.25)

            Q3 = df_clean[col].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR

            upper_bound = Q3 + 1.5 * IQR

            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        else:  # z-score

            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))

            df_clean = df_clean[z_scores < 3]

    return df_clean



def apply_transformation(df, columns, transform_type):

    # apply transformations

    try:

        df_transformed = df.copy()

        for col in columns:

            if transform_type == 'log' or transform_type == 'Log':

                # Check if column has positive values

                if (df_transformed[col] <= 0).any():

                    st.warning(f"⚠️ Column '{col}' contains non-positive values. Using log1p which handles this.")

                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])

            elif transform_type == 'sqrt' or transform_type == 'Square Root':

                df_transformed[f'{col}_sqrt'] = np.sqrt(df_transformed[col].abs())

            elif transform_type == 'square' or transform_type == 'Square':

                df_transformed[f'{col}_squared'] = df_transformed[col] ** 2

            elif transform_type == 'standardize' or transform_type == 'Standardize (Z-score)':

                scaler = StandardScaler()

                df_transformed[f'{col}_std'] = scaler.fit_transform(df_transformed[[col]])

            elif transform_type == 'normalize' or transform_type == 'Normalize (0-1)':

                scaler = MinMaxScaler()

                df_transformed[f'{col}_norm'] = scaler.fit_transform(df_transformed[[col]])

        return df_transformed, True

    except Exception as e:

        st.error(f"Error in transformation: {str(e)}")

        return df, False



def generate_insights(df):

    insights = []

    

    # basic stats

    insights.append(f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns")

    

    # missing data

    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100

    if missing_pct > 10:

        insights.append(f"⚠️ High missing data: {missing_pct:.1f}% of values are missing")

    elif missing_pct > 0:

        insights.append(f"ℹ️ {missing_pct:.1f}% of values are missing")

    

    # Numeric columns

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        # Find highly correlated features

        corr_matrix = df[numeric_cols].corr().abs()

        high_corr = []

        for i in range(len(corr_matrix.columns)):

            for j in range(i+1, len(corr_matrix.columns)):

                if corr_matrix.iloc[i, j] > 0.8:

                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        

        if high_corr:

            insights.append(f"Found {len(high_corr)} highly correlated feature pairs (>0.8)")

    

    # Duplicates

    dup_count = df.duplicated().sum()

    if dup_count > 0:

        insights.append(f"🔄 {dup_count} duplicate rows detected ({dup_count/len(df)*100:.1f}%)")

    

    # Categorical columns

    cat_cols = df.select_dtypes(include='object').columns

    if len(cat_cols) > 0:

        high_card = [col for col in cat_cols if df[col].nunique() > 50]

        if high_card:

            insights.append(f"📋 {len(high_card)} categorical columns with high cardinality (>50 unique values)")

    

    return insights



# HOME DASHBOARD

if step == "Home Dashboard":

    components.html("""
<div class='glass-card' style='text-align: center; margin: 20px auto; max-width: 900px; background: linear-gradient(135deg, #4f46e5 0%, #8b5cf6 45%, #ec4899 100%); color: #fff; border: none; box-shadow: 0 20px 60px rgba(0,0,0,0.25); padding: 40px 24px; border-radius: 24px;'>

    <h1 style='font-size: 4.4em; margin-bottom: 12px; letter-spacing: -2px; text-transform: uppercase; color: #ffffff;'>

        Intelligent Data Suite

    </h1>

    <p style='font-size: 1.3em; color: rgba(255,255,255,0.95); font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-top: 0;'>

        Advanced Analytics - ML Models - AI-Powered Insights

    </p>

</div>

""", height=260, scrolling=False)

    

    if st.session_state.df is not None:

        # Quick Stats Overview

        st.markdown("""

<div class='glass-card'>

    <h3 style='text-align: center; color: #1e293b; margin-bottom: 20px;'>Dataset Overview</h3>

</div>

        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        

        with col1:

            st.metric("Total Rows", f"{st.session_state.df.shape[0]:,}", 

                     help="Number of records in dataset")



    else:

        st.warning("Please upload a dataset first!")



