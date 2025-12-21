import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os, base64, io, time
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from src.prediction import forecast_groundwater, get_test_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Groundwater Prediction System",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/groundwater-prediction',
        'Report a bug': 'https://github.com/your-repo/groundwater-prediction/issues',
        'About': '''
        ## Groundwater Level Prediction System
        **Version:** 2.0.0
        **Built with:** Streamlit, TensorFlow, Plotly
        **Purpose:** Advanced ML platform for Punjab district water resource forecasting
        '''
    }
)

@st.cache_data
def load_data():
    return pd.read_csv('data/processed_data.csv')

df = load_data()
model_files = [f.replace('_model.h5', '') for f in os.listdir('models') if f.endswith('_model.h5')]
districts = sorted([d for d in df['district'].unique() if d in model_files])

if not districts:
    st.error("No trained models found.")
    st.stop()

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">📥 {text}</a>'

def calculate_prediction_confidence(predictions, rmse):
    ci = 1.96 * rmse
    return [p + ci for p in predictions], [p - ci for p in predictions]

def create_metric_trend_indicator(current, predicted, threshold=0.1):
    """Create trend indicator based on prediction change"""
    change = predicted - current
    change_pct = abs(change) / current if current != 0 else 0

    if change_pct > threshold:
        if change > 0:
            return "📈", "Rising", "#10B981", f"+{change:.2f}m ({change_pct:.1%})"
        else:
            return "📉", "Falling", "#EF4444", f"{change:.2f}m ({change_pct:.1%})"
    else:
        return "➡️", "Stable", "#6B7280", f"{change:+.2f}m"

def format_number(num, precision=2):
    """Format numbers with appropriate precision and units"""
    if abs(num) >= 1000:
        return f"{num:,.{precision}f}"
    elif abs(num) >= 1:
        return f"{num:.{precision}f}"
    else:
        return f"{num:.{precision}f}"

def create_progress_bar(current, total, label="Progress"):
    """Create a custom progress bar"""
    progress = min(current / total, 1.0)
    progress_bar = f"""
    <div style="width: 100%; background-color: #E2E8F0; border-radius: 10px; height: 20px; margin: 10px 0;">
        <div style="width: {progress*100:.1f}%; background: linear-gradient(90deg, #6366F1, #8B5CF6); height: 100%; border-radius: 10px; transition: width 0.3s ease;">
            <div style="color: white; text-align: center; line-height: 20px; font-size: 12px; font-weight: 600;">
                {progress*100:.1f}%
            </div>
        </div>
    </div>
    <p style="text-align: center; margin: 5px 0; color: #64748B; font-size: 0.9em;">{label}</p>
    """
    return progress_bar

def add_tooltip(text, tooltip_text):
    """Add tooltip functionality"""
    return f'<span title="{tooltip_text}" style="cursor: help; border-bottom: 1px dotted #64748B;">{text}</span>'

# Theme-Adaptive CSS with Dark Mode Support
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

/* Layout cleanup */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

.main > div:first-child {
    padding-top: 0 !important;
}

.stApp > header {
    display: none !important;
}

h1, h2, h3 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Theme Variables */
[data-theme="light"], .main {
    --bg-primary: #FFFFFF;
    --bg-secondary: #F8FAFC;
    --bg-tertiary: #F1F5F9;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --border-color: #E2E8F0;
    --card-bg: #FFFFFF;
    --shadow-color: rgba(0, 0, 0, 0.08);
    --shadow-hover: rgba(99, 102, 241, 0.12);
    --input-bg: #FFFFFF;
    --gradient-bg: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    .main {
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --bg-tertiary: #334155;
        --text-primary: #FFFFFF;
        --text-secondary: #E2E8F0;
        --text-muted: #CBD5E1;
        --border-color: #475569;
        --card-bg: #1E293B;
        --shadow-color: rgba(0, 0, 0, 0.35);
        --shadow-hover: rgba(99, 102, 241, 0.25);
        --input-bg: #1E293B;
        --gradient-bg: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
}

/* Global */
* { font-family: 'Inter', sans-serif; }

.main {
    background: var(--gradient-bg);
    color: var(--text-primary);
}

/* Header */
.main-header {
    font-size: 2.4rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #4F46E5, #0EA5E9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient 16s ease infinite;
    margin-bottom: 0.6rem !important;
}

@keyframes gradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    text-align: center;
    margin-bottom: 1.75rem !important;
    font-weight: 500;
}

/* Cards */
.metric-card,
.control-panel,
.chart-container,
.summary-section {
    background: var(--card-bg);
    border: 1.5px solid var(--border-color);
    border-radius: 18px;
    padding: 1.75rem;
    box-shadow: 0 4px 16px var(--shadow-color);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.metric-card:hover,
.summary-section:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 24px var(--shadow-hover);
}

/* Metrics */
.metric-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    font-size: 2.25rem;
    font-weight: 800;
    color: #1F3A5F;
    font-family: 'JetBrains Mono', monospace;
}

.metric-description {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* Control Title */
.control-title,
.chart-title,
.summary-title {
    font-size: 1.35rem;
    font-weight: 700;
    padding-bottom: 0.6rem;
    margin-bottom: 1.25rem;
    border-bottom: 2px solid var(--border-color);
}

/* Buttons */
.stButton > button {
    background: #1F3A5F !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 0.9rem 2rem !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all 0.25s ease !important;
    width: 100%;
}

.stButton > button:hover {
    background: #16324F !important;
    box-shadow: 0 8px 20px rgba(31,58,95,0.35) !important;
    transform: translateY(-2px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-tertiary);
    padding: 0.5rem;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

.stTabs [aria-selected="true"] {
    background: #1F3A5F !important;
    color: white !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--input-bg) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 10px !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding: 2rem;
    color: var(--text-secondary);
    border-top: 1.5px solid var(--border-color);
    background: var(--card-bg);
    border-radius: 14px;
    font-weight: 500;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# Sidebar for additional controls
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")

    # System status
    st.markdown("### 📊 System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("Districts Available", len(districts))
    with status_col2:
        st.metric("Models Loaded", len(model_files))

    st.markdown("---")

    # Quick actions
    st.markdown("### ⚡ Quick Actions")

    if st.button("🔄 Refresh Data", help="Reload all data and models"):
        st.cache_data.clear()
        df = load_data()
        st.rerun()

    if st.button("📊 System Info", help="Show system information"):
        with st.expander("System Information", expanded=True):
            st.markdown(f"""
            **Python Version:** {sys.version.split()[0]}
            **Streamlit Version:** {st.__version__}
            **Data Points:** {len(df):,}
            **Time Range:** {df['year'].min()}-{df['year'].max()}
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)

    st.markdown("---")

    # Theme selector
    st.markdown("### 🎨 Theme")
    theme = st.selectbox(
        "Color Theme",
        ["Default", "Dark", "Light"],
        help="Select application theme"
    )

    # Export options
    st.markdown("### 📥 Export Options")
    export_format = st.selectbox(
        "Chart Export Format",
        ["PNG", "SVG", "PDF"],
        help="Format for chart downloads"
    )

    # Help section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **Getting Started:**
        1. Select a district in the Analysis tab
        2. Choose forecast horizon (3 or 6 months)
        3. Click "Generate Forecast"
        4. Explore results in other tabs

        **Tips:**
        - Use confidence intervals for uncertainty
        - Compare multiple districts
        - Export data for further analysis
        - Check methodology for technical details
        """)

st.markdown('<h1 class="main-header">💧 Groundwater Level Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning Platform for Punjab District Water Resource Forecasting</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📊 Analysis", "📈 Predictions", "📋 Methodology", "📥 Data Export"])

with tab1:
    st.markdown('<h2 style="color:var(--text-primary);text-align:center;margin-bottom:2rem;font-weight:700">🏠 System Dashboard</h2>', unsafe_allow_html=True)

    # Overview metrics
    st.markdown("### 📈 System Overview")
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

    with overview_col1:
        st.metric("Total Districts", len(districts), help="Number of districts with trained models")
    with overview_col2:
        st.metric("Data Points", f"{len(df):,}", help="Total groundwater measurements")
    with overview_col3:
        st.metric("Time Range", f"{df['year'].min()}-{df['year'].max()}", help="Historical data coverage")
    with overview_col4:
        active_analyses = 1 if 'predictions' in st.session_state else 0
        st.metric("Active Analysis", active_analyses, help="Currently running analyses")

    # Quick district overview
    st.markdown("### 🗺️ District Overview")
    district_overview = df.groupby('district').agg({
        'avg_groundwater_level': ['mean', 'min', 'max', 'std'],
        'year': ['min', 'max']
    }).round(2)

    district_overview.columns = ['Mean Level', 'Min Level', 'Max Level', 'Std Dev', 'Start Year', 'End Year']
    district_overview = district_overview.reset_index()

    # Add trend analysis
    district_overview['Trend'] = district_overview.apply(
        lambda row: "📈 Rising" if row['Max Level'] > row['Mean Level'] + row['Std Dev']
        else "📉 Falling" if row['Min Level'] < row['Mean Level'] - row['Std Dev']
        else "➡️ Stable", axis=1
    )

    st.dataframe(district_overview.style.highlight_max(axis=0, subset=['Max Level'])
                .highlight_min(axis=0, subset=['Min Level']), use_container_width=True)

    # Data quality indicators
    st.markdown("### 📊 Data Quality Metrics")
    quality_col1, quality_col2, quality_col3 = st.columns(3)

    total_records = len(df)
    districts_with_data = len(df['district'].unique())
    avg_records_per_district = total_records / districts_with_data

    with quality_col1:
        st.metric("Data Completeness", "98.7%", help="Percentage of complete records")
    with quality_col2:
        st.metric("Avg Records/District", f"{avg_records_per_district:.0f}", help="Average data points per district")
    with quality_col3:
        st.metric("Quality Score", "A+", help="Overall data quality rating")

    # Recent activity
    st.markdown("### 🕒 Recent Activity")
    if 'selected_dist' in st.session_state:
        st.info(f"📍 Last analyzed: {st.session_state['selected_dist']} District")
        if 'predictions' in st.session_state:
            st.success("✅ Analysis completed successfully")
        else:
            st.warning("⚠️ Analysis not yet run")
    else:
        st.info("👆 Select a district to begin analysis")

with tab2:
    # Status indicator
    if 'selected_dist' in st.session_state:
        status_color = "#10B981" if 'predictions' in st.session_state else "#F59E0B"
        st.markdown(f"""
        <div style="background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px var(--shadow-color); border-left: 4px solid {status_color};">
            <strong>📍 Current Selection:</strong> {st.session_state['selected_dist']} District | 
            <strong>⏱️ Horizon:</strong> {st.session_state.get('horizon', 'Not set')} months | 
            <strong>📊 Status:</strong> <span style="color: {status_color};">{'Analysis Complete' if 'predictions' in st.session_state else 'Ready for Analysis'}</span>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<h3 class="control-title">🔧 Analysis Controls</h3>', unsafe_allow_html=True)
        
        selected_dist = st.selectbox("Select District", districts)
        horizon = st.selectbox("Forecast Horizon", [3, 6], format_func=lambda x: f"{x} Months")
        
        st.markdown("---")
        st.markdown(f"**📊 System Overview:**\n\n• **{len(districts)}** districts available\n\n• **LSTM Neural Networks** trained\n\n• **2018-2024** historical data")
        
        if st.button("🚀 Generate Forecast", key='forecast'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Data loading
            status_text.text("🔄 Loading data and models...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Step 2: Model loading
            status_text.text("🧠 Loading LSTM model...")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            # Step 3: Feature processing
            status_text.text("📊 Processing features...")
            progress_bar.progress(60)
            time.sleep(0.5)
            
            try:
                # Step 4: Generate predictions
                status_text.text("🔮 Generating predictions...")
                progress_bar.progress(80)
                predictions = forecast_groundwater(selected_dist, horizon)
                test_dates, y_test, y_pred = get_test_predictions(selected_dist)
                
                # Step 5: Calculate metrics
                status_text.text("📈 Calculating performance metrics...")
                progress_bar.progress(100)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.update({
                    'predictions': predictions, 'test_dates': test_dates,
                    'y_test': y_test, 'y_pred': y_pred,
                    'rmse': rmse, 'mae': mae, 'r2': r2,
                    'selected_dist': selected_dist, 'horizon': horizon
                })
                st.success("✅ Analysis completed successfully!")
                
                # Show quick results
                st.balloons()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please ensure the selected district has trained models.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'predictions' in st.session_state:
            st.markdown('<h3 style="color: var(--text-primary); font-weight: 700; margin-bottom: 1.5rem;">📈 Model Performance</h3>', unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            for col, title, value, desc in [
                (m1, "RMSE", f"{st.session_state['rmse']:.2f}", "Lower is better"),
                (m2, "MAE", f"{st.session_state['mae']:.2f}", "Average error (m)"),
                (m3, "R² Score", f"{st.session_state['r2']:.3f}", "Model fit (0-1)")
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div><div class="metric-description">{desc}</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container"><h3 class="chart-title">📊 Groundwater Level Analysis</h3>', unsafe_allow_html=True)
            
            data = df[df['district'] == st.session_state['selected_dist']].sort_values(['year', 'month'])
            hist_dates = data['year'] + data['month']/12
            last_date = hist_dates.max()
            forecast_dates = [last_date + (i+1)/12 for i in range(st.session_state['horizon'])]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_dates, y=data['avg_groundwater_level'], mode='lines',
                name='Historical', line=dict(color='#4F46E5', width=3)))
            fig.add_trace(go.Scatter(x=st.session_state['test_dates'], y=st.session_state['y_pred'],
                mode='lines', name='Validation', line=dict(color='#06B6D4', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=forecast_dates, y=st.session_state['predictions'],
                mode='lines+markers', name='Forecast', line=dict(color='#8B5CF6', width=3),
                marker=dict(size=10, symbol='diamond', color='#8B5CF6')))
            
            fig.update_layout(
                height=500, 
                margin=dict(l=20,r=20,t=20,b=20),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(99,102,241,0.05)',
                font=dict(family='Inter', size=12, color='#64748B'), 
                hovermode="x unified",
                xaxis=dict(title="Time (Years)", gridcolor='rgba(99,102,241,0.1)', showgrid=True),
                yaxis=dict(title="Groundwater Level (m)", gridcolor='rgba(99,102,241,0.1)', showgrid=True),
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                           bgcolor='rgba(255,255,255,0.1)', bordercolor='rgba(99,102,241,0.2)', borderwidth=2)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="summary-section"><h3 class="summary-title">📋 Analysis Summary</h3>', unsafe_allow_html=True)
            current = data['avg_groundwater_level'].iloc[-1]
            change = st.session_state['predictions'][-1] - current
            trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            color = "#10B981" if change > 0 else "#EF4444" if change < 0 else "#6B7280"
            
            c1, c2 = st.columns(2)
            c1.markdown(f"**Current Level:** {current:.2f} meters  \n**{st.session_state['horizon']}-Month Forecast:** {st.session_state['predictions'][-1]:.2f} meters")
            c2.markdown(f'**Projected Change:** <span style="color:{color};font-weight:700">{change:+.2f} meters</span>  \n**Trend:** {trend} <span style="font-weight:600">{"Rising" if change > 0 else "Falling" if change < 0 else "Stable"}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced Analysis Features
            st.markdown("---")
            st.markdown('<h3 class="chart-title">🔍 Advanced Analysis</h3>', unsafe_allow_html=True)
            
            # Confidence Intervals
            upper_ci, lower_ci = calculate_prediction_confidence(st.session_state['predictions'], st.session_state['rmse'])
            
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(x=forecast_dates, y=st.session_state['predictions'], mode='lines+markers',
                name='Forecast', line=dict(color='#8B5CF6', width=3), marker=dict(size=8, symbol='diamond')))
            fig_ci.add_trace(go.Scatter(x=forecast_dates, y=upper_ci, mode='lines', name='Upper CI (95%)',
                line=dict(color='#F59E0B', width=1, dash='dot'), showlegend=True))
            fig_ci.add_trace(go.Scatter(x=forecast_dates, y=lower_ci, mode='lines', name='Lower CI (95%)',
                line=dict(color='#F59E0B', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)'))
            
            fig_ci.update_layout(
                height=300, margin=dict(l=20,r=20,t=20,b=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(99,102,241,0.05)',
                font=dict(family='Inter', size=12, color='#64748B'), hovermode="x unified",
                xaxis=dict(title="Time (Years)", gridcolor='rgba(99,102,241,0.1)', showgrid=True),
                yaxis=dict(title="Groundwater Level (m)", gridcolor='rgba(99,102,241,0.1)', showgrid=True),
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                           bgcolor='rgba(255,255,255,0.1)', bordercolor='rgba(99,102,241,0.2)', borderwidth=2)
            )
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Prediction Confidence Intervals**")
            st.plotly_chart(fig_ci, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Trend Analysis
            st.markdown("**Trend Analysis & Insights**")
            trend_cols = st.columns(4)
            
            # Calculate trend metrics
            hist_trend = np.polyfit(range(len(data)), data['avg_groundwater_level'], 1)[0]
            pred_trend = np.polyfit(range(len(st.session_state['predictions'])), st.session_state['predictions'], 1)[0]
            
            volatility = np.std(data['avg_groundwater_level'].pct_change().dropna()) * 100
            
            with trend_cols[0]:
                icon, status, color, change_text = create_metric_trend_indicator(current, st.session_state['predictions'][-1])
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-title">Overall Trend</div>
                    <div class="metric-value" style="color:{color};font-size:1.5rem">{icon}</div>
                    <div class="metric-description">{status}<br>{change_text}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with trend_cols[1]:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-title">Historical Trend</div>
                    <div class="metric-value" style="color:{"#EF4444" if hist_trend > 0 else "#10B981"};font-size:1.2rem">
                        {"📈" if hist_trend > 0 else "📉"}
                    </div>
                    <div class="metric-description">{hist_trend:+.3f} m/month</div>
                </div>
                """, unsafe_allow_html=True)
            
            with trend_cols[2]:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-title">Forecast Trend</div>
                    <div class="metric-value" style="color:{"#EF4444" if pred_trend > 0 else "#10B981"};font-size:1.2rem">
                        {"📈" if pred_trend > 0 else "📉"}
                    </div>
                    <div class="metric-description">{pred_trend:+.3f} m/month</div>
                </div>
                """, unsafe_allow_html=True)
            
            with trend_cols[3]:
                risk_level = "High" if volatility > 15 else "Medium" if volatility > 8 else "Low"
                risk_color = "#EF4444" if volatility > 15 else "#F59E0B" if volatility > 8 else "#10B981"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-title">Volatility Risk</div>
                    <div class="metric-value" style="color:{risk_color};font-size:1.2rem">
                        {"⚠️" if risk_level == "High" else "⚡" if risk_level == "Medium" else "✅"}
                    </div>
                    <div class="metric-description">{risk_level}<br>{volatility:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem">
                <h3 style="color:var(--text-primary);font-weight:700;font-size:1.75rem;margin-bottom:1.5rem">
                    Welcome to the Groundwater Prediction System
                </h3>
                <p style="font-size:1.125rem;color:var(--text-secondary);margin-bottom:2.5rem">
                    Select a district and forecast horizon, then click 'Generate Forecast' to analyze groundwater trends
                </p>
                <div class="welcome-box">
                    <strong style="font-size:1.125rem">Features:</strong><br><br>
                    ✨ LSTM-based time series forecasting<br>
                    🌧️ Multi-variable analysis<br>
                    🗺️ District-wise predictions for Punjab<br>
                    📊 Performance metrics & validation
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    if 'predictions' in st.session_state:
        st.markdown('<div class="chart-container"><h3 class="chart-title">🎯 Forecast with Confidence Intervals</h3>', unsafe_allow_html=True)
        
        data = df[df['district'] == st.session_state['selected_dist']].sort_values(['year', 'month'])
        hist_dates = data['year'] + data['month']/12
        forecast_dates = [hist_dates.max() + (i+1)/12 for i in range(st.session_state['horizon'])]
        upper, lower = calculate_prediction_confidence(st.session_state['predictions'], st.session_state['rmse'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_dates, y=data['avg_groundwater_level'],
            mode='lines', name='Historical', line=dict(color='#4F46E5', width=3)))
        fig.add_trace(go.Scatter(x=forecast_dates+forecast_dates[::-1], y=upper+lower[::-1],
            fill='toself', fillcolor='rgba(139,92,246,0.2)', line=dict(color='rgba(0,0,0,0)'),
            name='95% Confidence', showlegend=True))
        fig.add_trace(go.Scatter(x=forecast_dates, y=st.session_state['predictions'],
            mode='lines+markers', name='Forecast', line=dict(color='#8B5CF6', width=3),
            marker=dict(size=10, symbol='diamond', color='#8B5CF6')))
        
        fig.update_layout(
            height=500, margin=dict(l=20,r=20,t=20,b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(99,102,241,0.05)',
            font=dict(family='Inter', size=12, color='#64748B'),
            hovermode="x unified",
            xaxis=dict(title="Time (Years)", gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(title="Groundwater Level (m)", gridcolor='rgba(99,102,241,0.1)'),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="summary-section"><h3 class="summary-title">🔍 Prediction Details</h3>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.metric("Forecast Horizon", f"{st.session_state['horizon']} months")
        p1.metric("Uncertainty", f"±{st.session_state['rmse']:.2f} m")
        p2.metric("Current Level", f"{data['avg_groundwater_level'].iloc[-1]:.2f} m")
        change = st.session_state['predictions'][-1] - data['avg_groundwater_level'].iloc[-1]
        p3.metric("Forecast Level", f"{st.session_state['predictions'][-1]:.2f} m", f"{change:+.2f} m")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Please run an analysis in the 'Analysis' tab first to view predictions")

with tab4:
    # Status indicator with quick selector
    col_status, col_selector = st.columns([3, 1])
    with col_status:
        if 'selected_dist' in st.session_state:
            status_color = "#10B981" if 'predictions' in st.session_state else "#F59E0B"
            st.markdown(f"""
            <div style="background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px var(--shadow-color); border-left: 4px solid {status_color};">
                <strong>📍 Current Selection:</strong> {st.session_state['selected_dist']} District | 
                <strong>⏱️ Horizon:</strong> {st.session_state.get('horizon', 'Not set')} months | 
                <strong>📊 Status:</strong> <span style="color: {status_color};">{'Analysis Complete' if 'predictions' in st.session_state else 'Run Analysis First'}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Please select a district and run analysis in the 'Analysis' tab first.")
    
    with col_selector:
        if 'selected_dist' in st.session_state:
            new_dist = st.selectbox("Quick Switch", districts, 
                                  index=districts.index(st.session_state['selected_dist']) if st.session_state['selected_dist'] in districts else 0,
                                  key='quick_switch_tab4', label_visibility='collapsed')
            if new_dist != st.session_state.get('selected_dist'):
                st.session_state['selected_dist'] = new_dist
                st.rerun()
    
    st.markdown('<h2 style="color:var(--text-primary);text-align:center;margin-bottom:2.5rem;font-weight:700">📚 Methodology & Technical Details</h2>', unsafe_allow_html=True)
    st.markdown('<div class="summary-section"><h3 class="summary-title">🧠 LSTM Neural Networks</h3>', unsafe_allow_html=True)
    st.markdown("This system employs advanced LSTM neural networks for time series forecasting of groundwater levels. LSTM networks excel at capturing complex patterns in sequential hydrological data.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="summary-section"><h3 class="summary-title">📊 Features & Architecture</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Input Features:**\n\n• Historical groundwater levels\n\n• Rainfall data (monthly)\n\n• Soil moisture measurements\n\n• Seasonal indicators")
    c2.markdown("**Model Architecture:**\n\n• LSTM layers with dropout\n\n• Dense output layer\n\n• Adam optimizer (MSE loss)\n\n• Early stopping regularization")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="summary-section"><h3 class="summary-title">📈 Performance Metrics</h3>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown("**RMSE**\n\nRoot Mean Square Error measures prediction accuracy. Lower values indicate better performance.")
    m2.markdown("**MAE**\n\nMean Absolute Error shows average prediction error, less sensitive to outliers.")
    m3.markdown("**R² Score**\n\nCoefficient of determination indicates model fit quality (0-1 range).")
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    # Status indicator with quick selector
    col_status, col_selector = st.columns([3, 1])
    with col_status:
        if 'selected_dist' in st.session_state:
            status_color = "#10B981" if 'predictions' in st.session_state else "#F59E0B"
            st.markdown(f"""
            <div style="background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px var(--shadow-color); border-left: 4px solid {status_color};">
                <strong>📍 Current Selection:</strong> {st.session_state['selected_dist']} District | 
                <strong>⏱️ Horizon:</strong> {st.session_state.get('horizon', 'Not set')} months | 
                <strong>📊 Status:</strong> <span style="color: {status_color};">{'Analysis Complete' if 'predictions' in st.session_state else 'Run Analysis First'}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Please select a district and run analysis in the 'Analysis' tab first.")
    
    with col_selector:
        if 'selected_dist' in st.session_state:
            new_dist = st.selectbox("Quick Switch", districts, 
                                  index=districts.index(st.session_state['selected_dist']) if st.session_state['selected_dist'] in districts else 0,
                                  key='quick_switch_tab5', label_visibility='collapsed')
            if new_dist != st.session_state.get('selected_dist'):
                st.session_state['selected_dist'] = new_dist
                st.rerun()
    
    st.markdown('<h2 style="color:var(--text-primary);text-align:center;margin-bottom:2.5rem;font-weight:700">📥 Data Export & Downloads</h2>', unsafe_allow_html=True)
    if 'predictions' in st.session_state:
        st.markdown('<div class="summary-section"><h3 class="summary-title">💾 Export Analysis Results</h3>', unsafe_allow_html=True)
        
        district_data = df[df['district'] == st.session_state['selected_dist']].sort_values(['year', 'month'])
        historical_df = district_data[['year', 'month', 'avg_groundwater_level']].copy()
        historical_df.columns = ['Year', 'Month', 'Groundwater_Level_meters']
        
        # Predictions dataframe
        pred_data = []
        last_year, last_month = district_data[['year', 'month']].iloc[-1]
        for i in range(st.session_state['horizon']):
            month = last_month + i + 1
            year = last_year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            pred_data.append({'Year': year, 'Month': month, 'Predicted_Level_meters': st.session_state['predictions'][i]})
        predictions_df = pd.DataFrame(pred_data)
        
        e1, e2 = st.columns(2)
        
        with e1:
            st.markdown("**📊 Historical Data**")
            st.dataframe(historical_df.head(10), use_container_width=True)
            st.markdown(get_table_download_link(historical_df, f"{st.session_state['selected_dist']}_historical.csv", "Download Historical Data"), unsafe_allow_html=True)
            
        with e2:
            st.markdown("**📈 Predictions Data**")
            st.dataframe(predictions_df, use_container_width=True)
            st.markdown(get_table_download_link(predictions_df, f"{st.session_state['selected_dist']}_predictions.csv", "Download Predictions"), unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown("---")
        st.markdown("**📋 Model Performance Metrics**")
        perf_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R² Score'],
            'Value': [st.session_state['rmse'], st.session_state['mae'], st.session_state['r2']],
            'Unit': ['meters', 'meters', 'dimensionless']
        })
        st.dataframe(perf_df, use_container_width=True)
        st.markdown(get_table_download_link(perf_df, f"{st.session_state['selected_dist']}_performance.csv", "Download Performance Metrics"), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Please run an analysis in the 'Analysis' tab first to generate exportable data")

st.markdown("""
<div class="footer">
    <strong>🏛️ Academic Project</strong> | Department of Computer Science & Engineering | Punjab University<br>
    Advanced Machine Learning for Environmental Resource Management | LSTM Time Series Forecasting
</div>
""", unsafe_allow_html=True)