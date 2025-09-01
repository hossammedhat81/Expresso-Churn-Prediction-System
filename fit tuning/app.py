import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

# ====== Custom Transformers (Fixed version) ======
class ClipTransformer(BaseEstimator, TransformerMixin):
    """Clip numerical columns at 1st and 99th percentile using column indices"""
    def __init__(self, cols_idx=None):  # Fixed: was _init now __init__
        self.cols_idx = cols_idx
        self.lower = {}
        self.upper = {}
        
    def fit(self, X, y=None):
        if self.cols_idx is None:
            return self
        for idx in self.cols_idx:
            self.lower[idx] = np.percentile(X[:, idx], 1)
            self.upper[idx] = np.percentile(X[:, idx], 99)
        return self
    
    def transform(self, X):
        if self.cols_idx is None:
            return X
        X = X.copy()
        for idx in self.cols_idx:
            X[:, idx] = np.clip(X[:, idx], self.lower[idx], self.upper[idx])
        return X

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_dicts = []
        for i in range(X.shape[1]):
            vals, counts = np.unique(X[:, i], return_counts=True)
            freq = dict(zip(vals, counts / X.shape[0]))
            self.freq_dicts.append(freq)
        return self

    def transform(self, X):
        X = X.copy()
        for i in range(X.shape[1]):
            freq = self.freq_dicts[i]
            X[:, i] = np.array([freq.get(v, 0) for v in X[:, i]])
        return X.astype(float)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # X[:,0] = FREQUENCE_RECH, X[:,1] = REVENUE
        X = np.hstack([X, (X[:,0] / (X[:,1] + 1)).reshape(-1,1)])
        return X

def tenure_mapper(X):
    mapping = {
        "D 3-6 month": 1,
        "E 6-9 month": 2,
        "F 9-12 month": 3,
        "G 12-15 month": 4,
        "H 15-18 month": 5,
        "I 18-21 month": 6,
        "J 21-24 month": 7,
        "K > 24 month": 8
    }
    X_mapped = np.vectorize(lambda x: mapping.get(x, np.nan))(X)
    return X_mapped.astype(float)

# ====== Model Results Configuration ======
model_results = {
    "XGBoost": {
        "auc": 0.9159,
        "params": {
            "subsample": 0.8,
            "scale_pos_weight": 1,
            "n_estimators": 500,
            "min_child_weight": 1,
            "max_depth": 8,
            "learning_rate": 0.01,
            "colsample_bytree": 0.8
        },
        "color": "#FF6B6B",
        "rank": 1
    },
    "LightGBM": {
        "auc": 0.9158,
        "params": {
            "subsample": 0.9,
            "scale_pos_weight": 1.5,
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.01,
            "colsample_bytree": 0.8
        },
        "color": "#4ECDC4",
        "rank": 2
    },
    "CatBoost": {
        "auc": 0.9156,
        "params": {
            "subsample": 0.8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
            "iterations": 1000,
            "depth": 8
        },
        "color": "#45B7D1",
        "rank": 3
    },
    "Random Forest": {
        "auc": 0.8472,
        "params": {},
        "color": "#FFD700",
        "rank": 4
    },
    "Logistic Regression": {
        "auc": 0.8360,
        "params": {},
        "color": "#A9A9A9",
        "rank": 5
    }
}

drop_cols = ["ZONE1", "ZONE2", "MRG", "ARPU_SEGMENT", "MONTANT"]   
num_cols = ["FREQUENCE_RECH", "REVENUE", "FREQUENCE", 
            "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO",  
            "REGULARITY", "FREQ_TOP_PACK"]
cat_cols_low = ["REGION"]
cat_cols_high = ["TOP_PACK"]
tenure_col = ["TENURE"]
expected_columns = num_cols + tenure_col + cat_cols_low + cat_cols_high

# ====== Load Model ======
@st.cache_resource
def load_model():
    """Load the complete pipeline model"""
    try:
        model = joblib.load("expresso_churn_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'expresso_churn_model.pkl' not found!")
        return None
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

# ====== Enhanced CSV Reading Function ======
def read_csv_robust(uploaded_file):
    """
    Robust CSV reading with multiple fallback methods
    Returns: DataFrame or None
    """
    df = None
    error_messages = []
    
    # Check if file has content
    file_content = uploaded_file.read()
    if len(file_content) == 0:
        st.error("The uploaded file is empty!")
        return None
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    # Show file preview for debugging
    try:
        first_lines = file_content.decode('utf-8').split('\n')[:3]
        st.info("File preview (first 3 lines):")
        for i, line in enumerate(first_lines):
            if line.strip():
                st.code(f"Line {i+1}: {line[:100]}...")
    except:
        st.warning("Could not preview file content - trying alternative encodings")
    
    # Method 1: Standard UTF-8 with comma separator
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        if not df.empty and len(df.columns) > 1:
            st.success("File read successfully with UTF-8 encoding and comma separator")
            return df
        else:
            df = None
            error_messages.append("UTF-8 comma: resulted in empty or single-column dataframe")
    except Exception as e:
        error_messages.append(f"UTF-8 comma: {str(e)}")
    
    # Method 2: UTF-8 with semicolon separator
    if df is None:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
            if not df.empty and len(df.columns) > 1:
                st.success("File read successfully with UTF-8 encoding and semicolon separator")
                return df
            else:
                df = None
                error_messages.append("UTF-8 semicolon: resulted in empty or single-column dataframe")
        except Exception as e:
            error_messages.append(f"UTF-8 semicolon: {str(e)}")
    
    # Method 3: Auto-detect separator
    if df is None:
        try:
            uploaded_file.seek(0)
            # Read first line to detect separator
            first_line = uploaded_file.readline().decode('utf-8')
            separators = [',', ';', '\t', '|']
            sep_counts = {sep: first_line.count(sep) for sep in separators}
            best_sep = max(sep_counts, key=sep_counts.get)
            
            if sep_counts[best_sep] > 0:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=best_sep, encoding='utf-8')
                if not df.empty and len(df.columns) > 1:
                    st.success(f"File read successfully with auto-detected separator: '{best_sep}'")
                    return df
                else:
                    df = None
                    error_messages.append(f"Auto-detected '{best_sep}': resulted in empty or single-column dataframe")
        except Exception as e:
            error_messages.append(f"Auto-detection: {str(e)}")
    
    # Method 4: Try different encodings
    if df is None:
        encodings = ['latin1', 'cp1252', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            for sep in [',', ';']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
                    if not df.empty and len(df.columns) > 1:
                        st.success(f"File read successfully with {encoding} encoding and '{sep}' separator")
                        return df
                    else:
                        df = None
                        error_messages.append(f"{encoding} with '{sep}': resulted in empty or single-column dataframe")
                except Exception as e:
                    error_messages.append(f"{encoding} with '{sep}': {str(e)}")
    
    # If all methods failed, show detailed error information
    if df is None:
        st.error("Could not read the CSV file with any method")
        with st.expander("Detailed Error Analysis"):
            for i, msg in enumerate(error_messages[-10:], 1):  # Show last 10 errors
                st.write(f"{i}. {msg}")
        
        st.error("Troubleshooting Steps:")
        st.write("1. Check that your file is a valid CSV format")
        st.write("2. Ensure the file has proper column headers in the first row")
        st.write("3. Try different separators: comma (,), semicolon (;), or tab")
        st.write("4. Remove any empty rows at the beginning of the file")
        st.write("5. Save your file with UTF-8 encoding")
        
    return df

# ====== Page Configuration ======
st.set_page_config(
    page_title="Expresso Churn Predictor", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Custom CSS ======
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ====== Header ======
st.markdown("""
<div class="main-header">
    <h1>🔮 Expresso Churn Predictor</h1>
    <p>AI-Powered Customer Churn Prediction System</p>
</div>
""", unsafe_allow_html=True)

# ====== Sidebar ======
with st.sidebar:
    st.markdown("## Control Panel")
    
    st.markdown("### Model Information")
    st.info("Current Model: XGBoost Pipeline")
    st.metric("AUC Score", "0.9159")
    
    st.markdown("### Options")
    show_probabilities = st.checkbox("Show Prediction Probabilities", value=True)
    show_debug_info = st.checkbox("Show Debug Information", value=False)
    confidence_threshold = st.slider("Churn Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Model Comparison Section
    st.markdown("### 🏆 Model Comparison")
    
    # Create expandable sections for all models, using code font and emoji
    for model_name, model_data in model_results.items():
        if model_name == "XGBoost":
            rank_emoji = "🥇"
        elif model_name == "LightGBM":
            rank_emoji = "🥈"
        elif model_name == "CatBoost":
            rank_emoji = "🥉"
        elif model_name == "Random Forest":
            rank_emoji = "🌲"
        elif model_name == "Logistic Regression":
            rank_emoji = "📉"
        else:
            rank_emoji = ""
        
        with st.expander(f"{rank_emoji} `{model_name}` - AUC: `{model_data['auc']:.4f}`"):
            st.markdown("**📊 Performance:**")
            st.write(f"• **AUC Score**: `{model_data['auc']:.4f}`")
            st.write(f"• **Rank**: `#{model_data['rank']}`")
            
            st.markdown("**⚙️ Best Parameters:**")
            if model_data['params']:
                for param, value in model_data['params'].items():
                    clean_param = param.replace('classifier__', '')
                    st.write(f"• `{clean_param}`: `{value}`")
            else:
                st.write("*No hyperparameters specified (baseline model).*")
    
    # Quick comparison table
    st.markdown("### 📈 Quick Comparison")
    
    comparison_data = []
    for model_name, model_data in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'AUC (Validation/Test)': model_data['auc'],
            'Rank': model_data['rank']
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('AUC (Validation/Test)', ascending=False)
    
    def highlight_row(row):
        if row['Rank'] == 1:
            return ['background-color: #90EE90'] * len(row)
        elif row['Rank'] == 2:
            return ['background-color: #FFE4B5'] * len(row)
        elif row['Rank'] == 3:
            return ['background-color: #FFA07A'] * len(row)
        else:
            return [''] * len(row)
    styled_df = comparison_df.style.format({
        'AUC (Validation/Test)': '{:.4f}'
    }).apply(highlight_row, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.markdown("#### *AUC for Logistic Regression and Random Forest are from validation, not test.*")
    
    # Model insights and characteristics
    st.markdown("### 💡 Model Insights")
    best_model = min(model_results.items(), key=lambda x: x[1]['rank'])
    worst_model = max(model_results.items(), key=lambda x: x[1]['rank'])
    auc_diff = best_model[1]['auc'] - worst_model[1]['auc']
    
    st.write(f"• **Best Performer:** `{best_model[0]}`")
    st.write(f"• **Performance Gap:** `{auc_diff:.4f}`")
    st.write("• **All Top 3 Models:** Excellent (>0.91 AUC)")
    st.write("• **Random Forest/Logistic Regression:** Baseline, lower but useful for benchmarking")
    st.info("Consider ensemble methods for best results.")
    
    st.markdown("### 🔍 Model Characteristics")
    st.write("**XGBoost**: Best overall performance, robust")
    st.write("**LightGBM**: Fast training, good accuracy")  
    st.write("**CatBoost**: Handles categoricals well")
    st.write("**Random Forest**: Baseline, interpretable, lower AUC")
    st.write("**Logistic Regression**: Simple, interpretable, lowest AUC")
    
    st.markdown("---")
    st.markdown("**💡 Tip:** All tree-based models show excellent performance (AUC > 0.91). The current deployment uses XGBoost for optimal results. Baselines (RF/LR) included for reference.")

# ====== Main Content ======
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Analysis", "🔬 Advanced Analytics", "ℹ️ Help"])

with tab1:
    st.markdown("## Upload Your Data")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose your Test.csv file",
            type=["csv"],
            help="Upload your customer data for churn prediction"
        )
    
    with col2:
        if uploaded_file:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            st.metric("File Name", uploaded_file.name)
    
    # Load model
    model_pipeline = load_model()
    if model_pipeline:
        st.success("Model loaded successfully!")
    else:
        st.warning("Model not available. Please ensure 'expresso_churn_model.pkl' exists.")

    # Process uploaded file
    if uploaded_file and model_pipeline:
        st.markdown("## Processing Your Data")
        
        # Read CSV with robust method
        df = read_csv_robust(uploaded_file)
        
        if df is not None:
            st.info(f"Data shape: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")
            
            # Validate required columns
            if "user_id" not in df.columns:
                st.error("Missing required 'user_id' column!")
                st.error(f"Available columns: {list(df.columns)}")
                st.stop()
            
            # Show data preview
            with st.expander("Preview your data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data preprocessing
            st.markdown("### Preprocessing Pipeline")
            
            # Step 1: Drop unnecessary columns
            df_processed = df.copy()
            columns_to_drop = [col for col in drop_cols if col in df_processed.columns]
            if columns_to_drop:
                df_processed = df_processed.drop(columns=columns_to_drop)
                st.success(f"Dropped columns: {columns_to_drop}")
            
            # Step 2: Check for required columns
            available_cols = [col for col in expected_columns if col in df_processed.columns]
            missing_cols = [col for col in expected_columns if col not in df_processed.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: **{missing_cols}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Available:")
                    for col in available_cols:
                        st.write(f"  ✅ {col}")
                        
                with col2:
                    st.write("Missing:")
                    for col in missing_cols:
                        st.write(f"  ❌ {col}")
                
                st.stop()
            
            # Step 3: Prepare features
            X_test = df_processed[expected_columns].copy()
            user_ids = df_processed["user_id"].copy()
            
            st.success(f"Features prepared: **{X_test.shape[0]:,}** samples × **{X_test.shape[1]}** features")
            
            # Show debug info if requested
            if show_debug_info:
                with st.expander("Debug Information"):
                    st.write("**Numerical columns:**")
                    for col in num_cols:
                        dtype = X_test[col].dtype
                        null_count = X_test[col].isnull().sum()
                        st.write(f"  • {col}: {dtype} (nulls: {null_count})")
                    
                    st.write("**Categorical columns:**")  
                    for col in cat_cols_low + cat_cols_high + tenure_col:
                        dtype = X_test[col].dtype
                        unique_count = X_test[col].nunique()
                        null_count = X_test[col].isnull().sum()
                        st.write(f"  • {col}: {dtype} (unique: {unique_count}, nulls: {null_count})")
            
            # Step 4: Make predictions
            st.markdown("### Making Predictions")
            
            with st.spinner("Processing predictions..."):
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Preprocessing data...")
                    progress_bar.progress(25)
                    time.sleep(0.5)
                    
                    status_text.text("Running model inference...")
                    progress_bar.progress(50)
                    
                    # Make predictions
                    y_pred = model_pipeline.predict(X_test)
                    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
                    
                    status_text.text("Finalizing results...")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("Predictions completed successfully!")
                    
                    # Calculate metrics
                    churn_count = (y_pred == 1).sum()
                    total_count = len(y_pred)
                    churn_rate = churn_count / total_count * 100
                    high_risk_count = (y_proba >= confidence_threshold).sum()
                    avg_churn_prob = y_proba.mean() * 100
                    
                    # Display results
                    st.markdown("### Results Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Customers", f"{total_count:,}")
                    with col2:
                        st.metric("Predicted Churners", f"{churn_count:,}")
                    with col3:
                        st.metric("Churn Rate", f"{churn_rate:.1f}%")
                    with col4:
                        st.metric("High Risk", f"{high_risk_count:,}")
                    
                    # Create submission dataframe
                    submission = pd.DataFrame({
                        "user_id": user_ids,
                        "CHURN": y_pred
                    })
                    
                    if show_probabilities:
                        submission["CHURN_PROBABILITY"] = y_proba.round(4)
                        submission["RISK_LEVEL"] = pd.cut(y_proba, 
                                                        bins=[0, 0.3, 0.7, 1.0], 
                                                        labels=["Low", "Medium", "High"])
                    
                    # Show results table
                    st.markdown("### Prediction Results")
                    st.dataframe(submission.head(20), use_container_width=True)
                    
                    if len(submission) > 20:
                        st.info(f"Showing first 20 rows. Full dataset contains {len(submission):,} predictions.")
                    
                    # Download section
                    st.markdown("### Download Results")
                    csv_data = submission.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # Store results for analysis tab
                    st.session_state['prediction_results'] = {
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'total_count': total_count,
                        'churn_count': churn_count,
                        'churn_rate': churn_rate,
                        'submission': submission,
                        'X_test': X_test,  # Store processed features for advanced analysis
                        'user_ids': user_ids
                    }
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    if show_debug_info:
                        st.error(f"Error type: {type(e).__name__}")
                        st.error(f"Input shape: {X_test.shape}")
                        st.error(f"Input dtypes: {dict(X_test.dtypes)}")

with tab2:
    st.markdown("## Analysis Dashboard")
    
    if 'prediction_results' in st.session_state:
        results = st.session_state['prediction_results']
        y_proba = results['y_proba']
        y_pred = results['y_pred']
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution pie chart
            fig_pie = px.pie(
                values=[results['total_count'] - results['churn_count'], results['churn_count']],
                names=['Retained', 'Churned'],
                title="Customer Churn Distribution",
                color_discrete_sequence=['#00CC96', '#FF6B6B']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Probability distribution
            fig_hist = px.histogram(
                x=y_proba,
                nbins=50,
                title="Churn Probability Distribution",
                labels={'x': 'Churn Probability', 'y': 'Count'},
                color_discrete_sequence=['#667eea']
            )
            fig_hist.add_vline(
                x=confidence_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Threshold: {confidence_threshold:.0%}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Risk segmentation
        st.markdown("### Risk Segmentation")
        risk_segments = pd.cut(y_proba, bins=[0, 0.25, 0.5, 0.75, 1.0], 
                              labels=['Very Low', 'Low', 'High', 'Very High'])
        risk_counts = risk_segments.value_counts()
        
        fig_risk = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Customer Risk Distribution",
            labels={'x': 'Risk Level', 'y': 'Count'},
            color=risk_counts.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Insights
        st.markdown("### Key Insights")
        very_high_risk = (y_proba >= 0.75).sum()
        high_risk = ((y_proba >= 0.5) & (y_proba < 0.75)).sum()
        
        insights = [
            f"🚨 {very_high_risk:,} customers are at very high risk (>75% probability)",
            f"⚠️ {high_risk:,} customers are at high risk (50-75% probability)",
            f"📊 Average churn probability is {results['churn_rate']:.1f}%",
            f"🎯 Focus retention efforts on {(very_high_risk + high_risk):,} high-risk customers"
        ]
        
        for insight in insights:
            st.write(f"- {insight}")
            
    else:
        st.info("Upload and process data in the Prediction tab to see analysis")

with tab3:
    st.markdown("## Advanced Analytics & EDA Dashboard")
    
    if 'prediction_results' in st.session_state:
        results = st.session_state['prediction_results']
        y_proba = results['y_proba']
        y_pred = results['y_pred']
        X_test = results['X_test']
        user_ids = results['user_ids']
        
        # Create comprehensive analysis dataset
        analysis_df = X_test.copy()
        analysis_df['user_id'] = user_ids
        analysis_df['churn_prediction'] = y_pred
        analysis_df['churn_probability'] = y_proba
        analysis_df['risk_score'] = pd.cut(y_proba, bins=[0, 0.25, 0.5, 0.75, 1.0], 
                                          labels=['Very Low', 'Low', 'High', 'Critical'])
        
        # Calculate feature statistics for churners vs non-churners
        churners = analysis_df[analysis_df['churn_prediction'] == 1]
        non_churners = analysis_df[analysis_df['churn_prediction'] == 0]
        
        # Enhanced EDA Dashboard
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "📊 Data Overview", "🔥 Feature Impact", "📈 Advanced Visualizations", 
            "🎯 Customer Segmentation", "💼 Business Intelligence"
        ])
        
        with eda_tab1:
            st.markdown("### Dataset Overview")
            
            # Enhanced overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(analysis_df):,}</h3>
                    <p>Total Customers</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                churn_rate = (analysis_df['churn_prediction'] == 1).mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{churn_rate:.1f}%</h3>
                    <p>Churn Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_prob = analysis_df['churn_probability'].mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_prob:.1f}%</h3>
                    <p>Avg Churn Prob</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                high_risk = (analysis_df['churn_probability'] >= 0.7).sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{high_risk:,}</h3>
                    <p>High Risk (≥70%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced distribution visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn probability distribution with KDE
                fig_prob_dist = go.Figure()
                
                # Histogram
                fig_prob_dist.add_trace(go.Histogram(
                    x=analysis_df['churn_probability'],
                    nbinsx=50,
                    opacity=0.7,
                    name='Distribution',
                    marker_color='rgba(102, 126, 234, 0.7)'
                ))
                
                # Add KDE-like smooth curve
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(analysis_df['churn_probability'])
                x_range = np.linspace(0, 1, 100)
                kde_values = kde(x_range) * len(analysis_df) * 0.02  # Scale for histogram
                
                fig_prob_dist.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode='lines',
                    name='Density Curve',
                    line=dict(color='red', width=3)
                ))
                
                fig_prob_dist.update_layout(
                    title="Churn Probability Distribution with Density Curve",
                    xaxis_title="Churn Probability",
                    yaxis_title="Count",
                    showlegend=True,
                    template='plotly_white'
                )
                st.plotly_chart(fig_prob_dist, use_container_width=True)
            
            with col2:
                # Enhanced risk level donut chart
                risk_counts = analysis_df['risk_score'].value_counts()
                colors = ['#00CC96', '#FFA15A', '#FF6B6B', '#8B0000']
                
                fig_donut = go.Figure(data=[go.Pie(
                    labels=risk_counts.index, 
                    values=risk_counts.values,
                    hole=0.5,
                    marker_colors=colors,
                    textinfo='label+percent+value',
                    textposition='outside'
                )])
                
                fig_donut.update_layout(
                    title="Customer Risk Level Distribution",
                    annotations=[dict(text='Risk<br>Levels', x=0.5, y=0.5, font_size=16, showarrow=False)],
                    template='plotly_white'
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            
            # Data quality assessment
            st.markdown("### Data Quality Assessment")
            
            # Missing values heatmap
            missing_data = analysis_df[num_cols + cat_cols_low + cat_cols_high + tenure_col].isnull().sum()
            if missing_data.sum() > 0:
                fig_missing = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Feature",
                    labels={'x': 'Features', 'y': 'Missing Count'},
                    color=missing_data.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("No missing values detected in the dataset!")
            
            # Statistical summary
            st.markdown("### Statistical Summary")
            summary_stats = analysis_df[num_cols].describe().T
            summary_stats['missing'] = analysis_df[num_cols].isnull().sum()
            summary_stats['skewness'] = analysis_df[num_cols].skew()
            st.dataframe(summary_stats.round(3), use_container_width=True)
        
        with eda_tab2:
            st.markdown("### Feature Impact Analysis")
            
            # Calculate feature importance metrics
            risk_factors = []
            correlations = []
            
            for col in num_cols:
                if col in analysis_df.columns:
                    churner_mean = churners[col].mean()
                    non_churner_mean = non_churners[col].mean()
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((churners[col].var() + non_churners[col].var()) / 2))
                    cohens_d = abs(churner_mean - non_churner_mean) / pooled_std if pooled_std > 0 else 0
                    
                    # Statistical significance (t-test p-value approximation)
                    from scipy.stats import ttest_ind
                    try:
                        t_stat, p_value = ttest_ind(churners[col].dropna(), non_churners[col].dropna())
                        significance = "High" if p_value < 0.01 else "Medium" if p_value < 0.05 else "Low"
                    except:
                        p_value = 1.0
                        significance = "Low"
                    
                    risk_factors.append({
                        'feature': col,
                        'churner_mean': churner_mean,
                        'non_churner_mean': non_churner_mean,
                        'mean_diff': abs(churner_mean - non_churner_mean),
                        'cohens_d': cohens_d,
                        'p_value': p_value,
                        'significance': significance
                    })
                    
                    # Correlation
                    corr = np.corrcoef(analysis_df[col], analysis_df['churn_probability'])[0,1]
                    if not np.isnan(corr):
                        correlations.append((col, abs(corr), corr))
            
            # Feature importance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Effect size (Cohen's d) visualization
                risk_df = pd.DataFrame(risk_factors).sort_values('cohens_d', ascending=True)
                
                fig_effect = px.bar(
                    risk_df,
                    x='cohens_d',
                    y='feature',
                    orientation='h',
                    title="Feature Effect Sizes (Cohen's d)",
                    labels={'cohens_d': 'Effect Size', 'feature': 'Features'},
                    color='cohens_d',
                    color_continuous_scale='Viridis'
                )
                
                # Add effect size interpretation lines
                fig_effect.add_vline(x=0.2, line_dash="dash", annotation_text="Small Effect")
                fig_effect.add_vline(x=0.5, line_dash="dash", annotation_text="Medium Effect")
                fig_effect.add_vline(x=0.8, line_dash="dash", annotation_text="Large Effect")
                
                st.plotly_chart(fig_effect, use_container_width=True)
            
            with col2:
                # Correlation waterfall chart
                correlations.sort(key=lambda x: x[1], reverse=True)
                corr_df = pd.DataFrame(correlations, columns=['feature', 'abs_corr', 'corr'])
                
                fig_corr = go.Figure(go.Waterfall(
                    name="Correlations",
                    orientation="v",
                    x=corr_df['feature'],
                    y=corr_df['corr'],
                    connector={"line":{"color":"rgb(63, 63, 63)"}},
                    increasing={"marker":{"color":"red"}},
                    decreasing={"marker":{"color":"blue"}},
                    totals={"marker":{"color":"green"}}
                ))
                
                fig_corr.update_layout(
                    title="Feature-Churn Correlations (Waterfall)",
                    xaxis_title="Features",
                    yaxis_title="Correlation"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Feature importance table with advanced metrics
            st.markdown("### Detailed Feature Analysis")
            
            feature_analysis = pd.DataFrame(risk_factors)
            feature_analysis = feature_analysis.sort_values('cohens_d', ascending=False)
            
            # Format the dataframe for display
            display_df = feature_analysis.copy()
            display_df['churner_mean'] = display_df['churner_mean'].round(3)
            display_df['non_churner_mean'] = display_df['non_churner_mean'].round(3)
            display_df['mean_diff'] = display_df['mean_diff'].round(3)
            display_df['cohens_d'] = display_df['cohens_d'].round(3)
            display_df['p_value'] = display_df['p_value'].round(4)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Top insights
            st.markdown("### Key Insights from Feature Analysis")
            
            top_feature = feature_analysis.iloc[0]
            most_corr_feature = corr_df.iloc[0]
            
            insights = [
                f"**Strongest Differentiator**: {top_feature['feature']} shows the largest effect size (Cohen's d = {top_feature['cohens_d']:.3f})",
                f"**Statistical Significance**: {len([f for f in risk_factors if f['significance'] == 'High'])} features show high statistical significance",
                f"**Highest Correlation**: {most_corr_feature['feature']} has the strongest correlation with churn ({most_corr_feature['corr']:.3f})",
                f"**Revenue Impact**: {'High' if any(f['feature'] == 'REVENUE' and f['cohens_d'] > 0.5 for f in risk_factors) else 'Moderate'} revenue differentiation between segments"
            ]
            
            for insight in insights:
                st.write(f"• {insight}")
        
        with eda_tab3:
            st.markdown("### Advanced Visualizations")
            
            # Subplots for comprehensive analysis
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "Revenue Deep Dive", "Usage Behavior", "Geographic Analysis", "Temporal Patterns"
            ])
            
            with viz_tab1:
                if 'REVENUE' in analysis_df.columns:
                    # Multi-dimensional revenue analysis
                    fig_revenue_multi = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Revenue Distribution', 'Revenue vs Probability', 
                                      'Revenue Quantiles', 'Revenue Risk Matrix'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": True}, {"type": "heatmap"}]]
                    )
                    
                    # 1. Box plots by churn status
                    # Calculate quantile counts for revenue
                    revenue_quantiles = pd.qcut(analysis_df['REVENUE'], q=10, duplicates='drop')
                    quantile_count = revenue_quantiles.value_counts().sort_index()
                    
                    fig_revenue_multi.add_trace(
                        go.Scatter(x=quantile_count.index.astype(str), y=quantile_count.values, 
                                 mode='lines+markers', name='Customer Count', yaxis='y2'),
                        row=2, col=1, secondary_y=True
                    )
                    
                    # 4. Revenue-Risk heatmap
                    revenue_bins = pd.cut(analysis_df['REVENUE'], bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
                    risk_bins = analysis_df['risk_score']
                    heatmap_data = pd.crosstab(revenue_bins, risk_bins, normalize='index')
                    
                    fig_revenue_multi.add_trace(
                        go.Heatmap(z=heatmap_data.values, 
                                 x=heatmap_data.columns, 
                                 y=heatmap_data.index,
                                 colorscale='Reds'),
                        row=2, col=2
                    )
                    
                    fig_revenue_multi.update_layout(height=800, title_text="Comprehensive Revenue Analysis")
                    st.plotly_chart(fig_revenue_multi, use_container_width=True)
                    
                    # Revenue funnel analysis
                    revenue_deciles = pd.qcut(analysis_df['REVENUE'], q=10, labels=False)
                    funnel_data = analysis_df.groupby(revenue_deciles).agg({
                        'churn_prediction': 'mean',
                        'churn_probability': 'mean',
                        'user_id': 'count'
                    }).reset_index()
                    funnel_data['decile'] = funnel_data['REVENUE'] + 1
                    
                    fig_funnel = go.Figure()
                    fig_funnel.add_trace(go.Funnel(
                        y=[f"Decile {i+1}" for i in range(10)],
                        x=funnel_data['user_id'],
                        textinfo="value+percent initial",
                        marker_color=[f"rgba(255, {int(255-i*25)}, {int(255-i*25)}, 0.8)" for i in range(10)]
                    ))
                    fig_funnel.update_layout(title="Customer Revenue Funnel")
                    st.plotly_chart(fig_funnel, use_container_width=True)
            
            with viz_tab2:
                # Advanced usage behavior analysis
                usage_cols = ['FREQUENCE_RECH', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'FREQUENCE']
                available_usage_cols = [col for col in usage_cols if col in analysis_df.columns]
                
                if len(available_usage_cols) >= 3:
                    # Parallel coordinates plot
                    fig_parallel = go.Figure(data=go.Parcoords(
                        line=dict(color=analysis_df['churn_probability'],
                                colorscale='Viridis',
                                showscale=True),
                        dimensions=[dict(label=col, 
                                       values=analysis_df[col]) for col in available_usage_cols[:6]]
                    ))
                    fig_parallel.update_layout(title="Usage Pattern Parallel Coordinates")
                    st.plotly_chart(fig_parallel, use_container_width=True)
                    
                    # Usage behavior clustering visualization
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans
                    
                    # Prepare data for clustering
                    usage_data = analysis_df[available_usage_cols].fillna(0)
                    scaler = StandardScaler()
                    scaled_usage = scaler.fit_transform(usage_data)
                    
                    # K-means clustering
                    kmeans = KMeans(n_clusters=4, random_state=42)
                    clusters = kmeans.fit_predict(scaled_usage)
                    analysis_df['usage_cluster'] = clusters
                    
                    # Cluster analysis
                    cluster_analysis = analysis_df.groupby('usage_cluster').agg({
                        'churn_prediction': 'mean',
                        'churn_probability': 'mean',
                        'user_id': 'count'
                    }).round(3)
                    
                    fig_cluster = px.bar(
                        x=cluster_analysis.index,
                        y=cluster_analysis['churn_prediction'],
                        title="Churn Rate by Usage Cluster",
                        labels={'x': 'Usage Cluster', 'y': 'Churn Rate'},
                        color=cluster_analysis['churn_prediction'],
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    # Network usage patterns (if network data available)
                    network_cols = ['ON_NET', 'ORANGE', 'TIGO']
                    available_network = [col for col in network_cols if col in analysis_df.columns]
                    
                    if len(available_network) >= 2:
                        # Create sunburst chart for network usage
                        network_data = analysis_df[available_network].fillna(0)
                        network_data['total'] = network_data.sum(axis=1)
                        
                        # Calculate network preferences
                        network_prefs = []
                        for idx, row in network_data.iterrows():
                            if row['total'] > 0:
                                dominant_network = row[available_network].idxmax()
                                network_prefs.append(dominant_network)
                            else:
                                network_prefs.append('None')
                        
                        analysis_df['dominant_network'] = network_prefs
                        
                        network_churn = analysis_df.groupby('dominant_network')['churn_prediction'].mean()
                        
                        fig_network = px.pie(
                            values=network_churn.values,
                            names=network_churn.index,
                            title="Churn Rate by Dominant Network Usage"
                        )
                        st.plotly_chart(fig_network, use_container_width=True)
            
            with viz_tab3:
                # Enhanced geographic analysis
                if 'REGION' in analysis_df.columns:
                    region_stats = analysis_df.groupby('REGION').agg({
                        'churn_prediction': 'mean',
                        'churn_probability': ['mean', 'std'],
                        'user_id': 'count'
                    }).round(3)
                    
                    region_stats.columns = ['Churn_Rate', 'Avg_Prob', 'Prob_Std', 'Customer_Count']
                    
                    # Multi-metric regional analysis
                    fig_regional = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Churn Rate by Region', 'Customer Distribution', 
                                      'Risk Variability', 'Regional Performance Matrix'),
                        specs=[[{"secondary_y": False}, {"type": "pie"}],
                               [{"secondary_y": False}, {"type": "scatter"}]]
                    )
                    
                    # Churn rate by region
                    fig_regional.add_trace(
                        go.Bar(x=region_stats.index, y=region_stats['Churn_Rate'], 
                             name='Churn Rate', marker_color='red'),
                        row=1, col=1
                    )
                    
                    # Customer distribution pie
                    fig_regional.add_trace(
                        go.Pie(labels=region_stats.index, values=region_stats['Customer_Count']),
                        row=1, col=2
                    )
                    
                    # Risk variability
                    fig_regional.add_trace(
                        go.Bar(x=region_stats.index, y=region_stats['Prob_Std'], 
                             name='Risk Std Dev', marker_color='orange'),
                        row=2, col=1
                    )
                    
                    # Performance scatter
                    fig_regional.add_trace(
                        go.Scatter(x=region_stats['Customer_Count'], 
                                 y=region_stats['Churn_Rate'],
                                 mode='markers+text',
                                 text=region_stats.index,
                                 textposition="top center",
                                 marker=dict(size=region_stats['Prob_Std']*500, 
                                           color=region_stats['Churn_Rate'],
                                           colorscale='Reds',
                                           showscale=True)),
                        row=2, col=2
                    )
                    
                    fig_regional.update_layout(height=800, title_text="Regional Analysis Dashboard")
                    st.plotly_chart(fig_regional, use_container_width=True)
                    
                    # Regional recommendations
                    st.markdown("#### Regional Insights")
                    highest_risk_region = region_stats['Churn_Rate'].idxmax()
                    largest_region = region_stats['Customer_Count'].idxmax()
                    most_volatile = region_stats['Prob_Std'].idxmax()
                    
                    regional_insights = [
                        f"**Highest Risk Region**: {highest_risk_region} ({region_stats.loc[highest_risk_region, 'Churn_Rate']:.1%} churn rate)",
                        f"**Largest Market**: {largest_region} ({region_stats.loc[largest_region, 'Customer_Count']:,} customers)",
                        f"**Most Volatile**: {most_volatile} (highest prediction uncertainty)",
                        f"**Priority Regions**: Focus on {highest_risk_region} for risk mitigation and {largest_region} for volume impact"
                    ]
                    
                    for insight in regional_insights:
                        st.write(f"• {insight}")
            
            with viz_tab4:
                # Temporal and behavioral patterns
                st.markdown("#### Behavioral Pattern Analysis")
                
                # Create tenure-based analysis if available
                if 'TENURE' in analysis_df.columns:
                    # Tenure impact analysis
                    tenure_analysis = analysis_df.groupby('TENURE').agg({
                        'churn_prediction': 'mean',
                        'churn_probability': 'mean',
                        'user_id': 'count'
                    }).reset_index()
                    
                    fig_tenure = px.scatter(
                        tenure_analysis,
                        x='TENURE',
                        y='churn_prediction',
                        size='user_id',
                        color='churn_probability',
                        title="Churn Rate vs Customer Tenure",
                        labels={'churn_prediction': 'Churn Rate', 'user_id': 'Customer Count'}
                    )
                    st.plotly_chart(fig_tenure, use_container_width=True)
                
                # Activity level segmentation
                if 'FREQUENCE_RECH' in analysis_df.columns and 'REGULARITY' in analysis_df.columns:
                    # Create activity quadrants
                    freq_median = analysis_df['FREQUENCE_RECH'].median()
                    reg_median = analysis_df['REGULARITY'].median()
                    
                    def categorize_activity(row):
                        freq_high = row['FREQUENCE_RECH'] >= freq_median
                        reg_high = row['REGULARITY'] >= reg_median
                        
                        if freq_high and reg_high:
                            return 'High Activity & Regular'
                        elif freq_high and not reg_high:
                            return 'High Activity & Irregular'
                        elif not freq_high and reg_high:
                            return 'Low Activity & Regular'
                        else:
                            return 'Low Activity & Irregular'
                    
                    analysis_df['activity_segment'] = analysis_df.apply(categorize_activity, axis=1)
                    
                    # Activity segment analysis
                    activity_analysis = analysis_df.groupby('activity_segment').agg({
                        'churn_prediction': 'mean',
                        'user_id': 'count'
                    }).reset_index()
                    
                    fig_activity = px.treemap(
                        activity_analysis,
                        path=['activity_segment'],
                        values='user_id',
                        color='churn_prediction',
                        color_continuous_scale='RdYlGn_r',
                        title="Customer Activity Segments (Size=Count, Color=Churn Rate)"
                    )
                    st.plotly_chart(fig_activity, use_container_width=True)
        
        with eda_tab4:
            st.markdown("### Advanced Customer Segmentation")
            
            # Multi-dimensional segmentation
            segmentation_features = []
            
            # Revenue segmentation
            if 'REVENUE' in analysis_df.columns:
                analysis_df['revenue_segment'] = pd.qcut(analysis_df['REVENUE'], 
                                                       q=4, labels=['Low', 'Medium', 'High', 'Premium'])
                segmentation_features.append('revenue_segment')
            
            # Usage segmentation
            if 'FREQUENCE_RECH' in analysis_df.columns:
                analysis_df['usage_segment'] = pd.qcut(analysis_df['FREQUENCE_RECH'], 
                                                     q=3, labels=['Light', 'Medium', 'Heavy'])
                segmentation_features.append('usage_segment')
            
            # Create comprehensive segmentation
            if len(segmentation_features) >= 2:
                # Cross-segmentation analysis
                cross_segment = pd.crosstab(analysis_df[segmentation_features[0]], 
                                          analysis_df[segmentation_features[1]], 
                                          analysis_df['churn_prediction'], 
                                          aggfunc='mean').round(3)
                
                fig_cross = px.imshow(
                    cross_segment,
                    title="Churn Rate by Revenue and Usage Segments",
                    color_continuous_scale='Reds',
                    aspect="auto"
                )
                st.plotly_chart(fig_cross, use_container_width=True)
                
                # Segment performance matrix
                segment_matrix = analysis_df.groupby(segmentation_features).agg({
                    'churn_prediction': 'mean',
                    'churn_probability': 'mean',
                    'user_id': 'count'
                }).reset_index()
                
                # Create bubble chart for segment analysis
                fig_bubble = px.scatter(
                    segment_matrix,
                    x='churn_prediction',
                    y='churn_probability',
                    size='user_id',
                    color=segmentation_features[0],
                    hover_data=segmentation_features,
                    title="Customer Segment Performance Matrix",
                    labels={'churn_prediction': 'Actual Churn Rate', 
                           'churn_probability': 'Predicted Churn Probability'}
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
                
                # Segment priorities
                st.markdown("#### Segment-Based Recommendations")
                
                # High-value, high-risk segments
                high_risk_segments = segment_matrix[segment_matrix['churn_prediction'] > 0.3]
                large_segments = segment_matrix[segment_matrix['user_id'] > segment_matrix['user_id'].median()]
                
                priority_segments = pd.merge(high_risk_segments, large_segments, how='inner')
                
                if not priority_segments.empty:
                    st.write("**Priority Segments (High Risk + Large Volume):**")
                    for _, row in priority_segments.iterrows():
                        segment_desc = " & ".join([f"{row[feat]}" for feat in segmentation_features])
                        st.write(f"• {segment_desc}: {row['churn_prediction']:.1%} churn rate, {row['user_id']:,} customers")
            
            # Advanced clustering using multiple features
            if len(num_cols) >= 4:
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                
                # Prepare features for clustering
                cluster_features = [col for col in num_cols[:6] if col in analysis_df.columns]
                X_cluster = analysis_df[cluster_features].fillna(analysis_df[cluster_features].median())
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                
                # DBSCAN clustering for outlier detection
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = dbscan.fit_predict(X_scaled)
                analysis_df['behavior_cluster'] = clusters
                
                # Cluster analysis
                cluster_summary = analysis_df.groupby('behavior_cluster').agg({
                    'churn_prediction': 'mean',
                    'churn_probability': 'mean',
                    'user_id': 'count'
                }).reset_index()
                
                # Outlier analysis (cluster -1 in DBSCAN)
                outliers = analysis_df[analysis_df['behavior_cluster'] == -1]
                
                if len(outliers) > 0:
                    st.markdown("#### Behavioral Outliers Analysis")
                    
                    outlier_churn_rate = outliers['churn_prediction'].mean()
                    normal_churn_rate = analysis_df[analysis_df['behavior_cluster'] != -1]['churn_prediction'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Outlier Customers", f"{len(outliers):,}")
                    with col2:
                        st.metric("Outlier Churn Rate", f"{outlier_churn_rate:.1%}")
                    with col3:
                        st.metric("Risk vs Normal", f"{((outlier_churn_rate/normal_churn_rate - 1) * 100):+.0f}%")
                    
                    if outlier_churn_rate > normal_churn_rate * 1.2:
                        st.warning("⚠️ Behavioral outliers show significantly higher churn risk - investigate unusual patterns")
        
        with eda_tab5:
            st.markdown("### Business Intelligence Dashboard")
            
            # KPI Dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate advanced KPIs
            total_customers = len(analysis_df)
            predicted_churners = (analysis_df['churn_prediction'] == 1).sum()
            high_risk_customers = (analysis_df['churn_probability'] >= 0.7).sum()
            
            if 'REVENUE' in analysis_df.columns:
                total_revenue = analysis_df['REVENUE'].sum()
                at_risk_revenue = analysis_df[analysis_df['churn_prediction'] == 1]['REVENUE'].sum()
                avg_revenue_per_customer = analysis_df['REVENUE'].mean()
                revenue_at_risk_pct = (at_risk_revenue / total_revenue * 100) if total_revenue > 0 else 0
            else:
                total_revenue = 0
                at_risk_revenue = 0
                avg_revenue_per_customer = 0
                revenue_at_risk_pct = 0
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{predicted_churners:,}</h3>
                    <p>Predicted Churners</p>
                    <small>{(predicted_churners/total_customers*100):.1f}% of total</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{high_risk_customers:,}</h3>
                    <p>High Risk (≥70%)</p>
                    <small>Immediate attention needed</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>${at_risk_revenue:,.0f}</h3>
                    <p>Revenue at Risk</p>
                    <small>{revenue_at_risk_pct:.1f}% of total revenue</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # Calculate retention ROI potential
                retention_cost_per_customer = 50  # Assumed retention cost
                retention_success_rate = 0.3  # Assumed 30% success rate
                
                potential_saved_revenue = at_risk_revenue * retention_success_rate
                retention_cost = predicted_churners * retention_cost_per_customer
                roi = ((potential_saved_revenue - retention_cost) / retention_cost * 100) if retention_cost > 0 else 0
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{roi:.0f}%</h3>
                    <p>Retention Campaign ROI</p>
                    <small>Estimated with 30% success</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Executive Summary
            st.markdown("### Executive Summary & Strategic Insights")
            
            # Generate data-driven insights
            insights = []
            
            # Churn rate insights
            overall_churn_rate = predicted_churners / total_customers
            if overall_churn_rate > 0.20:
                insights.append("🔴 **Critical Alert**: Churn rate exceeds 20% - immediate strategic intervention required")
            elif overall_churn_rate > 0.15:
                insights.append("🟡 **High Alert**: Churn rate above industry average - enhanced retention programs needed")
            else:
                insights.append("🟢 **Stable**: Churn rate within acceptable range - maintain current strategies")
            
            # Revenue insights
            if revenue_at_risk_pct > 30:
                insights.append(f"💰 **Revenue Risk**: {revenue_at_risk_pct:.1f}% of revenue at risk - prioritize high-value customer retention")
            
            # Geographic insights
            if 'REGION' in analysis_df.columns:
                region_analysis = analysis_df.groupby('REGION')['churn_prediction'].mean()
                highest_risk_region = region_analysis.idxmax()
                lowest_risk_region = region_analysis.idxmin()
                insights.append(f"🗺️ **Geographic**: {highest_risk_region} shows highest risk, {lowest_risk_region} shows lowest - regional strategy needed")
            
            # Feature insights
            if hasattr(locals(), 'feature_analysis') and not feature_analysis.empty:
                top_differentiator = feature_analysis.iloc[0]['feature']
                insights.append(f"📊 **Key Factor**: {top_differentiator} is the strongest churn predictor - focus retention efforts here")
            
            for insight in insights:
                st.write(insight)
            
            # Action Plan Generator
            st.markdown("### Recommended Action Plan")
            
            action_plan = []
            
            # Immediate actions (next 30 days)
            immediate_actions = [
                f"Contact {high_risk_customers:,} customers with ≥70% churn probability",
                "Launch targeted retention campaign for high-value at-risk customers",
                "Implement enhanced monitoring for customers moving into risk categories"
            ]
            
            # Short-term actions (1-3 months)
            short_term_actions = [
                "Develop segment-specific retention strategies based on analysis",
                "A/B test retention offers on medium-risk customer segments",
                "Enhance customer experience in highest-risk regions"
            ]
            
            # Long-term actions (3-12 months)
            long_term_actions = [
                "Implement predictive alerting system for churn risk escalation",
                "Develop customer lifetime value optimization programs",
                "Build automated retention workflows based on risk scores"
            ]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Immediate (30 days)")
                for action in immediate_actions:
                    st.write(f"• {action}")
            
            with col2:
                st.markdown("#### Short-term (1-3 months)")
                for action in short_term_actions:
                    st.write(f"• {action}")
            
            with col3:
                st.markdown("#### Long-term (3-12 months)")
                for action in long_term_actions:
                    st.write(f"• {action}")
            
            # ROI Calculator
            st.markdown("### ROI Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Retention Campaign Parameters")
                campaign_cost_per_customer = st.slider("Retention Cost per Customer ($)", 10, 200, 50)
                success_rate = st.slider("Expected Success Rate (%)", 10, 80, 30) / 100
                campaign_target = st.selectbox("Target Segment", 
                                             ["All Predicted Churners", "High Risk Only (≥70%)", "Critical Risk Only (≥90%)"])
            
            with col2:
                # Calculate ROI based on selections
                if campaign_target == "All Predicted Churners":
                    target_customers = predicted_churners
                    target_revenue = at_risk_revenue
                elif campaign_target == "High Risk Only (≥70%)":
                    target_customers = high_risk_customers
                    target_revenue = analysis_df[analysis_df['churn_probability'] >= 0.7]['REVENUE'].sum() if 'REVENUE' in analysis_df.columns else 0
                else:  # Critical Risk
                    critical_risk_customers = (analysis_df['churn_probability'] >= 0.9).sum()
                    target_customers = critical_risk_customers
                    target_revenue = analysis_df[analysis_df['churn_probability'] >= 0.9]['REVENUE'].sum() if 'REVENUE' in analysis_df.columns else 0
                
                campaign_cost = target_customers * campaign_cost_per_customer
                expected_revenue_saved = target_revenue * success_rate
                net_benefit = expected_revenue_saved - campaign_cost
                roi_percentage = (net_benefit / campaign_cost * 100) if campaign_cost > 0 else 0
                
                st.markdown("#### Campaign ROI Projection")
                st.write(f"**Target Customers**: {target_customers:,}")
                st.write(f"**Campaign Cost**: ${campaign_cost:,.0f}")
                st.write(f"**Expected Revenue Saved**: ${expected_revenue_saved:,.0f}")
                st.write(f"**Net Benefit**: ${net_benefit:,.0f}")
                st.write(f"**ROI**: {roi_percentage:.0f}%")
                
                if roi_percentage > 100:
                    st.success("✅ Highly profitable campaign - proceed with implementation")
                elif roi_percentage > 50:
                    st.info("💡 Profitable campaign - good investment opportunity")
                else:
                    st.warning("⚠️ Low ROI - consider refining targeting or reducing costs")
        
        # Enhanced Export Section
        st.markdown("### 📥 Export Comprehensive Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Detailed customer analysis
            export_df = analysis_df.copy()
            export_df['risk_category'] = pd.cut(y_proba, 
                                              bins=[0, 0.25, 0.5, 0.75, 1.0],
                                              labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
            
            if 'REVENUE' in export_df.columns:
                export_df['revenue_percentile'] = export_df['REVENUE'].rank(pct=True)
            
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 Detailed Analysis CSV",
                data=csv_data,
                file_name=f"comprehensive_churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Executive summary report
            if hasattr(locals(), 'feature_analysis') and not feature_analysis.empty:
                summary_data = {
                    'Metric': ['Total Customers', 'Predicted Churners', 'Churn Rate (%)', 
                             'High Risk Customers', 'Revenue at Risk ($)', 'Revenue at Risk (%)',
                             'Top Risk Factor', 'Highest Risk Region'],
                    'Value': [total_customers, predicted_churners, f"{overall_churn_rate:.1%}",
                            high_risk_customers, f"{at_risk_revenue:,.0f}", f"{revenue_at_risk_pct:.1f}%",
                            feature_analysis.iloc[0]['feature'] if not feature_analysis.empty else 'N/A',
                            region_analysis.idxmax() if 'REGION' in analysis_df.columns else 'N/A']
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_csv = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📋 Executive Summary CSV",
                    data=summary_csv,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Action plan export
            action_data = {
                'Timeline': ['Immediate'] * len(immediate_actions) + 
                          ['Short-term'] * len(short_term_actions) + 
                          ['Long-term'] * len(long_term_actions),
                'Action': immediate_actions + short_term_actions + long_term_actions,
                'Priority': ['High'] * len(immediate_actions) + 
                          ['Medium'] * len(short_term_actions) + 
                          ['Low'] * len(long_term_actions)
            }
            
            action_df = pd.DataFrame(action_data)
            action_csv = action_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📝 Action Plan CSV",
                data=action_csv,
                file_name=f"action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("🔄 Process data in the Prediction tab to unlock advanced analytics and insights")

with tab4:
    st.markdown("## Help & Documentation")
    
    st.markdown("### Required Columns")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Features:**")
        for col in num_cols:
            st.write(f"• {col}")
        st.markdown("**Tenure:**")
        st.write("• TENURE")
    
    with col2:
        st.markdown("**Categorical Features:**")
        for col in cat_cols_low + cat_cols_high:
            st.write(f"• {col}")
        st.markdown("**Identifier:**")
        st.write("• user_id")
    
    st.markdown("### Data Requirements")
    st.write("1. CSV format with proper headers")
    st.write("2. All required columns must be present")
    st.write("3. user_id column for identification")
    st.write("4. No special characters in column names")
    
    st.markdown("### Troubleshooting")
    st.write("• **File won't read**: Try different encodings or separators")
    st.write("• **Missing columns**: Ensure all required columns are present")
    st.write("• **Prediction fails**: Check data types and values")
    st.write("• **Slow processing**: Large files take more time")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Expresso Churn Predictor | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)