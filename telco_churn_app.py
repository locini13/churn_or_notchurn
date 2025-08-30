import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = {}

def show_start_page():
    """Display the start page with problem statement and dataset overview"""
    st.markdown('<h1 class="main-header">Telco Customer Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Problem Statement Card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Problem Statement")
        st.write("""
        The telecommunications industry faces a high rate of customer churn, which directly impacts profitability. 
        Retaining existing customers is more cost-effective than acquiring new ones. Using customer demographic, 
        service usage, and billing data, we aim to build classification models to predict and analyze churn behavior.
        """)
        st.write("""
        **Formal Definition:** Given a dataset of telco customers with demographic, account, service, 
        and billing information, predict whether a customer is likely to churn (leave the company) or stay. 
        This is a supervised classification problem.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset Overview Card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##  Dataset Overview - Telco Customer Churn")
        st.write("**Source:** IBM Telco Customer Churn dataset")
        st.write("**Rows:** ~7,043 customers")
        st.write("**Columns:** 21 attributes related to customer demographics, account information, services subscribed, and billing")
        st.write("**Target Variable:** **Churn (Yes/No)** → whether a customer left the company or not")
        
        st.markdown("###  Key Features Include:")
        st.write("• **Demographics:** Gender, SeniorCitizen, Partner, Dependents")
        st.write("• **Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod")
        st.write("• **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.")
        st.write("• **Billing:** MonthlyCharges, TotalCharges")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Machine Learning Objectives
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##  Machine Learning Objectives")
        st.write("1. **Classification Models:** Compare Decision Tree, KNN, Naive Bayes, and Random Forest")
        st.write("2. **Model Optimization:** Use GridSearchCV and hyperparameter tuning")
        st.write("3. **Performance Analysis:** Evaluate using F1-score, accuracy, and confusion matrices")
        st.write("4. **Feature Importance:** Identify key factors influencing customer churn")
        st.write("5. **Business Insights:** Provide actionable recommendations for retention strategies")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Proceed button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(" Proceed to Analysis", use_container_width=True, key="proceed_btn"):
            st.session_state.show_analysis = True
            st.rerun()

def preprocess_data(df):
    """Preprocess the dataset for machine learning"""
    try:
        df = df.copy()
        
        # Convert TotalCharges to numeric (handle spaces and missing values)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Label encode categorical variables
        le = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return df

def validate_dataset(df):
    """Validate if the dataset is suitable for analysis"""
    if df is None:
        return False, "No dataset uploaded"
    
    if df.empty:
        return False, "Dataset is empty"
    
    if df.shape[1] < 2:
        return False, "Dataset needs at least 2 columns"
    
    return True, "Dataset is valid"

def show_analysis_dashboard():
    """Main analysis dashboard"""
    st.markdown('<h1 class="main-header"> Telco Customer Churn Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Reset button
    if st.sidebar.button(" Back to Home", use_container_width=True):
        st.session_state.show_analysis = False
        st.session_state.data = None
        st.session_state.results = {}
        st.rerun()
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            # Validate dataset
            is_valid, message = validate_dataset(df)
            if not is_valid:
                st.error(message)
                return
            
            # Algorithm selection
            algorithm = st.sidebar.selectbox(
                "Select Algorithm",
                ["Decision Tree", "KNN", "Naive Bayes", "Random Forest", "All Models"]
            )
            
            # Display basic info
            st.sidebar.success(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Main content tabs
            tabs = st.tabs([
                " Dataset Overview", 
                " EDA", 
                " Decision Tree", 
                " Naive Bayes", 
                " KNN", 
                " Random Forest", 
                "📈 Summary"
            ])
            
            with tabs[0]:
                show_dataset_overview(df)
            
            with tabs[1]:
                show_eda(df)
            
            with tabs[2]:
                show_decision_tree_analysis(df, algorithm)
            
            with tabs[3]:
                show_naive_bayes_analysis(df, algorithm)
            
            with tabs[4]:
                show_knn_analysis(df, algorithm)
            
            with tabs[5]:
                show_random_forest_analysis(df, algorithm)
            
            with tabs[6]:
                show_final_summary()
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.info("Please make sure you've uploaded a valid CSV file.")
    
    else:
        st.info(" Please upload a CSV file to begin analysis.")
        st.markdown("""
        ### Expected Dataset Format:
        - CSV file with customer data
        - Target column named 'Churn' (or will use last column)
        - Mix of categorical and numerical features
        - Recommended: IBM Telco Customer Churn dataset
        """)

def show_dataset_overview(df):
    """Display dataset overview and basic statistics"""
    st.header(" Dataset Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df.shape[0]:,}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df.shape[1]}</h3>
            <p>Total Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df.isnull().sum().sum()}</h3>
            <p>Null Values</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df.duplicated().sum()}</h3>
            <p>Duplicates</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset preview and info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader(" Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Statistical summary
    if len(df.select_dtypes(include=[np.number]).columns) > 0:
        st.subheader(" Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

def show_eda(df):
    """Exploratory Data Analysis"""
    st.header("🔍 Exploratory Data Analysis")
    
    # Target variable analysis
    target_col = 'Churn' if 'Churn' in df.columns else df.columns[-1]
    
    if target_col in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f" {target_col} Distribution")
            fig = px.pie(df, names=target_col, title=f'{target_col} Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(f" {target_col} Count")
            counts = df[target_col].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, title=f'{target_col} Count')
            fig.update_layout(xaxis_title=target_col, yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numerical_cols:
        st.subheader(" Numerical Features Analysis")
        
        # Select columns for analysis
        selected_cols = st.multiselect(
            "Select numerical columns to analyze:",
            numerical_cols,
            default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols
        )
        
        if selected_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Histograms**")
                for col in selected_cols:
                    fig = px.histogram(df, x=col, title=f'Distribution of {col}', nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Box Plots**")
                for col in selected_cols:
                    fig = px.box(df, y=col, title=f'Box Plot of {col}')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    if len(numerical_cols) > 1:
        st.subheader(" Correlation Analysis")
        corr_matrix = df[numerical_cols].corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto", 
            title="Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # High correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            st.subheader(" High Correlation Pairs (|r| > 0.7)")
            st.dataframe(pd.DataFrame(high_corr_pairs))

def show_decision_tree_analysis(df, algorithm):
    """Decision Tree analysis with hyperparameter tuning"""
    if algorithm not in ["Decision Tree", "All Models"]:
        st.info("Select 'Decision Tree' or 'All Models' to see this analysis.")
        return
        
    st.header(" Decision Tree Analysis")
    
    try:
        # Preprocess data
        df_processed = preprocess_data(df)
        target_col = 'Churn' if 'Churn' in df_processed.columns else df_processed.columns[-1]
        
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Manual depth tuning
        st.subheader(" Hyperparameter Tuning - Tree Depth")
        depths = range(1, 21)
        train_f1_scores = []
        test_f1_scores = []
        
        progress_bar = st.progress(0)
        for i, depth in enumerate(depths):
            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(X_train, y_train)
            
            train_pred = dt.predict(X_train)
            test_pred = dt.predict(X_test)
            
            train_f1_scores.append(f1_score(y_train, train_pred))
            test_f1_scores.append(f1_score(y_test, test_pred))
            
            progress_bar.progress((i + 1) / len(depths))
        
        # Plot F1 vs Depth
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(depths), y=train_f1_scores, 
                                mode='lines+markers', name='Training F1'))
        fig.add_trace(go.Scatter(x=list(depths), y=test_f1_scores, 
                                mode='lines+markers', name='Testing F1'))
        fig.update_layout(
            title='F1 Score vs Tree Depth',
            xaxis_title='Depth',
            yaxis_title='F1 Score'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best depth
        best_depth_idx = np.argmax(test_f1_scores)
        best_depth = depths[best_depth_idx]
        best_f1 = test_f1_scores[best_depth_idx]
        
        st.success(f" Best Depth: {best_depth} with Test F1 Score: {best_f1:.4f}")
        
        # GridSearchCV
        st.subheader(" Grid Search Optimization")
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        }
        
        dt_grid = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt_grid, param_grid, cv=5, scoring='f1', n_jobs=-1)
        
        with st.spinner('Performing Grid Search...'):
            grid_search.fit(X_train, y_train)
        
        # Final model evaluation
        best_dt = grid_search.best_estimator_
        y_pred = best_dt.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Model Performance")
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Best Parameters", "")
            st.json(grid_search.best_params_)
        
        with col2:
            st.subheader(" Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if hasattr(best_dt, 'feature_importances_'):
            st.subheader(" Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_dt.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), 
                        x='importance', y='feature', 
                        orientation='h', title='Top 10 Feature Importances')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Store results
        st.session_state.results['Decision Tree'] = {
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Best Params': grid_search.best_params_
        }
        
    except Exception as e:
        st.error(f"Error in Decision Tree analysis: {str(e)}")

def show_naive_bayes_analysis(df, algorithm):
    """Naive Bayes analysis with different variants"""
    if algorithm not in ["Naive Bayes", "All Models"]:
        st.info("Select 'Naive Bayes' or 'All Models' to see this analysis.")
        return
        
    st.header(" Naive Bayes Analysis")
    
    try:
        # Preprocess data
        df_processed = preprocess_data(df)
        target_col = 'Churn' if 'Churn' in df_processed.columns else df_processed.columns[-1]
        
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data for Gaussian NB
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Gaussian NB
        st.subheader(" Gaussian Naive Bayes")
        gnb = GaussianNB()
        gnb.fit(X_train_scaled, y_train)
        gnb_pred = gnb.predict(X_test_scaled)
        gnb_f1 = f1_score(y_test, gnb_pred)
        gnb_acc = accuracy_score(y_test, gnb_pred)
        
        # Bernoulli NB
        st.subheader(" Bernoulli Naive Bayes")
        bnb = BernoulliNB()
        bnb.fit(X_train, y_train)
        bnb_pred = bnb.predict(X_test)
        bnb_f1 = f1_score(y_test, bnb_pred)
        bnb_acc = accuracy_score(y_test, bnb_pred)
        
        # Results comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Gaussian NB Results")
            st.metric("F1 Score", f"{gnb_f1:.4f}")
            st.metric("Accuracy", f"{gnb_acc:.4f}")
            
            cm = confusion_matrix(y_test, gnb_pred)
            fig = px.imshow(cm, text_auto=True, title="Gaussian NB Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(" Bernoulli NB Results")
            st.metric("F1 Score", f"{bnb_f1:.4f}")
            st.metric("Accuracy", f"{bnb_acc:.4f}")
            
            cm = confusion_matrix(y_test, bnb_pred)
            fig = px.imshow(cm, text_auto=True, title="Bernoulli NB Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.subheader(" Model Comparison")
        comparison_df = pd.DataFrame({
            'Model': ['Gaussian NB', 'Bernoulli NB'],
            'F1 Score': [gnb_f1, bnb_f1],
            'Accuracy': [gnb_acc, bnb_acc]
        })
        
        fig = px.bar(comparison_df, x='Model', y='F1 Score', 
                    title='Naive Bayes Models Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # VIF Analysis
        if len(X.columns) > 1:
            st.subheader(" Variance Inflation Factor (VIF)")
            try:
                vif_data = pd.DataFrame()
                vif_data["Feature"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                                  for i in range(len(X.columns))]
                vif_data = vif_data.sort_values('VIF', ascending=False)
                
                # Color code VIF values
                def color_vif(val):
                    if val > 10:
                        return 'background-color: red; color: white'
                    elif val > 5:
                        return 'background-color: orange; color: white'
                    else:
                        return 'background-color: green; color: white'
                
                st.dataframe(vif_data.style.applymap(color_vif, subset=['VIF']))
                
                st.info("""
                **VIF Interpretation:**
                - VIF = 1: No multicollinearity
                - VIF = 1-5: Moderate multicollinearity
                - VIF = 5-10: High multicollinearity
                - VIF > 10: Very high multicollinearity (consider removal)
                """)
                
            except Exception as e:
                st.warning(f"Could not calculate VIF: {str(e)}")
        
        # Store results
        best_model = 'Gaussian NB' if gnb_f1 > bnb_f1 else 'Bernoulli NB'
        best_f1 = max(gnb_f1, bnb_f1)
        st.session_state.results['Naive Bayes'] = {
            'F1 Score': best_f1,
            'Best Model': best_model,
            'Gaussian F1': gnb_f1,
            'Bernoulli F1': bnb_f1
        }
        
    except Exception as e:
        st.error(f"Error in Naive Bayes analysis: {str(e)}")

def show_knn_analysis(df, algorithm):
    """KNN analysis with optimal K selection"""
    if algorithm not in ["KNN", "All Models"]:
        st.info("Select 'KNN' or 'All Models' to see this analysis.")
        return
        
    st.header("👥 K-Nearest Neighbors Analysis")
    
    try:
        # Preprocess data
        df_processed = preprocess_data(df)
        target_col = 'Churn' if 'Churn' in df_processed.columns else df_processed.columns[-1]
        
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data (essential for KNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Elbow curve for optimal K
        st.subheader(" Finding Optimal K Value")
        k_range = range(1, min(31, len(X_train) // 5))  # Ensure we don't exceed reasonable limits
        train_f1_scores = []
        test_f1_scores = []
        
        progress_bar = st.progress(0)
        for i, k in enumerate(k_range):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            
            train_pred = knn.predict(X_train_scaled)
            test_pred = knn.predict(X_test_scaled)
            
            train_f1_scores.append(f1_score(y_train, train_pred))
            test_f1_scores.append(f1_score(y_test, test_pred))
            
            progress_bar.progress((i + 1) / len(k_range))
        
        # Plot elbow curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=train_f1_scores, 
                                mode='lines+markers', name='Training F1'))
        fig.add_trace(go.Scatter(x=list(k_range), y=test_f1_scores, 
                                mode='lines+markers', name='Testing F1'))
        fig.update_layout(
            title='KNN Elbow Curve (F1 Score vs K)',
            xaxis_title='K Value',
            yaxis_title='F1 Score'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best K
        best_k_idx = np.argmax(test_f1_scores)
        best_k = list(k_range)[best_k_idx]
        best_f1 = test_f1_scores[best_k_idx]
        
        st.success(f" Optimal K: {best_k} with Test F1 Score: {best_f1:.4f}")
        
        # Cross-validation with best K
        st.subheader(" Cross-Validation Analysis")
        knn_best = KNeighborsClassifier(n_neighbors=best_k)
        cv_scores = cross_val_score(knn_best, X_train_scaled, y_train, cv=5, scoring='f1')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Cross-Validation Results")
            st.metric("Mean CV F1 Score", f"{cv_scores.mean():.4f}")
            st.metric("Std CV F1 Score", f"{cv_scores.std():.4f}")
            
            # CV scores distribution
            fig = px.box(y=cv_scores, title='Cross-Validation F1 Scores Distribution')
            fig.update_layout(yaxis_title='F1 Score')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(" Final Model Evaluation")
            knn_best.fit(X_train_scaled, y_train)
            y_pred = knn_best.predict(X_test_scaled)
            final_f1 = f1_score(y_test, y_pred)
            final_acc = accuracy_score(y_test, y_pred)
            
            st.metric("Test F1 Score", f"{final_f1:.4f}")
            st.metric("Test Accuracy", f"{final_acc:.4f}")
            
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title="KNN Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Distance metrics comparison
        st.subheader(" Distance Metrics Comparison")
        distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
        metric_results = []
        
        for metric in distance_metrics:
            knn_metric = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
            knn_metric.fit(X_train_scaled, y_train)
            pred = knn_metric.predict(X_test_scaled)
            f1 = f1_score(y_test, pred)
            acc = accuracy_score(y_test, pred)
            metric_results.append({'Metric': metric, 'F1 Score': f1, 'Accuracy': acc})
        
        metric_df = pd.DataFrame(metric_results)
        fig = px.bar(metric_df, x='Metric', y='F1 Score', 
                    title='F1 Score by Distance Metric')
        st.plotly_chart(fig, use_container_width=True)
        
        # Store results
        st.session_state.results['KNN'] = {
            'F1 Score': final_f1,
            'Accuracy': final_acc,
            'Best K': best_k,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        }
        
    except Exception as e:
        st.error(f"Error in KNN analysis: {str(e)}")

def show_random_forest_analysis(df, algorithm):
    """Random Forest analysis with hyperparameter tuning"""
    if algorithm not in ["Random Forest", "All Models"]:
        st.info("Select 'Random Forest' or 'All Models' to see this analysis.")
        return
        
    st.header(" Random Forest Analysis")
    
    try:
        # Preprocess data
        df_processed = preprocess_data(df)
        target_col = 'Churn' if 'Churn' in df_processed.columns else df_processed.columns[-1]
        
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Default Random Forest
        st.subheader(" Baseline Random Forest")
        rf_default = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_default.fit(X_train, y_train)
        default_pred = rf_default.predict(X_test)
        default_f1 = f1_score(y_test, default_pred)
        default_acc = accuracy_score(y_test, default_pred)
        
        # Number of estimators analysis
        st.subheader(" Optimal Number of Estimators")
        n_estimators_range = [10, 50, 100, 200, 300, 500]
        estimator_f1_scores = []
        
        progress_bar = st.progress(0)
        for i, n_est in enumerate(n_estimators_range):
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            estimator_f1_scores.append(f1_score(y_test, pred))
            progress_bar.progress((i + 1) / len(n_estimators_range))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=n_estimators_range, y=estimator_f1_scores, 
                                mode='lines+markers', name='F1 Score'))
        fig.update_layout(
            title='F1 Score vs Number of Estimators',
            xaxis_title='Number of Estimators',
            yaxis_title='F1 Score'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # GridSearchCV tuning
        st.subheader(" Advanced Hyperparameter Tuning")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        with st.spinner('Performing Grid Search (this may take a while)...'):
            grid_search = GridSearchCV(rf_grid, param_grid, cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
        
        tuned_pred = grid_search.predict(X_test)
        tuned_f1 = f1_score(y_test, tuned_pred)
        tuned_acc = accuracy_score(y_test, tuned_pred)
        
        # Results comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Default Random Forest")
            st.metric("F1 Score", f"{default_f1:.4f}")
            st.metric("Accuracy", f"{default_acc:.4f}")
            
            cm = confusion_matrix(y_test, default_pred)
            fig = px.imshow(cm, text_auto=True, title="Default RF Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(" Tuned Random Forest")
            st.metric("F1 Score", f"{tuned_f1:.4f}")
            st.metric("Accuracy", f"{tuned_acc:.4f}")
            st.metric("Improvement", f"{((tuned_f1 - default_f1) / default_f1 * 100):+.2f}%")
            
            cm = confusion_matrix(y_test, tuned_pred)
            fig = px.imshow(cm, text_auto=True, title="Tuned RF Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Best parameters
        st.subheader(" Best Parameters")
        st.json(grid_search.best_params_)
        
        # Feature importance analysis
        st.subheader(" Feature Importance Analysis")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top 15 features
            fig = px.bar(feature_importance.head(15), 
                        x='importance', y='feature', 
                        orientation='h', title='Top 15 Feature Importances')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(" Feature Stats")
            st.metric("Total Features", len(feature_importance))
            st.metric("Features > 0.05", len(feature_importance[feature_importance['importance'] > 0.05]))
            st.metric("Features > 0.01", len(feature_importance[feature_importance['importance'] > 0.01]))
            
            # Show top 10 features table
            st.dataframe(feature_importance.head(10))
        
        # Model performance over trees
        st.subheader(" Learning Curve")
        train_scores = []
        test_scores = []
        tree_range = range(1, min(201, grid_search.best_params_.get('n_estimators', 100) + 1), 10)
        
        for n_trees in tree_range:
            rf_temp = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=grid_search.best_params_.get('max_depth'),
                min_samples_split=grid_search.best_params_.get('min_samples_split'),
                min_samples_leaf=grid_search.best_params_.get('min_samples_leaf'),
                random_state=42,
                n_jobs=-1
            )
            rf_temp.fit(X_train, y_train)
            
            train_pred = rf_temp.predict(X_train)
            test_pred = rf_temp.predict(X_test)
            
            train_scores.append(f1_score(y_train, train_pred))
            test_scores.append(f1_score(y_test, test_pred))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(tree_range), y=train_scores, 
                                mode='lines', name='Training F1'))
        fig.add_trace(go.Scatter(x=list(tree_range), y=test_scores, 
                                mode='lines', name='Testing F1'))
        fig.update_layout(
            title='Learning Curve - F1 Score vs Number of Trees',
            xaxis_title='Number of Trees',
            yaxis_title='F1 Score'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Store results
        st.session_state.results['Random Forest'] = {
            'F1 Score': tuned_f1,
            'Accuracy': tuned_acc,
            'Default F1': default_f1,
            'Best Params': grid_search.best_params_,
            'Improvement': (tuned_f1 - default_f1) / default_f1 * 100
        }
        
    except Exception as e:
        st.error(f"Error in Random Forest analysis: {str(e)}")

def show_final_summary():
    """Display final model comparison and insights"""
    st.header(" Final Model Comparison & Insights")
    
    if st.session_state.results:
        # Create comparison dataframe
        comparison_data = []
        for model, metrics in st.session_state.results.items():
            comparison_data.append({
                'Model': model,
                'F1 Score': metrics.get('F1 Score', 0),
                'Accuracy': metrics.get('Accuracy', 0) if 'Accuracy' in metrics else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        # Performance comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Model Performance Ranking")
            
            # Styled dataframe
            def highlight_best(s):
                is_max = s == s.max()
                return ['background-color: gold' if v else '' for v in is_max]
            
            styled_df = comparison_df.style.apply(highlight_best, subset=['F1 Score'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Performance chart
            fig = px.bar(comparison_df, x='Model', y='F1 Score', 
                        title='Model Performance Comparison (F1 Score)',
                        color='F1 Score', color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Best model details
            if len(comparison_df) > 0:
                best_model = comparison_df.iloc[0]
                st.subheader("🥇 Best Model")
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{best_model['Model']}</h3>
                    <p>F1 Score: <strong>{best_model['F1 Score']:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model-specific insights
                model_name = best_model['Model']
                if model_name in st.session_state.results:
                    st.subheader("Model Details")
                    model_details = st.session_state.results[model_name]
                    
                    for key, value in model_details.items():
                        if key not in ['F1 Score', 'Accuracy']:
                            if isinstance(value, dict):
                                st.json({key: value})
                            else:
                                st.metric(key.replace('_', ' ').title(), str(value))
        
        # Detailed Analysis
        st.subheader("🔍 Detailed Model Analysis")
        
        # Performance metrics comparison
        if len(comparison_df) > 1:
            metrics_data = []
            for model, results in st.session_state.results.items():
                metrics_data.append({
                    'Model': model,
                    'F1 Score': results.get('F1 Score', 0),
                    'Accuracy': results.get('Accuracy', 0) if isinstance(results.get('Accuracy'), (int, float)) else 0
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=metrics_df['Model'], y=metrics_df['F1 Score'], 
                                    mode='markers+lines', name='F1 Score', 
                                    marker=dict(size=10)))
            fig.add_trace(go.Scatter(x=metrics_df['Model'], y=metrics_df['Accuracy'], 
                                    mode='markers+lines', name='Accuracy', 
                                    marker=dict(size=10)))
            fig.update_layout(title='F1 Score vs Accuracy Comparison',
                             yaxis_title='Score')
            st.plotly_chart(fig, use_container_width=True)
        
        # Business Insights
        st.subheader("💼 Business Insights & Recommendations")
        
        best_f1 = comparison_df.iloc[0]['F1 Score'] if len(comparison_df) > 0 else 0
        
        insights = []
        
        if best_f1 > 0.8:
            insights.append(" **Excellent Model Performance**: The best model achieves high predictive accuracy, suitable for production deployment.")
        elif best_f1 > 0.7:
            insights.append(" **Good Model Performance**: The models show promising results with room for improvement through feature engineering.")
        else:
            insights.append(" **Moderate Performance**: Consider collecting more data or exploring advanced techniques like ensemble methods.")
        
        # Model-specific recommendations
        if len(comparison_df) > 0:
            best_model_name = comparison_df.iloc[0]['Model']
            
            if 'Random Forest' in best_model_name:
                insights.append(" **Random Forest Advantage**: Provides feature importance rankings and handles mixed data types well.")
            elif 'Decision Tree' in best_model_name:
                insights.append("**Decision Tree Advantage**: Offers interpretable rules that can be easily explained to stakeholders.")
            elif 'KNN' in best_model_name:
                insights.append(" **KNN Advantage**: Non-parametric approach that adapts well to local patterns in data.")
            elif 'Naive Bayes' in best_model_name:
                insights.append(" **Naive Bayes Advantage**: Fast training and prediction, works well with limited data.")
        
        insights.extend([
            " **Feature Importance**: Focus on the top contributing features for targeted retention strategies.",
            " **Model Monitoring**: Regularly retrain models as customer behavior patterns may change over time.",
            
            " **Customer Segmentation**: Use model insights to create targeted retention campaigns for high-risk customers."
        ])
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Export results option
        st.subheader(" Export Results")
        
        if st.button(" Generate Detailed Report", use_container_width=True):
            # Create detailed report
            report = {
                'Model Performance': comparison_df.to_dict('records'),
                'Best Model': comparison_df.iloc[0].to_dict() if len(comparison_df) > 0 else {},
                'Detailed Results': st.session_state.results,
                'Analysis Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.json(report)
            st.success("Detailed report generated! You can copy the JSON above for further analysis.")
    
    else:
        st.info(" Please run at least one model analysis to see the comparison.")
        st.markdown("""
        ### How to get started:
        1. **Upload your dataset** using the file uploader in the sidebar
        2. **Select an algorithm** or choose "All Models" for comprehensive analysis
        3. **Navigate through the tabs** to see individual model results
        4. **Return to this tab** to see the final comparison and insights
        
        ### Supported file format:
        - CSV files with customer churn data
        - Target column should be named 'Churn' or will use the last column
        """)

# Main application logic
def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Debug info (remove in production)
    # st.sidebar.write(f"Debug: show_analysis = {st.session_state.show_analysis}")
    
    # Main navigation logic
    if st.session_state.show_analysis:
        show_analysis_dashboard()
    else:
        show_start_page()

if __name__ == "__main__":
    main()