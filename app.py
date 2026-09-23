import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Loan Approval Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 2rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING (works on Streamlit Cloud AND Colab)
# ============================================================
@st.cache_data
def load_data():
    search_paths = ['.', '/content/data']
    files = []
    for path in search_paths:
        if os.path.isdir(path):
            files += glob.glob(os.path.join(path, '*.csv'))
            files += glob.glob(os.path.join(path, '*.xlsx'))
            files += glob.glob(os.path.join(path, '*.xls'))

    if not files:
        return None, None

    preferred = [f for f in files if 'loan' in os.path.basename(f).lower()]
    target_file = preferred[0] if preferred else files[0]

    try:
        if target_file.lower().endswith('.csv'):
            try:
                df = pd.read_csv(target_file, encoding='utf-8-sig', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(target_file, encoding='latin-1', low_memory=False)
        else:
            df = pd.read_excel(target_file, header=0)
    except Exception as e:
        st.error(f"Error reading {target_file}: {e}")
        return None, None

    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    df = df.dropna(how='all').dropna(axis=1, how='all')
    return df, os.path.basename(target_file)


def clean_numeric_series(series):
    """Strip currency symbols, commas, whitespace → float. Returns NaN for non-numeric."""
    s = series.astype(str)
    s = s.str.replace(r'[,\s]', '', regex=True)
    s = s.str.replace(r'[₱$]', '', regex=True)
    s = s.str.replace(r'^\-$', '', regex=True)
    s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'N/A': np.nan})
    return pd.to_numeric(s, errors='coerce')


def try_numeric(series):
    """Return numeric version if >=70% of values parse as numeric, else None."""
    cleaned = clean_numeric_series(series)
    if cleaned.notna().sum() >= 0.7 * max(1, len(series)):
        return cleaned
    return None


@st.cache_data
def preprocess_data(df, target_col, feature_cols):
    df = df.copy()
    df = df[df[target_col].notna()].copy()
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()

    mapping = {
        'yes': 'Qualified', 'qualified': 'Qualified', '1': 'Qualified',
        'true': 'Qualified', 'approved': 'Qualified',
        'no': 'Not Qualified', 'not qualified': 'Not Qualified',
        '0': 'Not Qualified', 'false': 'Not Qualified',
        'rejected': 'Not Qualified'
    }
    df[target_col + '_label'] = df[target_col].map(lambda x: mapping.get(x, 'Not Qualified'))
    y = (df[target_col + '_label'] == 'Qualified').astype(int)

    X = df[feature_cols].copy()
    for col in X.columns:
        num = try_numeric(X[col])
        if num is not None:
            X[col] = num.fillna(num.median() if num.notna().any() else 0)
        else:
            X[col] = X[col].astype(str).fillna('Unknown')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]
    return X, y, df


def detect_target_column(df):
    candidates = ['loanapproved', 'loan_approved', 'approved', 'loanapproval',
                  'loan_qualification_status', 'loanqualificationstatus',
                  'qualification', 'status', 'target', 'label', 'approved_status']
    cols_lower = {c.lower().replace(' ', '').replace('_', ''): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('_', '')
        if key in cols_lower:
            return cols_lower[key]
    return df.columns[-1]


def detect_feature_columns(df, target_col, max_features=15):
    features = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        if any(k in col_lower for k in ['date', 'id', 'name', 'index']):
            continue
        if df[col].dtype == 'object' and df[col].nunique(dropna=True) > 50:
            continue
        features.append(col)
    return features[:max_features]


# ============================================================
# MODEL TRAINING
# ============================================================
def train_models(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=random_state, solver='liblinear')
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    svm = SVC(kernel='rbf', probability=True, random_state=random_state)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]

    results = {
        'Logistic Regression': {
            'model': lr, 'y_pred': y_pred_lr, 'y_prob': y_prob_lr,
            'y_test': y_test, 'X_test': X_test_scaled
        },
        'Support Vector Machine (SVM)': {
            'model': svm, 'y_pred': y_pred_svm, 'y_prob': y_prob_svm,
            'y_test': y_test, 'X_test': X_test_scaled
        }
    }
    return results, scaler, X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_true, y_prob)
    }


# ============================================================
# LOAD & PREP
# ============================================================
df_raw, filename = load_data()

st.markdown('<div class="main-header">🏦 Loan Approval Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting Loan Approval Status Using Machine Learning Techniques<br>'
            '<i>Logistic Regression & Support Vector Machine (SVM)</i></div>', unsafe_allow_html=True)

with st.expander("🔍 Debug: Detected Data", expanded=False):
    if df_raw is None:
        st.error("❌ No dataset found. Please upload a CSV or Excel file to the repo root.")
    else:
        st.success(f"✅ Loaded file: `{filename}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Numeric Cols", df_raw.select_dtypes(include=np.number).shape[1])

        st.write("**All detected columns:**")
        st.code(list(df_raw.columns))

        st.write("**First 5 rows:**")
        st.dataframe(df_raw.head(), use_container_width=True)

        st.write("**Data types & missing values:**")
        info_df = pd.DataFrame({
            'dtype': df_raw.dtypes.astype(str),
            'missing': df_raw.isna().sum(),
            'missing_%': (df_raw.isna().sum() / len(df_raw) * 100).round(2),
            'unique': df_raw.nunique()
        })
        st.dataframe(info_df, use_container_width=True)

if df_raw is None:
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Model Configuration")

target_col = detect_target_column(df_raw)
st.sidebar.success(f"🎯 Target column: **{target_col}**")

all_cols = [c for c in df_raw.columns if c != target_col]
default_features = detect_feature_columns(df_raw, target_col)

feature_cols = st.sidebar.multiselect(
    "Select predictor variables:",
    options=all_cols,
    default=default_features
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

if len(feature_cols) < 2:
    st.warning("⚠️ Please select at least 2 predictor variables.")
    st.stop()

X, y, df_clean = preprocess_data(df_raw, target_col, feature_cols)

results, scaler, X_train, X_test, y_train, y_test = train_models(
    X, y, test_size=test_size, random_state=int(random_state)
)

metrics_table = pd.DataFrame({
    name: compute_metrics(r['y_test'], r['y_pred'], r['y_prob'])
    for name, r in results.items()
}).T

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 EDA",
    "🔥 Logistic Regression", "🌳 SVM",
    "📋 Cross-Validation", "💡 Recommendations"
])


# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
with tab1:
    st.header("📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Records</p><h2>{len(df_clean):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Predictors Used</p><h2>{len(feature_cols)}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Qualified</p><h2>{(y==1).sum():,}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Not Qualified</p><h2>{(y==0).sum():,}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🏆 Model Performance Summary")
    st.dataframe(metrics_table.style.format("{:.4f}").highlight_max(axis=0, color='#c6efce'),
                 use_container_width=True)

    st.subheader("📋 Dataset Sample")
    st.dataframe(df_clean.head(10), use_container_width=True)


# ------------------------------------------------------------
# TAB 2: EDA  (FIXED — only original numeric columns)
# ------------------------------------------------------------
with tab2:
    st.header("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_clean[target_col + '_label'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors[:len(counts)], startangle=90)
        ax.set_title("Qualified vs Not Qualified")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Class Counts")
        st.dataframe(
            counts.rename('Count').to_frame().assign(
                Percentage=lambda d: (d['Count'] / d['Count'].sum() * 100).round(2)
            ),
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Numeric Feature Distributions")

    numeric_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            if df_raw[c].dropna().nunique() > 5:
                numeric_cols.append(c)

    if not numeric_cols:
        for c in feature_cols:
            vals = clean_numeric_series(df_raw[c])
            if vals.notna().sum() >= 0.7 * len(df_raw) and vals.dropna().nunique() > 5:
                numeric_cols.append(c)

    numeric_cols = numeric_cols[:6]

    if numeric_cols:
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        axes = np.array(axes).flatten()
        plot_idx = 0
        for col in numeric_cols:
            vals = clean_numeric_series(df_raw[col]).dropna()
            if len(vals) == 0:
                continue
            axes[plot_idx].hist(vals.astype(float), bins=30,
                                color='#3498db', edgecolor='black')
            axes[plot_idx].set_title(col)
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1
        for j in range(plot_idx, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("ℹ️ No strictly numeric columns detected for histogram plotting. "
                "All selected predictors appear categorical — see the correlation heatmap below.")

    st.markdown("---")

    st.subheader("Correlation Heatmap (Encoded Features + Target)")
    try:
        corr_data = X.copy()
        corr_data[target_col] = y.values
        corr = corr_data.corr(numeric_only=True)
        if corr.shape[0] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                        center=0, ax=ax, cbar_kws={'shrink': 0.8})
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough features for correlation heatmap.")
    except Exception as e:
        st.warning(f"Could not generate correlation heatmap: {e}")

    st.markdown("---")
    st.subheader("Qualification Rate by Categorical Feature")
    cat_cols = [c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(df_raw[c])
                and df_raw[c].nunique() <= 15][:3]

    if cat_cols:
        for c in cat_cols:
            tmp = df_clean[[c, target_col + '_label']].copy()
            tmp[c] = tmp[c].astype(str)
            ct = pd.crosstab(tmp[c], tmp[target_col + '_label'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 4))
            ct.plot(kind='bar', stacked=True, ax=ax,
                    color=['#e74c3c', '#2ecc71'])
            ax.set_ylabel("% of borrowers")
            ax.set_title(f"Qualification Rate by {c}")
            ax.legend(loc='lower right')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No low-cardinality categorical features for breakdown.")


# ------------------------------------------------------------
# HELPER: model tab
# ------------------------------------------------------------
def render_model_tab(name, result, color):
    st.header(f"{name} — Model Performance")

    m = compute_metrics(result['y_test'], result['y_pred'], result['y_prob'])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['Accuracy']:.4f}")
    c2.metric("Precision", f"{m['Precision']:.4f}")
    c3.metric("Recall",    f"{m['Recall']:.4f}")
    c4.metric("F1-Score",  f"{m['F1-Score']:.4f}")
    c5.metric("ROC-AUC",   f"{m['ROC-AUC']:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap=color,
                    xticklabels=['Not Qualified', 'Qualified'],
                    yticklabels=['Not Qualified', 'Qualified'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color=color[0], lw=2, label=f"AUC = {m['ROC-AUC']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.subheader("Classification Report")
    rep = classification_report(result['y_test'], result['y_pred'],
                                target_names=['Not Qualified', 'Qualified'],
                                output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    if name == "Logistic Regression":
        st.subheader("Coefficient Analysis (Significant Factors)")
        coefs = pd.Series(result['model'].coef_[0], index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(10, max(4, len(coefs) * 0.4)))
        colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in coefs.values]
        coefs.plot(kind='barh', color=colors_bar, ax=ax)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel("Coefficient Value")
        st.pyplot(fig)
        plt.close()
        st.caption("Positive coefficients → increase likelihood of approval; "
                   "Negative → decrease likelihood.")


# ------------------------------------------------------------
# TAB 3 & 4
# ------------------------------------------------------------
with tab3:
    render_model_tab("Logistic Regression",
                     results['Logistic Regression'],
                     ['#2ecc71', '#27ae60'])

with tab4:
    render_model_tab("Support Vector Machine (SVM)",
                     results['Support Vector Machine (SVM)'],
                     ['#e74c3c', '#c0392b'])


# ------------------------------------------------------------
# TAB 5
# ------------------------------------------------------------
with tab5:
    st.header("📋 Stratified K-Fold Cross-Validation")

    k = st.slider("Number of folds", 3, 10, 5)

    X_scaled = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    lr_cv = cross_val_score(LogisticRegression(max_iter=1000, solver='liblinear'),
                            X_scaled, y, cv=cv, scoring='accuracy')
    svm_cv = cross_val_score(SVC(kernel='rbf', probability=True),
                             X_scaled, y, cv=cv, scoring='accuracy')

    cv_df = pd.DataFrame({
        'Logistic Regression': lr_cv,
        'Support Vector Machine (SVM)': svm_cv
    })
    cv_df.index = [f"Fold {i+1}" for i in range(k)]
    cv_df.loc['Mean'] = cv_df.mean()
    cv_df.loc['Std']  = cv_df.iloc[:-2].std()

    st.dataframe(cv_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    cv_df.iloc[:-2].plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{k}-Fold Cross-Validation Accuracy per Fold")
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(f"**Mean CV Accuracy — LR:** {lr_cv.mean():.4f} ± {lr_cv.std():.4f}  \n"
            f"**Mean CV Accuracy — SVM:** {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")


# ------------------------------------------------------------
# TAB 6
# ------------------------------------------------------------
with tab6:
    st.header("💡 Recommendations & Interpretation")

    best_model = metrics_table['Accuracy'].idxmax()
    best_acc = metrics_table['Accuracy'].max()

    st.success(f"### 🏆 Best Performing Model: **{best_model}** (Accuracy = {best_acc:.4f})")

    st.markdown("---")
    st.subheader("📌 Model Performance Comparison")
    st.dataframe(metrics_table.style.format("{:.4f}"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_table.plot(kind='bar', ax=ax,
                       color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Across All Metrics")
    ax.legend(loc='lower right', ncol=5)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("🔑 Key Significant Factors (from Logistic Regression)")

    lr_model = results['Logistic Regression']['model']
    coefs = pd.Series(lr_model.coef_[0], index=feature_cols).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Factors INCREASING Approval")
        for f, v in coefs[coefs > 0].items():
            st.markdown(f"- **{f}**: +{v:.4f}")
    with col2:
        st.markdown("#### ⚠️ Factors DECREASING Approval")
        for f, v in coefs[coefs < 0].items():
            st.markdown(f"- **{f}**: {v:.4f}")

    st.markdown("---")
    st.subheader("🎯 Recommendations")

    st.markdown(f"""
    Based on the analysis using **{len(feature_cols)}** borrower-related attributes
    on **{len(df_clean):,}** records:

    1. **For Financial Institutions:**
       - Deploy the **{best_model}** model for automated preliminary loan screening.
       - Prioritize checking the top positive factors before final approval.
       - Use the model to flag high-risk applications for manual review.

    2. **For Borrowers:**
       - Focus on improving the strongest positive predictors (see above).
       - Maintain a clean financial record (no disputes, good moral standing).
       - Ensure income stability and manage debt-to-income ratio responsibly.

    3. **For Future Researchers:**
       - Extend this study with **Random Forest, XGBoost, or Neural Networks**.
       - Apply **SMOTE** to handle class imbalance if present.
       - Include **SHAP values** for deeper feature-importance interpretation.
       - Use larger, more balanced datasets across multiple lending institutions.

    4. **Limitations:**
       - Results are limited to the dataset used (n = {len(df_clean):,}).
       - Model is for academic purposes only — not a replacement for actual
         credit committee decisions.
    """)

    st.markdown("---")
    st.subheader("📥 Download Results")
    csv = metrics_table.to_csv().encode('utf-8')
    st.download_button("Download Metrics CSV", csv, "model_metrics.csv", "text/csv")

    preds = pd.DataFrame({
        'Actual': y_test.map({0: 'Not Qualified', 1: 'Qualified'}),
        'LR_Predicted': pd.Series(results['Logistic Regression']['y_pred']).map({0: 'Not Qualified', 1: 'Qualified'}),
        'SVM_Predicted': pd.Series(results['Support Vector Machine (SVM)']['y_pred']).map({0: 'Not Qualified', 1: 'Qualified'})
    })
    st.download_button("Download Predictions CSV",
                       preds.to_csv(index=False).encode('utf-8'),
                       "predictions.csv", "text/csv")


st.markdown("---")
st.caption("🎓 Capstone Dashboard • Predicting Loan Approval Status Using Machine Learning Techniques • "
           "Gorne & Tangliben • Agusan del Sur State University • 2026")