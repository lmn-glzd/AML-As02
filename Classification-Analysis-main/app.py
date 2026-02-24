import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Risk — ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────
# THEME / CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0b0f1a; }
  .banner {
    background: linear-gradient(120deg, #0f3460 0%, #16213e 60%, #1a1a2e 100%);
    border-radius: 14px; padding: 2rem 2.5rem; margin-bottom: 1.8rem;
  }
  .banner h1 { color: #e94560; font-size: 2rem; margin: 0 0 .3rem; }
  .banner p  { color: #a8b2d8; margin: 0; font-size: 0.95rem; }
  .card {
    background: linear-gradient(135deg, #16213e, #0f3460);
    border: 1px solid #1e4080; border-radius: 10px;
    padding: 1.1rem 1rem; text-align: center; height: 100%;
  }
  .card .val  { font-size: 1.9rem; font-weight: 700; color: #64ffda; line-height: 1.2; }
  .card .lbl  { font-size: 0.78rem; color: #8899bb; margin-top: .3rem; }
  .sec {
    background: linear-gradient(90deg,#0f3460,#1a1a2e);
    border-left: 4px solid #e94560; border-radius: 7px;
    padding: .6rem 1rem; margin: 1.2rem 0 .7rem;
  }
  .sec h4 { color:#e94560; margin:0; font-size:1rem; }
  .insight {
    background: #111827; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: .9rem 1.1rem; margin: .7rem 0;
    color: #a8b2d8; font-size: .88rem; line-height: 1.6;
  }
  .insight b { color: #64ffda; }
  .stTabs [data-baseweb="tab"] {
    background: #16213e; border-radius: 6px 6px 0 0;
    padding: 7px 14px; color: #8899bb; font-size:.85rem;
  }
  .stTabs [aria-selected="true"] {
    background: #0f3460 !important; color: #64ffda !important;
  }
  div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
DARK  = "#0b0f1a"
DARK2 = "#111827"
ACCENT= "#64ffda"
RED   = "#e94560"
AMBER = "#f59e0b"
PURPLE= "#a78bfa"
RISK_COLORS = {0: ACCENT, 1: AMBER, 2: RED}

def dark_fig(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK2)
    ax.set_facecolor(DARK2)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.tick_params(colors="white")
    return fig, ax

def card(val, lbl):
    return f'<div class="card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'

def sec(title):
    return f'<div class="sec"><h4>{title}</h4></div>'

def insight(html):
    return f'<div class="insight">{html}</div>'

# ──────────────────────────────────────────────────────────
# AUTO-DETECT CSV IN SAME DIRECTORY AS app.py
# ──────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def find_csv():
    """Return path to the first CSV found next to app.py, preferring known names."""
    preferred = [
        "hr_employee_attrition_data.csv",
        "hr_attrition.csv",
        "employee_attrition.csv",
    ]
    for name in preferred:
        path = os.path.join(APP_DIR, name)
        if os.path.exists(path):
            return path
    # Fall back to any CSV in the same directory
    csvs = glob.glob(os.path.join(APP_DIR, "*.csv"))
    return csvs[0] if csvs else None

CSV_PATH = find_csv()

# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    if CSV_PATH:
        st.success(f"✅ Dataset loaded:\n`{os.path.basename(CSV_PATH)}`")
    else:
        st.error("❌ No CSV found next to app.py")
    st.markdown("---")
    test_size    = st.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random State", 0, 200, 42, 1)
    st.markdown("---")
    st.markdown("**Target:** `Attrition_Risk_Level`  \n0 = Low · 1 = Medium · 2 = High")

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>📊 Employee Attrition Risk — ML Dashboard</h1>
  <p>Binary LR &nbsp;·&nbsp; Multinomial LR &nbsp;·&nbsp; LDA &nbsp;·&nbsp; QDA &nbsp;·&nbsp; Naive Bayes &nbsp;·&nbsp; Linear vs Poisson Regression</p>
</div>
""", unsafe_allow_html=True)

if CSV_PATH is None:
    st.error("No CSV file found in the same directory as `app.py`. "
             "Place `hr_employee_attrition_data.csv` next to `app.py` and restart.")
    st.stop()

# ──────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────
@st.cache_data
def load(path, ts, rs):
    df = pd.read_csv(path)
    df = df.drop(columns=["Employee_ID"], errors="ignore")

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    if "Job_Role" in df.columns:
        dummies = pd.get_dummies(df["Job_Role"], prefix="JobRole", drop_first=True)
        df = pd.concat([df.drop("Job_Role", axis=1), dummies], axis=1)

    df = df.drop_duplicates()

    if df["Attrition_Risk_Level"].dtype == object:
        df["Attrition_Risk_Level"] = df["Attrition_Risk_Level"].map(
            {"Low": 0, "Medium": 1, "High": 2})

    X = df.drop(columns=["Attrition_Risk_Level"])
    y = df["Attrition_Risk_Level"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=rs, stratify=y)

    scale_cols = [c for c in ["Age","Monthly_Income","Distance_From_Home_KM",
                               "Avg_Monthly_Hours","Years_at_Company",
                               "Years_Since_Last_Promotion"] if c in X.columns]
    sc = StandardScaler()
    X_train[scale_cols] = sc.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = sc.transform(X_test[scale_cols])

    return df, X_train, X_test, y_train.astype(int), y_test.astype(int)

# Re-run load when slider/number_input changes (cache key includes ts and rs)
df, X_train, X_test, y_train, y_test = load(CSV_PATH, test_size, random_state)

# Shared constants
LR_FEATURES = [c for c in ["Age","Gender","Monthly_Income","Job_Satisfaction",
                             "Work_Life_Balance","Distance_From_Home_KM",
                             "Num_Projects","Avg_Monthly_Hours","Years_at_Company"]
                if c in X_train.columns]

NUM_COLS = [c for c in ["Age","Monthly_Income","Distance_From_Home_KM",
                         "Avg_Monthly_Hours","Years_at_Company","Num_Projects"]
            if c in df.columns]

CORR_COLS = [c for c in ["Age","Gender","Monthly_Income","Distance_From_Home_KM",
                           "Num_Projects","Avg_Monthly_Hours","Years_at_Company",
                           "Years_Since_Last_Promotion"] if c in df.columns]

# ──────────────────────────────────────────────────────────
# TRAIN ALL MODELS (cached)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def fit_models(_X_train, _y_train):
    y_bin  = (_y_train == 2).astype(int)
    X_bin  = sm.add_constant(_X_train[LR_FEATURES])
    res_bin = sm.Logit(y_bin, X_bin).fit(disp=False)

    X_mc1  = sm.add_constant(_X_train[LR_FEATURES])
    res1   = sm.MNLogit(_y_train, X_mc1).fit(disp=False, maxiter=300)

    feat2  = [f for f in LR_FEATURES if f != "Years_at_Company"]
    X_mc2  = sm.add_constant(_X_train[feat2])
    res2   = sm.MNLogit(_y_train, X_mc2).fit(disp=False, maxiter=300)

    lda = LinearDiscriminantAnalysis();    lda.fit(_X_train, _y_train)
    qda = QuadraticDiscriminantAnalysis(); qda.fit(_X_train, _y_train)
    nb  = GaussianNB();                    nb.fit(_X_train, _y_train)

    return res_bin, res1, res2, feat2, lda, qda, nb

with st.spinner("Training all models…"):
    res_bin, res1, res2, feat2, lda, qda, nb = fit_models(X_train, y_train)

# ──────────────────────────────────────────────────────────
# PREDICTIONS
# ──────────────────────────────────────────────────────────
y_test_binary = (y_test == 2).astype(int)

X_test_bin  = sm.add_constant(X_test[LR_FEATURES]).reindex(
    columns=sm.add_constant(X_train[LR_FEATURES]).columns, fill_value=0)
y_prob_bin  = res_bin.predict(X_test_bin)
y_pred_bin  = (y_prob_bin >= 0.5).astype(int)
acc_bin     = accuracy_score(y_test_binary, y_pred_bin)

X_test_mc1  = sm.add_constant(X_test[LR_FEATURES]).reindex(
    columns=res1.model.exog_names, fill_value=0)
y_prob_mc1  = res1.predict(X_test_mc1)
y_pred_mc1  = np.argmax(y_prob_mc1.values, axis=1)
acc_mc1     = accuracy_score(y_test, y_pred_mc1)

X_test_mc2  = sm.add_constant(X_test[feat2]).reindex(
    columns=res2.model.exog_names, fill_value=0)
y_prob_mc2  = res2.predict(X_test_mc2)
y_pred_mc2  = np.argmax(y_prob_mc2.values, axis=1)
acc_mc2     = accuracy_score(y_test, y_pred_mc2)

y_pred_lda  = lda.predict(X_test)
y_prob_lda  = lda.predict_proba(X_test)
acc_lda     = accuracy_score(y_test, y_pred_lda)
cm_lda      = confusion_matrix(y_test, y_pred_lda)

y_pred_qda  = qda.predict(X_test)
acc_qda     = accuracy_score(y_test, y_pred_qda)
cm_qda      = confusion_matrix(y_test, y_pred_qda)

y_pred_nb   = nb.predict(X_test)
acc_nb      = accuracy_score(y_test, y_pred_nb)
cm_nb       = confusion_matrix(y_test, y_pred_nb)
report_nb   = classification_report(y_test, y_pred_nb,
                  target_names=["Low","Medium","High"], output_dict=True)

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
T = st.tabs([
    "📂 Dataset Overview",
    "🔍 EDA & Preprocessing",
    "🔵 Binary Logistic Regression",
    "📈 Multiclass Logistic Regression",
    "🧮 LDA",
    "🔮 QDA",
    "🌿 Naive Bayes",
    "⚖️ Model Comparison",
    "📉 Linear vs Poisson",
])

# ══════════════════════════════════════════════════════════
# TAB 0 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════
with T[0]:
    st.markdown("### 📂 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card(f"{len(df):,}", "Total Records"),              unsafe_allow_html=True)
    c2.markdown(card(df.shape[1],    "Features"),                   unsafe_allow_html=True)
    c3.markdown(card(df.isnull().sum().sum(), "Missing Values"),    unsafe_allow_html=True)
    c4.markdown(card(3,              "Risk Classes"),               unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns([2, 1])

    with ca:
        st.markdown("#### Sample Data (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

    with cb:
        st.markdown("#### Class Distribution")
        vc  = df["Attrition_Risk_Level"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor=DARK2)
        ax.set_facecolor(DARK2)
        ax.pie(vc.values,
               labels=["Low Risk","Medium Risk","High Risk"],
               colors=[ACCENT, AMBER, RED],
               autopct="%1.1f%%", startangle=90,
               textprops={"color":"white","fontsize":9})
        ax.set_title("Attrition Risk Levels", color="white", pad=10)
        st.pyplot(fig); plt.close()

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 1 — EDA & PREPROCESSING
# ══════════════════════════════════════════════════════════
with T[1]:
    st.markdown("### 🔍 Exploratory Data Analysis & Preprocessing")

    st.markdown(sec("📦 IQR Outlier Detection"), unsafe_allow_html=True)
    rows = []
    for col in NUM_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        n = ((df[col] < lb) | (df[col] > ub)).sum()
        rows.append({"Variable": col, "Lower Bound": round(lb,2),
                     "Upper Bound": round(ub,2), "Outliers": n,
                     "% Outliers": round(n/len(df)*100, 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown(sec("↗️ Feature Skewness"), unsafe_allow_html=True)
    skew = df[NUM_COLS].skew().reset_index()
    skew.columns = ["Variable","Skewness"]
    skew["Status"] = skew["Skewness"].apply(
        lambda x: "⚠️ High Skew" if abs(x) > 1.5 else "✅ Normal")
    st.dataframe(skew.round(4), use_container_width=True)

    st.markdown(sec("🔗 Correlation Matrix"), unsafe_allow_html=True)
    corr = df[CORR_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=DARK2)
    ax.set_facecolor(DARK2)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=.5, annot_kws={"size":8}, cbar_kws={"shrink":.8})
    ax.set_title("Correlation Matrix", color="white", pad=12)
    plt.xticks(color="white", fontsize=8, rotation=30, ha="right")
    plt.yticks(color="white", fontsize=8)
    st.pyplot(fig); plt.close()

    st.markdown(insight(
        "📌 <b>Key finding:</b> Age & Years_at_Company (r ≈ 0.63) and Age & Monthly_Income "
        "(r ≈ 0.61) show moderate correlation, but no pair exceeds 0.8 — no severe multicollinearity."
    ), unsafe_allow_html=True)

    st.markdown(sec("⚖️ Train/Test Class Balance"), unsafe_allow_html=True)
    bal = pd.DataFrame({
        "Train %": y_train.value_counts(normalize=True).sort_index().mul(100).round(2),
        "Test %":  y_test.value_counts(normalize=True).sort_index().mul(100).round(2),
    })
    bal.index = ["Low Risk","Medium Risk","High Risk"]
    st.dataframe(bal, use_container_width=True)

    st.markdown(sec("📊 Feature Distribution by Risk Level"), unsafe_allow_html=True)
    sel = st.selectbox("Choose a feature:", NUM_COLS)
    fig, ax = dark_fig(9, 4)
    for lvl in sorted(df["Attrition_Risk_Level"].dropna().unique()):
        data = df[df["Attrition_Risk_Level"] == lvl][sel].dropna()
        ax.hist(data, bins=35, alpha=0.55, color=RISK_COLORS.get(int(lvl),"gray"),
                label=["Low","Medium","High"][int(lvl)], edgecolor="none")
    ax.set_xlabel(sel, color="white"); ax.set_ylabel("Count", color="white")
    ax.set_title(f"{sel} — by Risk Level", color="white")
    ax.legend(facecolor=DARK2, labelcolor="white")
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# TAB 2 — BINARY LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════
with T[2]:
    st.markdown("### 🔵 Binary Logistic Regression — High Risk vs Others")
    st.markdown(insight(
        "📌 The binary target collapses the three risk classes: <b>1 = High Risk (class 2), "
        "0 = Low or Medium Risk</b>. A standard Logit model is fitted to identify which "
        "features best separate High Risk employees from the rest."
    ), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(card(f"{res_bin.llf:.1f}",       "Log-Likelihood"), unsafe_allow_html=True)
    c2.markdown(card(f"{res_bin.prsquared:.4f}",  "Pseudo R²"),     unsafe_allow_html=True)
    c3.markdown(card(f"{acc_bin*100:.2f}%",        "Test Accuracy"), unsafe_allow_html=True)

    st.markdown(sec("📋 Coefficients — Std Error, Z-stat, P-value"), unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Coefficient":    res_bin.params.round(5),
        "Std Error":      res_bin.bse.round(5),
        "Z-Statistic":    res_bin.tvalues.round(4),
        "P-Value":        res_bin.pvalues.round(6),
        "Odds Ratio":     np.exp(res_bin.params).round(4),
        "Sig (p < 0.05)": res_bin.pvalues < 0.05,
    })
    st.dataframe(coef_df, use_container_width=True)

    st.markdown(sec("🎯 Odds Ratios"), unsafe_allow_html=True)
    or_df = coef_df.drop("const", errors="ignore").sort_values("Odds Ratio")
    fig, ax = dark_fig(9, 4.5)
    ax.barh(or_df.index, or_df["Odds Ratio"],
            color=[RED if v >= 1 else ACCENT for v in or_df["Odds Ratio"]],
            edgecolor="none")
    ax.axvline(1, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Odds Ratio (exp(β))", color="white")
    ax.set_title("Odds Ratios — Binary LR (High Risk vs Others)", color="white")
    st.pyplot(fig); plt.close()

    st.markdown(sec("📉 ROC Curve — Binary LR"), unsafe_allow_html=True)
    fpr_b, tpr_b, _ = roc_curve(y_test_binary, y_prob_bin)
    auc_b = auc(fpr_b, tpr_b)
    fig, ax = dark_fig(7, 4.5)
    ax.plot(fpr_b, tpr_b, color=ACCENT, linewidth=2, label=f"AUC = {auc_b:.3f}")
    ax.plot([0,1],[0,1],"--", color="#555", linewidth=1)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate",  color="white")
    ax.set_title("ROC Curve — Binary Logistic Regression", color="white")
    ax.legend(facecolor=DARK2, labelcolor="white")
    st.pyplot(fig); plt.close()

    st.markdown(sec("🔢 Confusion Matrix"), unsafe_allow_html=True)
    cm_b = confusion_matrix(y_test_binary, y_pred_bin)
    fig, ax = dark_fig(5, 4)
    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Others","High Risk"],
                yticklabels=["Others","High Risk"],
                annot_kws={"size":13})
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual",    color="white")
    ax.set_title("Binary LR Confusion Matrix", color="white")
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# TAB 3 — MULTICLASS LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════
with T[3]:
    st.markdown("### 📈 Multiclass Logistic Regression")

    sub = st.tabs(["Model 1 — With Confounding Var",
                   "Model 2 — Without Confounding Var",
                   "🔀 Confounding Analysis"])

    with sub[0]:
        st.markdown("#### Model 1: All Predictors (incl. `Years_at_Company`)")
        c1, c2, c3 = st.columns(3)
        c1.markdown(card(f"{res1.llf:.1f}",      "Log-Likelihood"), unsafe_allow_html=True)
        c2.markdown(card(f"{res1.prsquared:.4f}", "Pseudo R²"),     unsafe_allow_html=True)
        c3.markdown(card(f"{acc_mc1*100:.2f}%",   "Test Accuracy"), unsafe_allow_html=True)

        st.markdown(sec("📋 Coefficients — Class 2 (High Risk vs Low)"), unsafe_allow_html=True)
        col_idx = 1 if res1.params.shape[1] > 1 else 0
        coef1 = pd.DataFrame({
            "Coeff":      res1.params.iloc[:, col_idx].round(5),
            "Std Error":  res1.bse.iloc[:, col_idx].round(5),
            "Z-Stat":     res1.tvalues.iloc[:, col_idx].round(4),
            "P-Value":    res1.pvalues.iloc[:, col_idx].round(6),
            "Odds Ratio": np.exp(res1.params.iloc[:, col_idx]).round(4),
            "Sig":        res1.pvalues.iloc[:, col_idx] < 0.05,
        })
        st.dataframe(coef1, use_container_width=True)

        or1 = coef1.drop("const", errors="ignore").sort_values("Odds Ratio")
        fig, ax = dark_fig(9, 4.5)
        ax.barh(or1.index, or1["Odds Ratio"],
                color=[RED if v >= 1 else ACCENT for v in or1["Odds Ratio"]],
                edgecolor="none")
        ax.axvline(1, color="white", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Odds Ratio", color="white")
        ax.set_title("Odds Ratios — Model 1 (High Risk vs Low)", color="white")
        st.pyplot(fig); plt.close()

    with sub[1]:
        st.markdown("#### Model 2: Without `Years_at_Company` (confounding removed)")
        c1, c2, c3 = st.columns(3)
        c1.markdown(card(f"{res2.llf:.1f}",      "Log-Likelihood"), unsafe_allow_html=True)
        c2.markdown(card(f"{res2.prsquared:.4f}", "Pseudo R²"),     unsafe_allow_html=True)
        c3.markdown(card(f"{acc_mc2*100:.2f}%",   "Test Accuracy"), unsafe_allow_html=True)

        st.markdown(sec("📋 Coefficients — Class 2 (High Risk vs Low)"), unsafe_allow_html=True)
        col_idx = 1 if res2.params.shape[1] > 1 else 0
        coef2 = pd.DataFrame({
            "Coeff":      res2.params.iloc[:, col_idx].round(5),
            "Std Error":  res2.bse.iloc[:, col_idx].round(5),
            "Z-Stat":     res2.tvalues.iloc[:, col_idx].round(4),
            "P-Value":    res2.pvalues.iloc[:, col_idx].round(6),
            "Odds Ratio": np.exp(res2.params.iloc[:, col_idx]).round(4),
            "Sig":        res2.pvalues.iloc[:, col_idx] < 0.05,
        })
        st.dataframe(coef2, use_container_width=True)

        or2 = coef2.drop("const", errors="ignore").sort_values("Odds Ratio")
        fig, ax = dark_fig(9, 4.5)
        ax.barh(or2.index, or2["Odds Ratio"],
                color=[RED if v >= 1 else ACCENT for v in or2["Odds Ratio"]],
                edgecolor="none")
        ax.axvline(1, color="white", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Odds Ratio", color="white")
        ax.set_title("Odds Ratios — Model 2 (No Years_at_Company)", color="white")
        st.pyplot(fig); plt.close()

    with sub[2]:
        st.markdown("#### Confounding Variable Analysis")
        st.markdown(insight(
            "📌 <b>Hypothesis:</b> <code>Years_at_Company</code> may confound the "
            "relationship between <code>Age</code> and attrition risk "
            "(correlation r ≈ 0.63). We compare Model 1 (with) vs Model 2 (without) "
            "and check how Age's coefficient changes."
        ), unsafe_allow_html=True)

        st.markdown(sec("📊 VIF Analysis — Model 1 Features"), unsafe_allow_html=True)
        X_vif = X_train[LR_FEATURES].copy()
        vif_vals = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        vif_df = pd.DataFrame({"Variable": LR_FEATURES, "VIF": np.round(vif_vals, 4)})
        vif_df["Status"] = vif_df["VIF"].apply(
            lambda v: "✅ OK" if v < 5 else ("⚠️ Moderate" if v < 10 else "🚨 High"))
        st.dataframe(vif_df, use_container_width=True)

        st.markdown(sec("⚖️ Model 1 vs Model 2 — Key Metrics"), unsafe_allow_html=True)
        cmp = pd.DataFrame({
            "Metric": ["Log-Likelihood","Pseudo R²","Test Accuracy"],
            "Model 1 (with Yrs_at_Company)": [
                round(res1.llf,2), round(res1.prsquared,4), round(acc_mc1,4)],
            "Model 2 (without Yrs_at_Company)": [
                round(res2.llf,2), round(res2.prsquared,4), round(acc_mc2,4)],
        })
        st.dataframe(cmp, use_container_width=True)

        age_coef1 = res1.params.loc["Age"] if "Age" in res1.params.index else res1.params.iloc[0]
        age_coef2 = res2.params.loc["Age"] if "Age" in res2.params.index else res2.params.iloc[0]
        if hasattr(age_coef1, 'iloc'):
            age_c1 = float(age_coef1.iloc[1]) if len(age_coef1) > 1 else float(age_coef1.iloc[0])
            age_c2 = float(age_coef2.iloc[1]) if len(age_coef2) > 1 else float(age_coef2.iloc[0])
        else:
            age_c1, age_c2 = float(age_coef1), float(age_coef2)

        fig, ax = dark_fig(6, 4)
        ax.bar(["With Years_at_Company\n(Model 1)","Without Years_at_Company\n(Model 2)"],
               [abs(age_c1), abs(age_c2)],
               color=[AMBER, RED], edgecolor="none", width=0.5)
        ax.set_ylabel("|Age Coefficient|", color="white")
        ax.set_title("Age Coefficient — Before vs After Removing Confounder", color="white")
        for i, v in enumerate([abs(age_c1), abs(age_c2)]):
            ax.text(i, v+0.01, f"{v:.4f}", ha="center", color="white", fontsize=10)
        st.pyplot(fig); plt.close()

        st.markdown(insight(
            "📌 <b>Conclusion:</b> When <code>Years_at_Company</code> is removed, "
            "Age's coefficient magnitude increases notably and becomes more significant. "
            "Pseudo R² drops, confirming Years_at_Company carries independent predictive "
            "power and acts as a <b>confounding variable</b> in the Age → Attrition relationship."
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — LDA
# ══════════════════════════════════════════════════════════
with T[4]:
    st.markdown("### 🧮 Linear Discriminant Analysis (LDA)")

    fpr_lda, tpr_lda, roc_thresh = roc_curve(y_test_binary, y_prob_lda[:, 2])
    auc_lda  = auc(fpr_lda, tpr_lda)
    opt_idx  = int(np.argmax(tpr_lda - fpr_lda))

    c1, c2, c3 = st.columns(3)
    c1.markdown(card(f"{acc_lda*100:.2f}%",         "Test Accuracy"),       unsafe_allow_html=True)
    c2.markdown(card(f"{auc_lda:.3f}",              "AUC (High Risk)"),     unsafe_allow_html=True)
    c3.markdown(card(f"{roc_thresh[opt_idx]:.2f}",  "Optimal ROC Threshold"), unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown(sec("🔢 Confusion Matrix"), unsafe_allow_html=True)
        fig, ax = dark_fig(5.5, 4.5)
        sns.heatmap(cm_lda, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low","Medium","High"],
                    yticklabels=["Low","Medium","High"],
                    annot_kws={"size":13})
        ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
        ax.set_title("LDA Confusion Matrix", color="white")
        st.pyplot(fig); plt.close()

    with cb:
        st.markdown(sec("📉 ROC Curve (High Attrition Risk)"), unsafe_allow_html=True)
        fig, ax = dark_fig(5.5, 4.5)
        ax.plot(fpr_lda, tpr_lda, color=ACCENT, linewidth=2, label=f"AUC = {auc_lda:.3f}")
        ax.plot([0,1],[0,1],"--", color="#555", linewidth=1)
        ax.scatter(fpr_lda[opt_idx], tpr_lda[opt_idx], color=RED, zorder=5, s=80,
                   label=f"Optimal ≈ {roc_thresh[opt_idx]:.2f}")
        ax.set_xlabel("False Positive Rate", color="white")
        ax.set_ylabel("True Positive Rate",  color="white")
        ax.set_title("ROC Curve — LDA (High Attrition)", color="white")
        ax.legend(facecolor=DARK2, labelcolor="white", fontsize=9)
        st.pyplot(fig); plt.close()

    st.markdown(sec("🎚️ Threshold Optimization (F1-Score)"), unsafe_allow_html=True)
    y_prob_high_lda = y_prob_lda[:, 2]
    t_rows = []
    for t in np.arange(0.10, 0.90, 0.05):
        yp = (y_prob_high_lda >= t).astype(int)
        t_rows.append({
            "Threshold": round(t, 2),
            "Precision": round(precision_score(y_test_binary, yp, zero_division=0), 4),
            "Recall":    round(recall_score(y_test_binary, yp, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test_binary, yp, zero_division=0), 4),
        })
    thresh_df = pd.DataFrame(t_rows)
    best      = thresh_df.loc[thresh_df["F1-Score"].idxmax()]

    cc, cd = st.columns([1, 2])
    with cc:
        st.dataframe(thresh_df, use_container_width=True, height=360)
        st.markdown(insight(
            f"🏆 <b>Best threshold:</b> {best['Threshold']}<br>"
            f"F1 = {best['F1-Score']} | Precision = {best['Precision']} | Recall = {best['Recall']}"
        ), unsafe_allow_html=True)
    with cd:
        fig, ax = dark_fig(8, 4.5)
        ax.plot(thresh_df["Threshold"], thresh_df["Precision"], color=ACCENT, marker="o", ms=4, label="Precision")
        ax.plot(thresh_df["Threshold"], thresh_df["Recall"],    color=AMBER,  marker="o", ms=4, label="Recall")
        ax.plot(thresh_df["Threshold"], thresh_df["F1-Score"],  color=RED,    marker="o", ms=4, linewidth=2, label="F1-Score")
        ax.axvline(best["Threshold"], color="white", linestyle="--", linewidth=1, alpha=0.6,
                   label=f'Best = {best["Threshold"]}')
        ax.set_xlabel("Threshold", color="white"); ax.set_ylabel("Score", color="white")
        ax.set_title("Precision / Recall / F1 vs Threshold", color="white")
        ax.legend(facecolor=DARK2, labelcolor="white", fontsize=9)
        st.pyplot(fig); plt.close()

    st.markdown(sec("📋 Classification Report"), unsafe_allow_html=True)
    rep_lda = pd.DataFrame(classification_report(
        y_test, y_pred_lda, target_names=["Low","Medium","High"], output_dict=True)).T
    st.dataframe(rep_lda.round(4), use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 5 — QDA
# ══════════════════════════════════════════════════════════
with T[5]:
    st.markdown("### 🔮 Quadratic Discriminant Analysis (QDA)")
    rep_qda = classification_report(y_test, y_pred_qda,
                  target_names=["Low","Medium","High"], output_dict=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(card(f"{acc_qda*100:.2f}%",                      "Test Accuracy"), unsafe_allow_html=True)
    c2.markdown(card(f"{rep_qda['macro avg']['f1-score']:.3f}",  "Macro F1"),      unsafe_allow_html=True)
    diff = (acc_qda - acc_lda) * 100
    c3.markdown(card(f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%", "Δ vs LDA"), unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown(sec("🔢 Confusion Matrix"), unsafe_allow_html=True)
        fig, ax = dark_fig(5.5, 4.5)
        sns.heatmap(cm_qda, annot=True, fmt="d", cmap="Purples", ax=ax,
                    xticklabels=["Low","Medium","High"],
                    yticklabels=["Low","Medium","High"],
                    annot_kws={"size":13})
        ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
        ax.set_title("QDA Confusion Matrix", color="white")
        st.pyplot(fig); plt.close()

    with cb:
        st.markdown(sec("📋 Classification Report"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rep_qda).T.round(4), use_container_width=True)

    st.markdown(insight(
        "📌 <b>QDA Insight:</b> By allowing each class its own covariance matrix, QDA captures "
        "non-linear decision boundaries — explaining why it surpasses LDA. The attrition risk "
        "classes have distinct variance structures, making QDA the best-performing model overall."
    ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 6 — NAIVE BAYES
# ══════════════════════════════════════════════════════════
with T[6]:
    st.markdown("### 🌿 Gaussian Naïve Bayes")

    c1, c2, c3 = st.columns(3)
    c1.markdown(card(f"{acc_nb*100:.2f}%",                         "Test Accuracy"),    unsafe_allow_html=True)
    c2.markdown(card(f"{report_nb['macro avg']['f1-score']:.3f}",  "Macro F1"),         unsafe_allow_html=True)
    c3.markdown(card(f"{report_nb['macro avg']['precision']:.3f}", "Macro Precision"),  unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown(sec("🔢 Confusion Matrix"), unsafe_allow_html=True)
        fig, ax = dark_fig(5.5, 4.5)
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Low","Medium","High"],
                    yticklabels=["Low","Medium","High"],
                    annot_kws={"size":13})
        ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
        ax.set_title("Naive Bayes Confusion Matrix", color="white")
        st.pyplot(fig); plt.close()

    with cb:
        st.markdown(sec("📋 Classification Report"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(report_nb).T.round(4), use_container_width=True)

    st.markdown(insight(
        "📌 <b>Naive Bayes Insight:</b> The independence assumption is violated here — "
        "Age, Monthly_Income, and Years_at_Company are correlated. This reduces NB's "
        "effectiveness compared to LDA and QDA, despite its speed advantage."
    ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 7 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════
with T[7]:
    st.markdown("### ⚖️ Full Model Comparison")

    rows = []
    configs = [
        ("Binary LR (High vs Others)",       y_pred_bin,  y_test_binary),
        ("MNLogit — Model 1 (with Confound)", y_pred_mc1,  y_test),
        ("MNLogit — Model 2 (no Confound)",   y_pred_mc2,  y_test),
        ("LDA",                               y_pred_lda,  y_test),
        ("QDA",                               y_pred_qda,  y_test),
        ("Naive Bayes",                       y_pred_nb,   y_test),
    ]
    for name, yp, yt in configs:
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(yt, yp), 4),
            "Precision": round(precision_score(yt, yp, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(yt, yp, average="weighted", zero_division=0), 4),
            "F1-Score":  round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
        })
    cmp_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    st.dataframe(cmp_df, use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK2)
    bar_colors = [PURPLE, AMBER, AMBER, ACCENT, RED, "#64b5f6"]
    x = np.arange(len(cmp_df))

    for ax in axes:
        ax.set_facecolor(DARK2)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.tick_params(colors="white")

    axes[0].bar(x, cmp_df["Accuracy"], color=bar_colors[:len(x)], width=0.55, edgecolor="none")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cmp_df["Model"], rotation=25, ha="right", fontsize=8, color="white")
    axes[0].set_ylabel("Accuracy", color="white"); axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Test Accuracy", color="white")
    for i, v in enumerate(cmp_df["Accuracy"]):
        axes[0].text(i, v+.01, f"{v*100:.1f}%", ha="center", color="white", fontsize=8)

    axes[1].bar(x, cmp_df["F1-Score"], color=bar_colors[:len(x)], width=0.55, edgecolor="none")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cmp_df["Model"], rotation=25, ha="right", fontsize=8, color="white")
    axes[1].set_ylabel("Weighted F1-Score", color="white"); axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Weighted F1-Score", color="white")
    for i, v in enumerate(cmp_df["F1-Score"]):
        axes[1].text(i, v+.01, f"{v:.3f}", ha="center", color="white", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(insight(
        "📌 <b>Summary:</b><br>"
        "• <b>QDA</b> — Highest accuracy: class-specific covariance captures non-linear structure.<br>"
        "• <b>LDA</b> — Close second with near-perfect AUC (0.997) for High Risk detection.<br>"
        "• <b>Naive Bayes</b> — Outperforms Logistic Regression but hurt by correlated predictors.<br>"
        "• <b>Logistic Regression (Model 1)</b> — Best interpretability; Pseudo R² = 0.37.<br>"
        "• Removing the confounding variable (Years_at_Company) slightly reduces MNLogit accuracy."
    ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 8 — LINEAR vs POISSON REGRESSION
# ══════════════════════════════════════════════════════════
with T[8]:
    st.markdown("### 📉 Linear vs Poisson Regression — Predicting `Num_Projects`")

    if "Num_Projects" not in df.columns:
        st.warning("Column `Num_Projects` not found.")
    else:
        reg_feats = [c for c in ["Age","Monthly_Income","Years_at_Company",
                                  "Job_Satisfaction","Work_Life_Balance"] if c in df.columns]
        y_reg = df["Num_Projects"]
        X_reg = sm.add_constant(df[reg_feats])

        with st.spinner("Fitting OLS and Poisson models…"):
            lin_m = sm.OLS(y_reg, X_reg).fit()
            poi_m = sm.GLM(y_reg, X_reg, family=sm.families.Poisson()).fit()

        y_pred_lin = lin_m.predict(X_reg)
        y_pred_poi = poi_m.predict(X_reg)
        mse_lin = float(np.mean((y_reg - y_pred_lin)**2))
        mse_poi = float(np.mean((y_reg - y_pred_poi)**2))

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(card(f"{lin_m.rsquared:.4f}",  "OLS R²"),      unsafe_allow_html=True)
        c2.markdown(card(f"{mse_lin:.4f}",         "OLS MSE"),     unsafe_allow_html=True)
        c3.markdown(card(f"{round(lin_m.aic):,}",  "OLS AIC"),     unsafe_allow_html=True)
        c4.markdown(card(f"{round(poi_m.aic):,}",  "Poisson AIC"), unsafe_allow_html=True)

        st.markdown(sec("📋 OLS Regression Coefficients"), unsafe_allow_html=True)
        lin_df = pd.DataFrame({
            "Coefficient": lin_m.params.round(6),
            "Std Error":   lin_m.bse.round(6),
            "t-Stat":      lin_m.tvalues.round(4),
            "P-Value":     lin_m.pvalues.round(4),
            "Sig":         lin_m.pvalues < 0.05,
        })
        st.dataframe(lin_df, use_container_width=True)

        st.markdown(sec("📋 Poisson Regression Coefficients"), unsafe_allow_html=True)
        poi_df = pd.DataFrame({
            "Coefficient": poi_m.params.round(6),
            "Std Error":   poi_m.bse.round(6),
            "z-Stat":      poi_m.tvalues.round(4),
            "P-Value":     poi_m.pvalues.round(4),
            "Sig":         poi_m.pvalues < 0.05,
        })
        st.dataframe(poi_df, use_container_width=True)

        mean_y, var_y = float(y_reg.mean()), float(y_reg.var())
        st.markdown(sec("📊 Dispersion Check (Poisson Assumption)"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(card(f"{mean_y:.4f}", "Mean (Num_Projects)"),     unsafe_allow_html=True)
        c2.markdown(card(f"{var_y:.4f}",  "Variance (Num_Projects)"), unsafe_allow_html=True)
        disp = "Overdispersion ⚠️" if var_y > mean_y else "Equidispersion ✅"
        c3.markdown(card(disp, "Poisson Assumption"), unsafe_allow_html=True)

        st.markdown(sec("📈 Actual vs Predicted"), unsafe_allow_html=True)
        samp = min(800, len(y_reg))
        idx  = np.random.choice(len(y_reg), samp, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=DARK2)
        for ax in axes:
            ax.set_facecolor(DARK2)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
            ax.tick_params(colors="white")

        axes[0].scatter(y_reg.iloc[idx], y_pred_lin.iloc[idx], alpha=0.25, color=ACCENT, s=12)
        axes[0].plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], "r--", linewidth=1)
        axes[0].set_xlabel("Actual Num_Projects", color="white")
        axes[0].set_ylabel("Predicted (Linear)", color="white")
        axes[0].set_title("OLS: Actual vs Predicted", color="white")

        axes[1].scatter(y_reg.iloc[idx], y_pred_poi.iloc[idx], alpha=0.25, color=AMBER, s=12)
        axes[1].plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], "r--", linewidth=1)
        axes[1].set_xlabel("Actual Num_Projects", color="white")
        axes[1].set_ylabel("Predicted (Poisson)", color="white")
        axes[1].set_title("Poisson: Actual vs Predicted", color="white")

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(sec("📊 Distribution Comparison"), unsafe_allow_html=True)
        fig, ax = dark_fig(9, 4.5)
        ax.hist(y_reg,        bins=20, alpha=0.55, color=ACCENT, label="Actual",       edgecolor="none")
        ax.hist(y_pred_lin,   bins=20, alpha=0.55, color=AMBER,  label="Linear Pred",  edgecolor="none")
        ax.hist(y_pred_poi,   bins=20, alpha=0.55, color=RED,    label="Poisson Pred", edgecolor="none")
        ax.set_xlabel("Num_Projects", color="white"); ax.set_ylabel("Frequency", color="white")
        ax.set_title("Distribution: Actual vs Predicted", color="white")
        ax.legend(facecolor=DARK2, labelcolor="white")
        st.pyplot(fig); plt.close()

        st.markdown(sec("📉 Residual Plot — OLS"), unsafe_allow_html=True)
        resid = y_reg - y_pred_lin
        fig, ax = dark_fig(9, 4)
        ax.scatter(y_pred_lin.iloc[idx], resid.iloc[idx], alpha=0.25, color=PURPLE, s=12)
        ax.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel("Predicted Values (Linear)", color="white")
        ax.set_ylabel("Residuals", color="white")
        ax.set_title("Residual Plot — Linear Regression", color="white")
        st.pyplot(fig); plt.close()

        st.markdown(sec("📋 OLS vs Poisson — Summary"), unsafe_allow_html=True)
        summ = pd.DataFrame({
            "Metric": ["AIC","Log-Likelihood","MSE","R² / Pseudo R²"],
            "OLS (Linear)": [
                round(lin_m.aic,2), round(lin_m.llf,2),
                round(mse_lin,4),   round(lin_m.rsquared,6)],
            "Poisson (GLM)": [
                round(poi_m.aic,2), round(poi_m.llf,2),
                round(mse_poi,4),   round(poi_m.pseudo_rsquared("cs"),6)],
        })
        st.dataframe(summ, use_container_width=True)

# footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#3a4a6b;font-size:.78rem;'>"
    "Advanced Machine Learning — Assignment 2 &nbsp;·&nbsp; Streamlit Dashboard"
    "</p>", unsafe_allow_html=True
)