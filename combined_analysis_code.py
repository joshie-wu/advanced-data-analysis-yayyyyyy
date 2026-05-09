# =============================================================================
# NYC Restaurant Inspection Analysis — Combined Code
# STAT GR5291/GU4291, Columbia University
#
# Authors: Boyan Shang, Joshua Wenjun Wu, Amira Gbagba, Rebeca De La Garza Evia
#
# Repository: https://github.com/joshie-wu/advanced-data-analysis-yayyyyyy
#
# This file aggregates all analysis code into one place for submission.
# The Python sections can be run independently from the project folder.
# The R section (at the bottom) should be run via RStudio or knitted as .Rmd.
#
# Data: inspection_results_with_ntas.csv  (in project folder)
# Figures output to: writeup/figures/
# =============================================================================


# =============================================================================
# PYTHON SECTION 1: new_analysis.py
# Main EDA, score distribution, temporal trend, violation heatmap,
# and pre-inspection logistic regression (no data leakage).
# =============================================================================

"""
Clean analysis script for NYC Restaurant Inspection Project.
Generates all figures for the writeup.

Addresses the following gaps in the existing analysis:
  1. Formal bump-at-13 test (chi-square)
  2. Temporal trend in scores
  3. Borough x violation type analysis (Research Q3 - was never done)
  4. Clean predictive logistic model (pre-inspection features only, no leakage)
  5. Publication-quality EDA figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "./inspection_results_with_ntas.csv"
FIG_DIR     = "./writeup/figures/"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ── load and clean ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw shape: {df.shape}")

# cuisine hypercategory (match Amira's grouping)
hypercategory = {
    "American": "American", "New American": "American", "Californian": "American",
    "Steakhouse": "American", "Barbecue": "American", "Hamburgers": "American",
    "Soul Food": "American", "Cajun": "American", "Creole": "American",
    "Creole/Cajun": "American", "Tex-Mex": "American", "Southwestern": "American",
    "Chinese": "East Asian", "Japanese": "East Asian", "Korean": "East Asian",
    "Chinese/Japanese": "East Asian", "Chinese/Cuban": "East Asian",
    "Indian": "South & SE Asian", "Thai": "South & SE Asian",
    "Southeast Asian": "South & SE Asian", "Bangladeshi": "South & SE Asian",
    "Pakistani": "South & SE Asian", "Filipino": "South & SE Asian",
    "Indonesian": "South & SE Asian", "Afghan": "South & SE Asian",
    "Asian/Asian Fusion": "South & SE Asian",
    "Mexican": "Latin American", "Latin American": "Latin American",
    "Caribbean": "Latin American", "Peruvian": "Latin American",
    "Brazilian": "Latin American", "Chilean": "Latin American",
    "Polynesian": "Latin American", "Hawaiian": "Latin American",
    "Italian": "European", "French": "European", "Spanish": "European",
    "English": "European", "German": "European", "Portuguese": "European",
    "Polish": "European", "Russian": "European", "Eastern European": "European",
    "Irish": "European", "Scandinavian": "European", "Czech": "European",
    "Basque": "European", "New French": "European", "Continental": "European",
    "Haute Cuisine": "European", "Australian": "European",
    "Mediterranean": "Mediterranean", "Greek": "Mediterranean",
    "Turkish": "Mediterranean", "Lebanese": "Mediterranean",
    "Moroccan": "Mediterranean", "Tapas": "Mediterranean",
    "Middle Eastern": "Middle Eastern", "Egyptian": "Middle Eastern",
    "Iranian": "Middle Eastern", "Armenian": "Middle Eastern",
    "Jewish/Kosher": "Middle Eastern",
    "African": "African", "Ethiopian": "African",
    "Pizza": "Dish-Type", "Sandwiches": "Dish-Type", "Chicken": "Dish-Type",
    "Seafood": "Dish-Type", "Hotdogs": "Dish-Type", "Hotdogs/Pretzels": "Dish-Type",
    "Bagels/Pretzels": "Dish-Type", "Pancakes/Waffles": "Dish-Type",
    "Salads": "Dish-Type", "Soups": "Dish-Type",
    "Coffee/Tea": "Beverages & Sweets", "Juice, Smoothies, Fruit Salads": "Beverages & Sweets",
    "Bottled Beverages": "Beverages & Sweets", "Donuts": "Beverages & Sweets",
    "Frozen Desserts": "Beverages & Sweets", "Bakery Products/Desserts": "Beverages & Sweets",
    "Nuts/Confectionary": "Beverages & Sweets",
    "Vegan": "Dietary Style", "Vegetarian": "Dietary Style",
    "Fruits/Vegetables": "Dietary Style",
}

if "hypercategory" not in df.columns:
    df["hypercategory"] = df["cuisine"].map(hypercategory)

df = df.dropna(subset=["score"])
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])
df["score"] = df["score"].astype(int)

# parse dates and extract year
df["inspection_date"] = pd.to_datetime(df["inspection_date"], format="%m/%d/%Y", errors="coerce")
df["year"] = df["inspection_date"].dt.year

# grade binary
df["grade_binary"] = (df["score"] <= 13).astype(int)

# filter to cycle initial inspections for cleaner analysis
cycle_initial = df[df["inspection_type"] == "Cycle Inspection / Initial Inspection"].copy()
# exclude Staten Island and Dietary Style (sparse)
clean = df[
    (~df["borough"].isin(["Staten Island", "0"])) &
    (df["hypercategory"].notna()) &
    (~df["hypercategory"].isin(["Mixed/Uncategorized", "Dietary Style"]))
].copy()

print(f"  Clean shape (all inspection types, excl. SI): {clean.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Score distribution with bump highlighted (publication-quality)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Score distribution / bump at 13...")

score_counts = (
    clean[clean["score"] <= 50]
    .groupby("score")
    .size()
    .reset_index(name="count")
)

fig, ax = plt.subplots(figsize=(10, 5))

colors = [
    "seagreen" if s <= 13 else ("steelblue" if s <= 27 else "firebrick")
    for s in score_counts["score"]
]
ax.bar(score_counts["score"], score_counts["count"], color=colors, edgecolor="white", width=0.9)

# Wellington's expected flat baseline (mean of scores 14-16)
baseline_mask = score_counts["score"].between(14, 16)
baseline_val = score_counts.loc[baseline_mask, "count"].mean()
ax.axhline(baseline_val, xmin=12/50, xmax=16/50, color="black", linestyle="--",
           linewidth=1.8, label=f"Expected (flat baseline ≈ {baseline_val:.0f})")

# annotate the bump
ax.annotate("A/B cutoff\n(score = 13)", xy=(13, score_counts.loc[score_counts["score"]==13,"count"].values[0]),
            xytext=(20, score_counts.loc[score_counts["score"]==13,"count"].values[0]*0.85),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
ax.annotate("B/C cutoff\n(score = 28)", xy=(28, score_counts.loc[score_counts["score"]==28,"count"].values[0]),
            xytext=(35, score_counts.loc[score_counts["score"]==28,"count"].values[0]*0.85),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="seagreen", label="A grade (score ≤ 13)"),
    Patch(facecolor="steelblue", label="B grade (14–27)"),
    Patch(facecolor="firebrick", label="C grade (≥ 28)"),
    plt.Line2D([0], [0], color="black", linestyle="--", label=f"Expected baseline ≈ {baseline_val:.0f}"),
]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_xlabel("Inspection Score (lower is better)")
ax.set_ylabel("Number of Inspections")
ax.set_title("NYC Restaurant Inspection Score Distribution (2010–2026)\nwith Grade Cutoffs Highlighted")
ax.set_xlim(-0.5, 50.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig1_score_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved fig1_score_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Formal bump test — chi-square goodness of fit
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 2: Formal bump test (chi-square)...")

# Method: compare observed counts at scores 12-13 to expected
# Expected = mean count in "control" region 14-21 (away from both cutoffs)
# Logic: if grading were uniform, counts at 12-13 should equal the baseline

bump_region   = score_counts[score_counts["score"].between(12, 13)]
control_region = score_counts[score_counts["score"].between(14, 21)]

expected_per_score = control_region["count"].mean()
observed_12_13     = bump_region["count"].sum()
expected_12_13     = expected_per_score * 2

excess             = observed_12_13 - expected_12_13
excess_pct         = excess / expected_12_13 * 100

# chi-square test: observed vs expected across scores 12-21
test_region = score_counts[score_counts["score"].between(12, 21)].copy()
expected_uniform = np.full(len(test_region), test_region["count"].mean())
chi2, pval = stats.chisquare(f_obs=test_region["count"].values, f_exp=expected_uniform)

print(f"  Observed at scores 12-13: {observed_12_13:,.0f}")
print(f"  Expected at scores 12-13: {expected_12_13:,.0f}")
print(f"  Excess: {excess:,.0f} ({excess_pct:.1f}%)")
print(f"  Chi-square (scores 12-21): chi2={chi2:.1f}, p={pval:.2e}")

# Plot: zoomed-in view of the bump
fig, ax = plt.subplots(figsize=(8, 4.5))
zoom = score_counts[score_counts["score"].between(5, 25)].copy()
cols = ["seagreen" if s <= 13 else "steelblue" for s in zoom["score"]]
ax.bar(zoom["score"], zoom["count"], color=cols, edgecolor="white", width=0.9)
ax.axhline(expected_per_score, color="black", linestyle="--", linewidth=2,
           label=f"Expected baseline ≈ {expected_per_score:,.0f}/score")
ax.axvline(13.5, color="darkgreen", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(13.7, ax.get_ylim()[1]*0.9, "A/B cutoff", color="darkgreen", fontsize=9)
ax.set_xlabel("Inspection Score")
ax.set_ylabel("Count")
ax.set_title(f"Score Bump at Cutoff (scores 5–25)\n"
             f"Excess at 12–13: {excess:,.0f} inspections (+{excess_pct:.0f}%),  "
             f"χ²={chi2:.1f}, p<0.001")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig2_bump_test.png", bbox_inches="tight")
plt.close()
print("  Saved fig2_bump_test.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Temporal trend — mean initial-inspection score by year
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 3: Temporal trend...")

temporal = (
    clean[
        clean["inspection_type"].str.contains("Initial", na=False) &
        clean["year"].between(2015, 2025)
    ]
    .groupby("year")["score"]
    .agg(["mean", "median", "sem"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(temporal["year"], temporal["mean"], marker="o", color="steelblue",
        linewidth=2, label="Mean score")
ax.fill_between(temporal["year"],
                temporal["mean"] - 1.96 * temporal["sem"],
                temporal["mean"] + 1.96 * temporal["sem"],
                alpha=0.2, color="steelblue", label="95% CI")
ax.plot(temporal["year"], temporal["median"], marker="s", color="firebrick",
        linestyle="--", linewidth=1.5, label="Median score")
ax.axhline(13, color="seagreen", linestyle=":", linewidth=1.5, alpha=0.7,
           label="A-grade cutoff (13)")
ax.set_xlabel("Year")
ax.set_ylabel("Inspection Score (lower = better)")
ax.set_title("Trend in NYC Restaurant Inspection Scores Over Time\n(Initial Cycle Inspections Only)")
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(FIG_DIR + "fig3_temporal_trend.png", bbox_inches="tight")
plt.close()
print("  Saved fig3_temporal_trend.png")
print(temporal[["year","mean","median"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Score by borough (violin + boxplot)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 4: Score by borough...")

boro_order = (
    clean.groupby("borough")["score"].median()
    .sort_values()
    .index.tolist()
)
boro_clean = clean[clean["borough"].isin(boro_order) & clean["score"].between(0,60)]

fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=boro_clean, x="borough", y="score", order=boro_order,
               inner="box", palette="Set2", ax=ax, cut=0)
ax.axhline(13, color="seagreen", linestyle="--", linewidth=1.5, label="A-grade cutoff (13)")
ax.axhline(28, color="firebrick", linestyle="--", linewidth=1.5, alpha=0.6, label="B-grade cutoff (28)")
ax.set_xlabel("Borough")
ax.set_ylabel("Inspection Score (lower = better)")
ax.set_title("Distribution of Inspection Scores by Borough")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig4_score_by_borough.png", bbox_inches="tight")
plt.close()
print("  Saved fig4_score_by_borough.png")

# print medians
print(boro_clean.groupby("borough")["score"].median().sort_values())


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Borough x Violation Type heatmap (Research Question 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 5: Borough x violation heatmap (RQ3)...")

viol_boro = clean.dropna(subset=["violation_category", "borough"]).copy()
viol_boro = viol_boro[viol_boro["borough"].isin(["Bronx","Brooklyn","Manhattan","Queens"])]

# compute proportion of each violation type within each borough
# (% of that borough's inspections that had this violation)
boro_totals = viol_boro.groupby("borough").size()
heatmap_df  = (
    viol_boro.groupby(["borough", "violation_category"])
    .size()
    .reset_index(name="count")
)
heatmap_df["pct"] = heatmap_df.apply(
    lambda r: r["count"] / boro_totals[r["borough"]] * 100, axis=1
)
pivot = heatmap_df.pivot(index="violation_category", columns="borough", values="pct").fillna(0)

# chi-square test of independence
contingency = heatmap_df.pivot(index="violation_category", columns="borough", values="count").fillna(0)
chi2_viol, pval_viol, dof_viol, _ = stats.chi2_contingency(contingency)
print(f"  Chi-square (violation x borough): chi2={chi2_viol:.1f}, df={dof_viol}, p={pval_viol:.2e}")

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "% of inspections in borough"})
ax.set_title(f"Violation Type Rate by Borough (% of inspections)\n"
             f"χ²={chi2_viol:.0f}, df={dof_viol}, p < 0.001")
ax.set_xlabel("Borough")
ax.set_ylabel("Violation Category")
plt.tight_layout()
plt.savefig(FIG_DIR + "fig5_violation_by_borough.png", bbox_inches="tight")
plt.close()
print("  Saved fig5_violation_by_borough.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Score by cuisine hypercategory
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 6: Score by cuisine hypercategory...")

cat_order = (
    clean.groupby("hypercategory")["score"].median()
    .sort_values()
    .index.tolist()
)
cat_clean = clean[clean["score"].between(0,60)]

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=cat_clean, y="hypercategory", x="score", order=cat_order,
            palette="Set3", ax=ax, fliersize=1)
ax.axvline(13, color="seagreen", linestyle="--", linewidth=1.5, label="A-grade cutoff")
ax.axvline(28, color="firebrick", linestyle="--", linewidth=1.5, alpha=0.6, label="B-grade cutoff")
ax.set_xlabel("Inspection Score (lower = better)")
ax.set_ylabel("Cuisine Hypercategory")
ax.set_title("Distribution of Inspection Scores by Cuisine Type")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig6_score_by_cuisine.png", bbox_inches="tight")
plt.close()
print("  Saved fig6_score_by_cuisine.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 & 8: Clean predictive logistic regression (no leakage)
# Pre-inspection features only: borough, hypercategory, MdEWrkE, Pop_1E, year
# This answers: BEFORE an inspection, which restaurants are more likely to fail?
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 7-8: Clean predictive logistic model...")

# Use one row per inspection event per restaurant (avoid pseudo-replication)
# Take the most recent initial cycle inspection per restaurant
initial = clean[
    clean["inspection_type"].str.contains("Initial", na=False) &
    clean["score"].notna() &
    clean["year"].between(2015, 2025)
].copy()

# aggregate to one row per inspection (score is duplicated across violations)
insp = (
    initial.groupby(["restaurant_id", "inspection_date"])
    .agg(
        score=("score", "first"),
        borough=("borough", "first"),
        hypercategory=("hypercategory", "first"),
        MdEWrkE=("MdEWrkE", "first"),
        Pop_1E=("Pop_1E", "first"),
        year=("year", "first"),
    )
    .reset_index()
)
insp["grade_binary"] = (insp["score"] <= 13).astype(int)
insp = insp.dropna(subset=["borough", "hypercategory", "MdEWrkE", "Pop_1E"])
insp = insp[insp["hypercategory"] != "Mixed/Uncategorized"]

print(f"  Inspection-level df shape: {insp.shape}")
print(f"  A-grade rate: {insp['grade_binary'].mean():.1%}")

model_df = pd.get_dummies(
    insp[["grade_binary", "borough", "hypercategory", "MdEWrkE", "Pop_1E", "year"]],
    columns=["borough", "hypercategory"],
    drop_first=True
)

X = model_df.drop(columns=["grade_binary"])
y = model_df["grade_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LASSO logistic regression (tuned C)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(C=1.0, max_iter=2000, random_state=42))
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print(f"  Accuracy: {acc:.4f}")
print(f"  AUC:      {auc:.4f}")
print(f"  Confusion matrix:\n{cm}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
ax.plot([0,1],[0,1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Pre-Inspection Logistic Regression\n(Borough, Cuisine, Neighborhood Variables)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig7_roc_curve.png", bbox_inches="tight")
plt.close()
print("  Saved fig7_roc_curve.png")

# coefficient plot (top predictors by |coef|)
coefs = pipe.named_steps["logreg"].coef_[0]
feat_names = X_train.columns.tolist()
coef_df = pd.DataFrame({"Feature": feat_names, "Coef": coefs})
coef_df["Abs"] = coef_df["Coef"].abs()
coef_df = coef_df.sort_values("Abs", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7, 5))
colors_coef = ["seagreen" if c > 0 else "firebrick" for c in coef_df["Coef"]]
ax.barh(coef_df["Feature"][::-1], coef_df["Coef"][::-1], color=colors_coef[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Logistic Regression Coefficient")
ax.set_title("Top Predictors of Earning an A Grade\n(Pre-Inspection Features Only)")
plt.tight_layout()
plt.savefig(FIG_DIR + "fig8_coef_plot.png", bbox_inches="tight")
plt.close()
print("  Saved fig8_coef_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary stats table (for writeup)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Summary statistics ──")
print(f"Total inspection records (clean): {len(clean):,}")
print(f"Unique restaurants: {clean['restaurant_id'].nunique():,}")
print(f"Boroughs: {sorted(clean['borough'].dropna().unique())}")
print(f"Years covered: {clean['year'].min()} – {clean['year'].max()}")
print(f"A-grade rate (score<=13): {(clean['score']<=13).mean():.1%}")
print(f"Score mean ± sd: {clean['score'].mean():.1f} ± {clean['score'].std():.1f}")
print(f"Score median: {clean['score'].median():.0f}")

print("\nDone. All figures saved to", FIG_DIR)


# =============================================================================
# PYTHON SECTION 2: extended_analysis.py
# OLS regression coefficients, violation-level logistic regression,
# A-grade rate by year and borough, cuisine violin plots.
# =============================================================================

"""
Extended analysis for NYC Restaurant Inspection writeup.
Generates additional figures beyond new_analysis.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "./inspection_results_with_ntas.csv"
FIG_DIR   = "./writeup/figures/"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#333",
    "text.color": "#333",
    "xtick.color": "#555",
    "ytick.color": "#555",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading...")
df = pd.read_csv(DATA_PATH)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df["inspection_date"] = pd.to_datetime(df["inspection_date"], format="%m/%d/%Y", errors="coerce")
df["year"] = df["inspection_date"].dt.year

hypercategory_map = {
    "American":"American","New American":"American","Californian":"American",
    "Steakhouse":"American","Barbecue":"American","Hamburgers":"American",
    "Soul Food":"American","Cajun":"American","Creole":"American",
    "Creole/Cajun":"American","Tex-Mex":"American","Southwestern":"American",
    "Chinese":"East Asian","Japanese":"East Asian","Korean":"East Asian",
    "Chinese/Japanese":"East Asian","Chinese/Cuban":"East Asian",
    "Indian":"South & SE Asian","Thai":"South & SE Asian","Southeast Asian":"South & SE Asian",
    "Bangladeshi":"South & SE Asian","Pakistani":"South & SE Asian","Filipino":"South & SE Asian",
    "Indonesian":"South & SE Asian","Afghan":"South & SE Asian","Asian/Asian Fusion":"South & SE Asian",
    "Mexican":"Latin American","Latin American":"Latin American","Caribbean":"Latin American",
    "Peruvian":"Latin American","Brazilian":"Latin American","Chilean":"Latin American",
    "Polynesian":"Latin American","Hawaiian":"Latin American",
    "Italian":"European","French":"European","Spanish":"European","English":"European",
    "German":"European","Portuguese":"European","Polish":"European","Russian":"European",
    "Eastern European":"European","Irish":"European","Scandinavian":"European",
    "Czech":"European","Basque":"European","New French":"European",
    "Continental":"European","Haute Cuisine":"European","Australian":"European",
    "Mediterranean":"Mediterranean","Greek":"Mediterranean","Turkish":"Mediterranean",
    "Lebanese":"Mediterranean","Moroccan":"Mediterranean","Tapas":"Mediterranean",
    "Middle Eastern":"Middle Eastern","Egyptian":"Middle Eastern","Iranian":"Middle Eastern",
    "Armenian":"Middle Eastern","Jewish/Kosher":"Middle Eastern",
    "African":"African","Ethiopian":"African",
    "Pizza":"Dish-Type","Sandwiches":"Dish-Type","Chicken":"Dish-Type","Seafood":"Dish-Type",
    "Hotdogs":"Dish-Type","Hotdogs/Pretzels":"Dish-Type","Bagels/Pretzels":"Dish-Type",
    "Pancakes/Waffles":"Dish-Type","Salads":"Dish-Type","Soups":"Dish-Type",
    "Coffee/Tea":"Beverages & Sweets","Juice, Smoothies, Fruit Salads":"Beverages & Sweets",
    "Bottled Beverages":"Beverages & Sweets","Donuts":"Beverages & Sweets",
    "Frozen Desserts":"Beverages & Sweets","Bakery Products/Desserts":"Beverages & Sweets",
    "Nuts/Confectionary":"Beverages & Sweets",
}

if "hypercategory" not in df.columns:
    df["hypercategory"] = df["cuisine"].map(hypercategory_map)

BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens"]
clean = df[
    df["borough"].isin(BOROUGHS) &
    df["score"].notna() &
    df["hypercategory"].notna() &
    ~df["hypercategory"].isin(["Mixed/Uncategorized", "Dietary Style"])
].copy()
clean["grade_binary"] = (clean["score"] <= 13).astype(int)
print(f"Clean shape: {clean.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG A: A-grade rate by year AND borough (4 lines)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig A: A-grade rate by year x borough...")

initial = clean[
    clean["inspection_type"].str.contains("Initial", na=False) &
    clean["year"].between(2015, 2025)
].copy()

# deduplicate to one row per inspection event
insp = (
    initial.groupby(["restaurant_id", "inspection_date"])
    .agg(score=("score","first"), borough=("borough","first"), year=("year","first"))
    .reset_index()
)
insp["grade_binary"] = (insp["score"] <= 13).astype(int)

boro_year = (
    insp.groupby(["year","borough"])["grade_binary"]
    .agg(["mean","count"])
    .reset_index()
)
boro_year.columns = ["year","borough","agrade_rate","n"]
# SE for proportions
boro_year["se"] = np.sqrt(boro_year["agrade_rate"]*(1-boro_year["agrade_rate"])/boro_year["n"])

palette = {"Bronx":"#e15759","Brooklyn":"#4e79a7","Manhattan":"#59a14f","Queens":"#f28e2b"}

fig, ax = plt.subplots(figsize=(9,5))
for boro in BOROUGHS:
    sub = boro_year[boro_year["borough"]==boro]
    ax.plot(sub["year"], sub["agrade_rate"]*100, marker="o", linewidth=2,
            color=palette[boro], label=boro)
    ax.fill_between(sub["year"],
                    (sub["agrade_rate"]-1.96*sub["se"])*100,
                    (sub["agrade_rate"]+1.96*sub["se"])*100,
                    alpha=0.12, color=palette[boro])

ax.set_xlabel("Year")
ax.set_ylabel("A-Grade Rate (%)")
ax.set_title("A-Grade Rate by Year and Borough\n(Initial Cycle Inspections Only, One Row Per Inspection)")
ax.legend(title="Borough", fontsize=9)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(FIG_DIR + "figA_agrade_by_year_borough.png", bbox_inches="tight")
plt.close()
print("  Saved figA")

# print the rates
print(boro_year[["year","borough","agrade_rate","n"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# FIG B: Bump by borough — excess A-grade inspections at score 12-13
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig B: Bump by borough...")

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
axes = axes.flatten()

bump_results = []
for i, boro in enumerate(BOROUGHS):
    sub = clean[clean["borough"]==boro]
    counts = sub[sub["score"]<=50].groupby("score").size().reset_index(name="count")
    baseline_mask = counts["score"].between(14, 21)
    expected_per = counts.loc[baseline_mask, "count"].mean()
    obs_12_13    = counts.loc[counts["score"].between(12,13), "count"].sum()
    exp_12_13    = expected_per * 2
    excess       = obs_12_13 - exp_12_13
    excess_pct   = excess / exp_12_13 * 100

    bump_results.append({"borough":boro,"excess":excess,"excess_pct":excess_pct,
                         "observed":obs_12_13,"expected":exp_12_13})

    ax = axes[i]
    zoom = counts[counts["score"].between(5,25)]
    cols = ["seagreen" if s<=13 else "steelblue" for s in zoom["score"]]
    ax.bar(zoom["score"], zoom["count"], color=cols, edgecolor="white", width=0.9)
    ax.axhline(expected_per, color="black", linestyle="--", linewidth=1.8)
    ax.set_title(f"{boro}\n+{excess_pct:.0f}% excess at 12-13", fontsize=11)
    ax.set_xlabel("Score" if i >= 2 else "")
    ax.set_ylabel("Count" if i % 2 == 0 else "")

fig.suptitle("Grade-Boundary Spike at Score 13 by Borough\n(Dashed = Expected Flat Baseline)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR + "figB_bump_by_borough.png", bbox_inches="tight")
plt.close()
print("  Saved figB")
for r in bump_results:
    print(f"  {r['borough']}: observed={r['observed']:.0f}, expected={r['expected']:.0f}, excess={r['excess_pct']:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FIG C: OLS regression — score on borough + cuisine + year (coefficient plot)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig C: OLS regression coefficient plot...")

# Use deduplicated inspection-level data
initial_all = clean[
    clean["inspection_type"].str.contains("Initial", na=False) &
    clean["score"].notna() &
    clean["year"].between(2015, 2025)
].copy()

reg_df = (
    initial_all.groupby(["restaurant_id","inspection_date"])
    .agg(score=("score","first"), borough=("borough","first"),
         hypercategory=("hypercategory","first"), year=("year","first"),
         MdEWrkE=("MdEWrkE","first"), Pop_1E=("Pop_1E","first"))
    .reset_index()
    .dropna(subset=["score","borough","hypercategory"])
)

# OLS with statsmodels for CIs
reg_df["borough_c"] = pd.Categorical(reg_df["borough"], categories=["Bronx","Brooklyn","Manhattan","Queens"])
reg_df["hcat_c"] = pd.Categorical(reg_df["hypercategory"])
reg_df["year_centered"] = reg_df["year"] - 2015

formula = "score ~ C(borough_c, Treatment('Bronx')) + C(hcat_c, Treatment('African')) + year_centered"
ols_model = smf.ols(formula, data=reg_df).fit()

print(ols_model.summary().tables[1])

# extract key coefficients (borough + year, skip cuisine interactions for main plot)
coef_names = {
    "C(borough_c, Treatment('Bronx'))[T.Brooklyn]":    "Brooklyn (vs. Bronx)",
    "C(borough_c, Treatment('Bronx'))[T.Manhattan]":   "Manhattan (vs. Bronx)",
    "C(borough_c, Treatment('Bronx'))[T.Queens]":      "Queens (vs. Bronx)",
    "year_centered":                                    "Year (centered at 2015)",
    "C(hcat_c, Treatment('African'))[T.American]":     "American cuisine",
    "C(hcat_c, Treatment('African'))[T.Beverages & Sweets]": "Beverages & Sweets",
    "C(hcat_c, Treatment('African'))[T.Dish-Type]":    "Dish-Type cuisine",
    "C(hcat_c, Treatment('African'))[T.East Asian]":   "East Asian cuisine",
    "C(hcat_c, Treatment('African'))[T.European]":     "European cuisine",
    "C(hcat_c, Treatment('African'))[T.Latin American]": "Latin American cuisine",
    "C(hcat_c, Treatment('African'))[T.Mediterranean]": "Mediterranean cuisine",
    "C(hcat_c, Treatment('African'))[T.Middle Eastern]": "Middle Eastern cuisine",
    "C(hcat_c, Treatment('African'))[T.South & SE Asian]": "South & SE Asian cuisine",
}

plot_data = []
for orig, label in coef_names.items():
    if orig in ols_model.params:
        plot_data.append({
            "label": label,
            "coef":  ols_model.params[orig],
            "ci_lo": ols_model.conf_int().loc[orig, 0],
            "ci_hi": ols_model.conf_int().loc[orig, 1],
        })

pdf = pd.DataFrame(plot_data).sort_values("coef")

fig, ax = plt.subplots(figsize=(8, 6))
colors_ols = ["seagreen" if c < 0 else "firebrick" for c in pdf["coef"]]
ax.barh(pdf["label"], pdf["coef"], color=colors_ols, alpha=0.75, height=0.6)
ax.errorbar(pdf["coef"], pdf["label"],
            xerr=[pdf["coef"]-pdf["ci_lo"], pdf["ci_hi"]-pdf["coef"]],
            fmt="none", color="black", linewidth=1.2, capsize=3)
ax.axvline(0, color="black", linewidth=0.9)
ax.set_xlabel("OLS Coefficient (change in inspection score)")
ax.set_title("OLS Regression: Predictors of Inspection Score\n(Baseline: Bronx, African cuisine, Year 2015)")
plt.tight_layout()
plt.savefig(FIG_DIR + "figC_ols_coefs.png", bbox_inches="tight")
plt.close()
print("  Saved figC")
print(f"  Year coefficient: {ols_model.params['year_centered']:.3f}, "
      f"p={ols_model.pvalues['year_centered']:.4f}")
print(f"  R-squared: {ols_model.rsquared:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG D: Violation-level logistic — which violations predict failing an A?
# (Inferential framing: given a violation occurred, does its TYPE predict grade?)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig D: Violation-level logistic regression...")

viol_df = clean[
    clean["violation_category"].notna() &
    clean["score"].notna() &
    clean["inspection_type"].str.contains("Initial", na=False)
].copy()

# one-hot encode violation_category, borough, hypercategory
model_df = pd.get_dummies(
    viol_df[["grade_binary","violation_category","borough","hypercategory","MdEWrkE","Pop_1E"]].dropna(),
    columns=["violation_category","borough","hypercategory"],
    drop_first=True
)

X = model_df.drop(columns=["grade_binary"])
y = model_df["grade_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe_viol = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(C=1.0, max_iter=2000, random_state=42))
])
pipe_viol.fit(X_train, y_train)

y_pred = pipe_viol.predict(X_test)
y_prob = pipe_viol.predict_proba(X_test)[:, 1]
auc_viol = roc_auc_score(y_test, y_prob)
acc_viol = accuracy_score(y_test, y_pred)
print(f"  Violation model AUC: {auc_viol:.4f}, Accuracy: {acc_viol:.4f}")

# extract violation-category coefficients only
coefs_all = pipe_viol.named_steps["logreg"].coef_[0]
feat_names = X_train.columns.tolist()

viol_coef = pd.DataFrame({"Feature": feat_names, "Coef": coefs_all})
viol_coef = viol_coef[viol_coef["Feature"].str.startswith("violation_category_")].copy()
viol_coef["Label"] = viol_coef["Feature"].str.replace("violation_category_", "", regex=False)
viol_coef["Abs"] = viol_coef["Coef"].abs()
viol_coef = viol_coef.sort_values("Coef")  # sort by coef value for readability

fig, ax = plt.subplots(figsize=(8, 6))
cols_viol = ["seagreen" if c > 0 else "firebrick" for c in viol_coef["Coef"]]
ax.barh(viol_coef["Label"], viol_coef["Coef"], color=cols_viol, alpha=0.8, height=0.65)
ax.axvline(0, color="black", linewidth=0.9)
ax.set_xlabel("Logistic Regression Coefficient\n(positive = associated with A grade)")
ax.set_title(f"Violation Type Association with Earning an A Grade\n"
             f"(Violation-Level Logistic Regression, AUC = {auc_viol:.3f})\n"
             f"Controlling for Borough, Cuisine, and Neighborhood Income")
plt.tight_layout()
plt.savefig(FIG_DIR + "figD_violation_coefs.png", bbox_inches="tight")
plt.close()
print("  Saved figD")

# ROC for violation model
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc_viol:.3f}")
ax.plot([0,1],[0,1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve: Violation-Type Logistic Regression")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR + "figD_roc_violation.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG E: Score distribution by cuisine (improved, horizontal boxplot)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig E: Score by cuisine (improved)...")

cat_medians = clean.groupby("hypercategory")["score"].median().sort_values()
cat_order = cat_medians.index.tolist()
cat_counts = clean.groupby("hypercategory").size()

fig, ax = plt.subplots(figsize=(9, 5.5))
parts = ax.violinplot(
    [clean.loc[clean["hypercategory"]==c, "score"].clip(0,60).values for c in cat_order],
    positions=range(len(cat_order)),
    vert=False, widths=0.7, showmedians=True
)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.5)
parts["cmedians"].set_color("navy")
parts["cmedians"].set_linewidth(2)

ax.axvline(13, color="seagreen", linestyle="--", linewidth=1.5, label="A-grade cutoff (13)")
ax.axvline(28, color="firebrick", linestyle="--", linewidth=1.5, alpha=0.6, label="B-grade cutoff (28)")

labels = [f"{c}\n(n={cat_counts[c]:,})" for c in cat_order]
ax.set_yticks(range(len(cat_order)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Inspection Score (lower is better)")
ax.set_title("Inspection Score Distribution by Cuisine Hypercategory")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR + "figE_score_by_cuisine_violin.png", bbox_inches="tight")
plt.close()
print("  Saved figE")


# ══════════════════════════════════════════════════════════════════════════════
# Print summary of all model results for writeup
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- MODEL SUMMARY FOR WRITEUP ---")
print(f"Pre-inspection logistic: AUC=0.60, Acc=59.4%")
print(f"Violation-level logistic: AUC={auc_viol:.3f}, Acc={acc_viol:.3f}")
print(f"OLS R-squared: {ols_model.rsquared:.4f}")
print(f"OLS year coef: {ols_model.params['year_centered']:.3f} pts/year")
mnh = ols_model.params["C(borough_c, Treatment('Bronx'))[T.Manhattan]"]
bk  = ols_model.params["C(borough_c, Treatment('Bronx'))[T.Brooklyn]"]
qns = ols_model.params["C(borough_c, Treatment('Bronx'))[T.Queens]"]
print(f"OLS Manhattan coef: {mnh:.3f}")
print(f"OLS Brooklyn coef:  {bk:.3f}")
print(f"OLS Queens coef:    {qns:.3f}")
print("\nViolation coefficients (sorted):")
print(viol_coef[["Label","Coef"]].to_string(index=False))


# =============================================================================
# PYTHON SECTION 3: analysis_extras.py
# Density discontinuity / McCrary-style bump test with bootstrap CI,
# restaurant-level boundary clustering, model comparison (logistic / RF / GBT),
# ANOVA diagnostics, consecutive inspection trajectory analysis.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "./inspection_results_with_ntas.csv"
FIG_DIR = "./writeup/figures/"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#333",
    "text.color": "#333",
    "xtick.color": "#555",
    "ytick.color": "#555",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df["inspection_date"] = pd.to_datetime(df["inspection_date"], format="%m/%d/%Y", errors="coerce")
df["year"] = df["inspection_date"].dt.year

hypercategory_map = {
    "American":"American","New American":"American","Californian":"American",
    "Steakhouse":"American","Barbecue":"American","Hamburgers":"American",
    "Soul Food":"American","Cajun":"American","Creole":"American",
    "Creole/Cajun":"American","Tex-Mex":"American","Southwestern":"American",
    "Chinese":"East Asian","Japanese":"East Asian","Korean":"East Asian",
    "Chinese/Japanese":"East Asian","Chinese/Cuban":"East Asian",
    "Indian":"South & SE Asian","Thai":"South & SE Asian","Southeast Asian":"South & SE Asian",
    "Bangladeshi":"South & SE Asian","Pakistani":"South & SE Asian","Filipino":"South & SE Asian",
    "Indonesian":"South & SE Asian","Afghan":"South & SE Asian","Asian/Asian Fusion":"South & SE Asian",
    "Mexican":"Latin American","Latin American":"Latin American","Caribbean":"Latin American",
    "Peruvian":"Latin American","Brazilian":"Latin American","Chilean":"Latin American",
    "Polynesian":"Latin American","Hawaiian":"Latin American",
    "Italian":"European","French":"European","Spanish":"European","English":"European",
    "German":"European","Portuguese":"European","Polish":"European","Russian":"European",
    "Eastern European":"European","Irish":"European","Scandinavian":"European",
    "Czech":"European","Basque":"European","New French":"European",
    "Continental":"European","Haute Cuisine":"European","Australian":"European",
    "Mediterranean":"Mediterranean","Greek":"Mediterranean","Turkish":"Mediterranean",
    "Lebanese":"Mediterranean","Moroccan":"Mediterranean","Tapas":"Mediterranean",
    "Middle Eastern":"Middle Eastern","Egyptian":"Middle Eastern","Iranian":"Middle Eastern",
    "Armenian":"Middle Eastern","Jewish/Kosher":"Middle Eastern",
    "African":"African","Ethiopian":"African",
    "Pizza":"Dish-Type","Sandwiches":"Dish-Type","Chicken":"Dish-Type","Seafood":"Dish-Type",
    "Hotdogs":"Dish-Type","Hotdogs/Pretzels":"Dish-Type","Bagels/Pretzels":"Dish-Type",
    "Pancakes/Waffles":"Dish-Type","Salads":"Dish-Type","Soups":"Dish-Type",
    "Coffee/Tea":"Beverages & Sweets","Juice, Smoothies, Fruit Salads":"Beverages & Sweets",
    "Bottled Beverages":"Beverages & Sweets","Donuts":"Beverages & Sweets",
    "Frozen Desserts":"Beverages & Sweets","Bakery Products/Desserts":"Beverages & Sweets",
    "Nuts/Confectionary":"Beverages & Sweets",
}

if "hypercategory" not in df.columns:
    df["hypercategory"] = df["cuisine"].map(hypercategory_map)

BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens"]
clean = df[
    df["borough"].isin(BOROUGHS) &
    df["score"].notna() &
    df["hypercategory"].notna() &
    ~df["hypercategory"].isin(["Mixed/Uncategorized", "Dietary Style"])
].copy()
clean["grade_binary"] = (clean["score"] <= 13).astype(int)


# ---- 1. McCrary-style density discontinuity test ----
# Fit a degree-2 polynomial to the score histogram excluding the
# boundary region (scores 8-16), then extrapolate into the boundary
# to get a smooth expected density. This is more defensible than
# a flat baseline.

print("\n1. McCrary-style density discontinuity...")

counts = (
    clean[clean["score"].between(1, 50)]
    .groupby("score").size()
    .reset_index(name="count")
)
counts = counts.set_index("score").reindex(range(1, 51), fill_value=0).reset_index()
counts.columns = ["score", "count"]

# exclude boundary region from the polynomial fit
EXCL_LOW, EXCL_HIGH = 8, 16
fit_data = counts[~counts["score"].between(EXCL_LOW, EXCL_HIGH)].copy()
fit_data = fit_data[fit_data["score"].between(1, 45)]

poly_coefs = np.polyfit(fit_data["score"], fit_data["count"], deg=4)
poly_fn = np.poly1d(poly_coefs)

counts["expected_poly"] = poly_fn(counts["score"])
counts["expected_poly"] = counts["expected_poly"].clip(lower=0)

obs_12_13 = counts.loc[counts["score"].isin([12, 13]), "count"].sum()
exp_12_13 = counts.loc[counts["score"].isin([12, 13]), "expected_poly"].sum()
excess = obs_12_13 - exp_12_13
excess_pct = excess / exp_12_13 * 100

print(f"  Polynomial-based expected at 12-13: {exp_12_13:.0f}")
print(f"  Observed at 12-13: {obs_12_13:.0f}")
print(f"  Excess: {excess:.0f} ({excess_pct:.1f}%)")

# bootstrap CI for the excess
np.random.seed(42)
n_boot = 2000
boot_excess = []
all_scores = clean["score"].dropna().astype(int).values
all_scores = all_scores[all_scores.between(1, 50) if hasattr(all_scores, 'between') else (all_scores >= 1) & (all_scores <= 50)]

for _ in range(n_boot):
    sample = np.random.choice(all_scores, size=len(all_scores), replace=True)
    boot_counts = pd.Series(sample).value_counts().reindex(range(1, 51), fill_value=0)
    boot_df = pd.DataFrame({"score": range(1, 51), "count": boot_counts.values})
    boot_fit = boot_df[~boot_df["score"].between(EXCL_LOW, EXCL_HIGH) & boot_df["score"].between(1, 45)]
    boot_coefs = np.polyfit(boot_fit["score"], boot_fit["count"], deg=4)
    boot_fn = np.poly1d(boot_coefs)
    boot_exp = boot_fn(np.array([12, 13])).clip(min=0).sum()
    boot_obs = boot_counts.loc[[12, 13]].sum()
    boot_excess.append(boot_obs - boot_exp)

boot_ci_lo = np.percentile(boot_excess, 2.5)
boot_ci_hi = np.percentile(boot_excess, 97.5)
print(f"  Bootstrap 95% CI for excess: [{boot_ci_lo:.0f}, {boot_ci_hi:.0f}]")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
zoom = counts[counts["score"].between(1, 45)]
cols = ["seagreen" if s <= 13 else ("steelblue" if s <= 27 else "firebrick") for s in zoom["score"]]
ax.bar(zoom["score"], zoom["count"], color=cols, edgecolor="white", width=0.9, alpha=0.8, label="Observed")
ax.plot(zoom["score"], zoom["expected_poly"], color="black", linewidth=2,
        linestyle="--", label="Expected (degree-4 polynomial, boundary excluded)")
ax.axvline(13.5, color="darkgreen", linestyle=":", linewidth=1.5, alpha=0.8)
ax.set_xlabel("Inspection Score")
ax.set_ylabel("Count")
ax.set_title(f"Observed vs. Expected Score Density\nExcess at 12-13: {excess:,.0f} ({excess_pct:.0f}%), "
             f"95% CI [{boot_ci_lo:.0f}, {boot_ci_hi:.0f}]")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(boot_excess, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(excess, color="firebrick", linewidth=2, label=f"Observed excess: {excess:,.0f}")
ax.axvline(boot_ci_lo, color="gray", linestyle="--", linewidth=1.5)
ax.axvline(boot_ci_hi, color="gray", linestyle="--", linewidth=1.5, label="95% CI")
ax.set_xlabel("Bootstrap Excess at Scores 12-13")
ax.set_ylabel("Frequency")
ax.set_title("Bootstrap Distribution of Boundary Excess\n(2,000 Resamples)")
ax.legend(fontsize=9)

plt.suptitle("Density Discontinuity at the A-Grade Cutoff (Score = 13)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig_mccrary.png", bbox_inches="tight")
plt.close()
print("  Saved fig_mccrary.png")


# ---- 2. Kruskal-Wallis sensitivity check for ANOVA ----

print("\n2. Kruskal-Wallis sensitivity check...")

kw_boro_groups = [
    clean.loc[clean["borough"] == b, "score"].dropna().values
    for b in BOROUGHS
]
kw_boro_stat, kw_boro_p = kruskal(*kw_boro_groups)
print(f"  Kruskal-Wallis by borough: H={kw_boro_stat:.2f}, p={kw_boro_p:.4f}")

# ANOVA residual normality check using a sample
from scipy.stats import shapiro
import statsmodels.formula.api as smf

sample_for_normality = clean.sample(n=3000, random_state=42)
ols_res = smf.ols("score ~ C(borough) + C(hypercategory)", data=sample_for_normality).fit()
resids = ols_res.resid

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.hist(resids, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title("ANOVA Residual Distribution\n(Sample of 3,000 observations)")

ax = axes[1]
stats.probplot(resids, dist="norm", plot=ax)
ax.set_title("Normal Q-Q Plot of Residuals")

plt.tight_layout()
plt.savefig(FIG_DIR + "fig_anova_diagnostics.png", bbox_inches="tight")
plt.close()
print("  Saved fig_anova_diagnostics.png")


# ---- 3. Restaurant fixed-effects: do restaurants hover near 13? ----

print("\n3. Restaurant-level threshold analysis...")

initial = clean[
    clean["inspection_type"].str.contains("Initial", na=False) &
    clean["score"].notna()
].copy()

rest_insp = (
    initial.groupby(["restaurant_id", "inspection_date"])
    .agg(score=("score","first"), borough=("borough","first"))
    .reset_index()
)

# for restaurants with at least 2 inspections, compute fraction landing at 12-13
rest_summary = (
    rest_insp.groupby("restaurant_id")
    .agg(
        n_inspections=("score","count"),
        mean_score=("score","mean"),
        pct_at_boundary=("score", lambda x: ((x == 12) | (x == 13)).mean()),
        ever_at_boundary=("score", lambda x: ((x == 12) | (x == 13)).any()),
        borough=("borough","first")
    )
    .reset_index()
)
rest_multi = rest_summary[rest_summary["n_inspections"] >= 2].copy()
print(f"  Restaurants with >= 2 initial inspections: {len(rest_multi):,}")
print(f"  Mean % of inspections landing at 12-13: {rest_multi['pct_at_boundary'].mean():.1%}")
print(f"  Restaurants ever at boundary (12-13): {rest_multi['ever_at_boundary'].mean():.1%}")

# under the null, expected fraction at boundary = count at 12-13 / total
null_rate = (clean["score"].isin([12, 13])).mean()
print(f"  Expected fraction at 12-13 if random: {null_rate:.1%}")
t_stat, t_p = stats.ttest_1samp(rest_multi["pct_at_boundary"], null_rate)
print(f"  One-sample t-test vs null rate: t={t_stat:.2f}, p={t_p:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(rest_multi["pct_at_boundary"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(null_rate, color="firebrick", linewidth=2, linestyle="--",
           label=f"Expected under null ({null_rate:.1%})")
ax.axvline(rest_multi["pct_at_boundary"].mean(), color="navy", linewidth=2,
           label=f"Observed mean ({rest_multi['pct_at_boundary'].mean():.1%})")
ax.set_xlabel("Fraction of Inspections Landing at Score 12 or 13")
ax.set_ylabel("Number of Restaurants")
ax.set_title(f"Restaurant-Level Boundary Clustering\n"
             f"(Restaurants with $\\geq$2 initial inspections, n={len(rest_multi):,})\n"
             f"One-sample t-test vs null: t={t_stat:.2f}, p={t_p:.4f}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig_restaurant_threshold.png", bbox_inches="tight")
plt.close()
print("  Saved fig_restaurant_threshold.png")


# ---- 4. Better predictive modeling: Logistic vs RF vs GBT ----

print("\n4. Expanded predictive model comparison...")

initial_insp = clean[
    clean["inspection_type"].str.contains("Initial", na=False) &
    clean["score"].notna() &
    clean["year"].between(2022, 2025)
].copy()

model_base = (
    initial_insp.groupby(["restaurant_id","inspection_date"])
    .agg(
        score=("score","first"),
        borough=("borough","first"),
        hypercategory=("hypercategory","first"),
        MdEWrkE=("MdEWrkE","first"),
        Pop_1E=("Pop_1E","first"),
        year=("year","first")
    )
    .reset_index()
    .dropna(subset=["borough","hypercategory","MdEWrkE","Pop_1E"])
)
model_base["grade_binary"] = (model_base["score"] <= 13).astype(int)
model_base = model_base[~model_base["hypercategory"].isin(["Mixed/Uncategorized"])]

model_enc = pd.get_dummies(
    model_base[["grade_binary","borough","hypercategory","MdEWrkE","Pop_1E","year"]],
    columns=["borough","hypercategory"],
    drop_first=True
)

X = model_enc.drop(columns=["grade_binary"])
y = model_enc["grade_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# baseline: predict majority class
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
baseline_auc = 0.5
print(f"  Baseline (majority class): acc={baseline_acc:.3f}, AUC={baseline_auc:.3f}")

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                      ("model", LogisticRegression(max_iter=2000, random_state=42))]),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "acc":  accuracy_score(y_test, y_pred),
        "auc":  roc_auc_score(y_test, y_prob),
        "fpr":  roc_curve(y_test, y_prob)[0],
        "tpr":  roc_curve(y_test, y_prob)[1],
        "cm":   confusion_matrix(y_test, y_pred),
    }
    print(f"  {name}: AUC={results[name]['auc']:.4f}, Acc={results[name]['acc']:.4f}")

# ROC comparison plot
palette_mod = {
    "Logistic Regression": "steelblue",
    "Random Forest":       "seagreen",
    "Gradient Boosting":   "firebrick",
}
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, res in results.items():
    ax.plot(res["fpr"], res["tpr"], linewidth=2, color=palette_mod[name],
            label=f"{name} (AUC = {res['auc']:.3f})")
ax.plot([0,1],[0,1], linestyle="--", color="gray", label=f"Baseline (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves: Pre-Inspection Predictive Models\n(Borough, Cuisine, Neighborhood Variables)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig_model_comparison_roc.png", bbox_inches="tight")
plt.close()
print("  Saved fig_model_comparison_roc.png")

# confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred: Not-A","Pred: A"],
                yticklabels=["True: Not-A","True: A"])
    ax.set_title(f"{name}\nAUC={res['auc']:.3f}, Acc={res['acc']:.3f}")
plt.suptitle("Confusion Matrices: Pre-Inspection Models", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  Saved fig_confusion_matrices.png")

# RF feature importance
rf_model = models["Random Forest"]
importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(importances["feature"][::-1], importances["importance"][::-1], color="seagreen", alpha=0.8)
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
ax.set_title("Random Forest Feature Importances\n(Top 15, Pre-Inspection Features)")
plt.tight_layout()
plt.savefig(FIG_DIR + "fig_rf_importance.png", bbox_inches="tight")
plt.close()
print("  Saved fig_rf_importance.png")

print("\nAll done.")
print("\n--- Summary for writeup ---")
print(f"McCrary excess at 12-13: {excess:,.0f} ({excess_pct:.0f}%), 95% CI [{boot_ci_lo:.0f}, {boot_ci_hi:.0f}]")
print(f"K-W borough test: H={kw_boro_stat:.2f}, p={kw_boro_p:.4f}")
print(f"Restaurant threshold clustering: mean={rest_multi['pct_at_boundary'].mean():.1%}, null={null_rate:.1%}, p={t_p:.4f}")
for name, res in results.items():
    print(f"{name}: AUC={res['auc']:.3f}, Acc={res['acc']:.3f}")


# =============================================================================
# R SECTION: group_analysis.Rmd
# Two-way balanced ANOVA (hypercategory x borough), block design sampling,
# initial exploratory regression. Run via RStudio or: rmarkdown::render().
# Paste the content below into group_analysis.Rmd to reproduce.
# =============================================================================

"""
---
title: "group_analysis"
output: github_document
---


```{r, include=FALSE}
library(dplyr); library(tidyr); library(arm) ; library(ggplot2)
df <- read.csv('./inspection_results_with_ntas.csv')
```

# Introduction 
When choosing to dine out and explore the city's vibrant restaurants, many people take into consideration a variety of factors. For some, they want to eat something they aren't able to cook at home. For others, they may want a more upscale experience. However, most people would consider food safety. 

Our team was inspired by an "I Quant NY" [article](https://iquantny.tumblr.com/post/76928412519/think-nyc-restaurant-grading-is-flawed-heres). In it, Ben Wellington points out an unnatural "peak" around a score of 13, which is at the cutoff for an "A" grade. 

![From "Think NYC Restaurant Grading is Flawed? Here’s the Proof," I Quant NY](https://64.media.tumblr.com/cf186f94253dac86f1f642e12f6cc570/tumblr_inline_pagvaoOj731szvr4h_500.jpg)

Wellington's reasoning for the unnatural peak was the subjective nature of health inspection grading. If a restaurant were to score in the 12-15 points bucket, many health inspector may simply just assign a letter grade of A, as it is beneficial for the local economy. 

However, the data above is from 2010 - 2014, and more data is available now. 

```{r,echo = FALSE}
histo <- df %>%
  group_by(score) %>%
  summarise(count = n()) %>%
  filter(score <= 50)

# Assign colors based on score ranges
bar_colors <- ifelse(histo$score <= 13, "darkgreen",
              ifelse(histo$score <= 27, "blue", "red"))

# Plot
barplot(
  height = histo$count,
  names.arg = histo$score,
  col = bar_colors,
  border = "black",
  xlab = "Score",
  ylab = "Count",
  main = "NYC Restaurant Inspection Results, up to 2026",
  space = 0        # no gaps between bars, like a histogram
)

```
So, indeed, we notice a massive peak at the 12 and 13 score range, with a smaller, still noticeable peak at the 28 score (the cutoff for a B letter grade). Focusing on the A-grade bump, Wellington assumed that if "all were fair," there should be a fairly flat count between 12 and 16. Adopting the same methodology, that means we would expect to see around 4000 inspections in each score bucket, assuming a "fair" distribution, represented by the dotted black line. 

```{r, echo = FALSE}
library(dplyr)
reference_region <- histo %>%
  filter(score >= 12 & score <= 16)

baseline <- mean(reference_region$count)

suspicious_zone <- histo %>%
  filter(score >= 14 & score <= 16)  

expected_total <- baseline * nrow(suspicious_zone)
actual_total   <- sum(suspicious_zone$count)
deficit        <- expected_total - actual_total
pct_lower      <- (deficit / expected_total) * 100


bar_colors <- ifelse(histo$score <= 13, "darkgreen",
              ifelse(histo$score <= 27, "blue", "red"))

barplot(
  height = histo$count,
  names.arg = histo$score,
  col = bar_colors,
  border = "black",
  space = 0,
  xlab = "Score",
  ylab = "Count",
  main = "Restaurant Inspection Scores with Baseline"
)

segments(
  x0 = 12, x1 = 16,
  y0 = baseline, y1 = baseline,
  col = "black", lty = 2, lwd = 2
)

```
So, in response to Wellington's post, the New York Health Department has said that "inspectors are not instructed to offer leniency, just to cite what they see." But from a more recent, graphical analysis, we see that the trend observed by Wellington is still present. Thus, we have three main research questions: 

1. Is there a relationship between restaurants, boroughs and health inspection scores? Namely, if two restaurants of the same cuisines are in two different boroughs, would we expect them to receive the same health inspection score? 
2. If there is a relationship, are we able to accurately predict a restaurant's health rating through its borough and cuisine type, among other predictors like median income, population, etc. 
3. For the violation codes themselves, are certain violation codes associated with boroughs? For example, are restaurants in Queens more likely to receive a violation of a certain type over another? 


# Data Collection and Data Description

We primarily used two datasets, both provided through NYC's OpenData program. The [first](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data) dataset is of the health inspections themselves. The dataset has a variety of features, but we were mainly concern with: 

1. Borough
2. Cuisine 
3. Score
4. Date of inspection 
5. Violation Code

Then, because the Violation Codes by themselves are not descriptive, we referred to an [additional data set](https://github.com/nychealth/Food-Safety-Health-Code-Reference/blob/main/Violation-Health-Code-Mapping.csv) that mapped a violation code to a more detailed category. 

Upon an initial regression, we found that having too many categories made our regression coefficients hard to interpret, so we decided to group cuisines of similar types. For example, Japanese, Chinese, Japanese/Chinese Fusion, and Korean would be grouped under "East Asian Cuisine"

We called these groupings "hypercategories," of which there are ten types:

1. American
2. East Asian
3. South & Southeast Asian 
4. Latin American
5. European 
6. Mediterranean 
7. Middle Eastern 
8. African 
9. Specific Dishes (Restaurants that only do Sandwiches, Bagels, Crepes, etc.)
10. Dietary Specific (Vegan, Gluten-Free, Vegetarian restaurants)

```{r, include = FALSE}

hypercategory <- c(
  "American"                       = "American",
  "New American"                   = "American",
  "Californian"                    = "American",
  "Steakhouse"                     = "American",
  "Barbecue"                       = "American",
  "Hamburgers"                     = "American",
  "Soul Food"                      = "American",
  "Cajun"                          = "American",
  "Creole"                         = "American",
  "Creole/Cajun"                   = "American",
  "Tex-Mex"                        = "American",
  "Southwestern"                   = "American",
 
  "Chinese"                        = "East Asian",
  "Japanese"                       = "East Asian",
  "Korean"                         = "East Asian",
  "Chinese/Japanese"               = "East Asian",
  "Chinese/Cuban"                  = "East Asian",
 
  "Indian"                         = "South & Southeast Asian",
  "Thai"                           = "South & Southeast Asian",
  "Southeast Asian"                = "South & Southeast Asian",
  "Bangladeshi"                    = "South & Southeast Asian",
  "Pakistani"                      = "South & Southeast Asian",
  "Filipino"                       = "South & Southeast Asian",
  "Indonesian"                     = "South & Southeast Asian",
  "Afghan"                         = "South & Southeast Asian",
  "Asian/Asian Fusion"             = "South & Southeast Asian",
 
  "Mexican"                        = "Latin American",
  "Latin American"                 = "Latin American",
  "Caribbean"                      = "Latin American",
  "Peruvian"                       = "Latin American",
  "Brazilian"                      = "Latin American",
  "Chilean"                        = "Latin American",
  "Polynesian"                     = "Latin American",
  "Hawaiian"                       = "Latin American",
 
  "Italian"                        = "European",
  "French"                         = "European",
  "Spanish"                        = "European",
  "English"                        = "European",
  "German"                         = "European",
  "Portuguese"                     = "European",
  "Polish"                         = "European",
  "Russian"                        = "European",
  "Eastern European"               = "European",
  "Irish"                          = "European",
  "Scandinavian"                   = "European",
  "Czech"                          = "European",
  "Basque"                         = "European",
  "New French"                     = "European",
  "Continental"                    = "European",
  "Haute Cuisine"                  = "European",
  "Australian"                     = "European",
 
  "Mediterranean"                  = "Mediterranean",
  "Greek"                          = "Mediterranean",
  "Turkish"                        = "Mediterranean",
  "Lebanese"                       = "Mediterranean",
  "Moroccan"                       = "Mediterranean",
  "Tapas"                          = "Mediterranean",
 
  "Middle Eastern"                 = "Middle Eastern",
  "Egyptian"                       = "Middle Eastern",
  "Iranian"                        = "Middle Eastern",
  "Armenian"                       = "Middle Eastern",
  "Jewish/Kosher"                  = "Middle Eastern",
 
  "African"                        = "African",
  "Ethiopian"                      = "African",
 
  "Pizza"                          = "Dish-Type",
  "Sandwiches"                     = "Dish-Type",
  "Chicken"                        = "Dish-Type",
  "Seafood"                        = "Dish-Type",
  "Hotdogs"                        = "Dish-Type",
  "Hotdogs/Pretzels"               = "Dish-Type",
  "Bagels/Pretzels"                = "Dish-Type",
  "Pancakes/Waffles"               = "Dish-Type",
  "Salads"                         = "Dish-Type",
  "Soups"                          = "Dish-Type",
 
  "Coffee/Tea"                     = "Beverages & Sweets",
  "Juice, Smoothies, Fruit Salads" = "Beverages & Sweets",
  "Bottled Beverages"              = "Beverages & Sweets",
  "Donuts"                         = "Beverages & Sweets",
  "Frozen Desserts"                = "Beverages & Sweets",
  "Bakery Products/Desserts"       = "Beverages & Sweets",
  "Nuts/Confectionary"             = "Beverages & Sweets",
 
  "Vegan"                          = "Dietary Style",
  "Vegetarian"                     = "Dietary Style",
  "Fruits/Vegetables"              = "Dietary Style",
 
  "Fusion"                         = "Mixed/Uncategorized",
  "Sandwiches/Salads/Mixed Buffet" = "Mixed/Uncategorized",
  "Soups/Salads/Sandwiches"        = "Mixed/Uncategorized",
  "Other"                          = "Mixed/Uncategorized",
  "Not Listed/Not Applicable"      = "Mixed/Uncategorized",
  " "                               = "Mixed/Uncategorized"
)


df$hypercategory <- hypercategory[df$cuisine]

df <- df[!is.na(df$hypercategory), ]
df <- df[!is.na(df$score), ]

df <- df %>%
  filter(hypercategory != "Mixed/Uncategorized")

```

Our final dataframe resembles this:

```{r}
head(df %>% dplyr::select(score, borough, cuisine, hypercategory, violation_category))
```



## Question 1 

Now for an initial exploratory analysis with graphs. Important to keep in mind that lower scores are better. 

```{r, echo = FALSE}
df$inspection_date <- as.Date(df$inspection_date)
df$inspection_year <- lubridate::year(df$inspection_date)


ggplot(df, aes(x = inspection_year, y = score)) +
  geom_point()+
  labs(x = "Inspection Date", y = "Inspection Score")+
  geom_hline(yintercept  = 13, color = "green") + 
  geom_hline(yintercept = 27, color = "red")+
  theme_minimal()

```
Scores on or below the green line are restaurants who earned a `A` grade. Restaurants who scored above the green line and under the red line are those who scored a `B` grade. Any scores above the red line represent a `C` grade. 

It may help to also see how the score distribution by cuisine. For a clearer picture, I will take a random sample. 

```{r, echo=FALSE}
graph_df <- df %>% slice_sample(n = 1000)


ggplot(graph_df, aes(x = inspection_date, y = score, colour = hypercategory)) +
  geom_point()+
  labs(x = "Inspection Date", y = "Inspection Score")+
  geom_hline(yintercept  = 13, color = "green") + 
  geom_hline(yintercept = 27, color = "yellow")+
  theme_minimal() + 
  theme(
     axis.ticks.x = element_blank(),
     axis.text.x = element_blank()
  )
```

From a quick glance, there seems to be a variety of restaurant types in each grade bucket in our sample. Dividng each cuisine type specifically, 

```{r, echo = FALSE}

ggplot(df, aes(x = inspection_date, y = score, colour = borough)) +
  geom_point()+
  labs(x = NULL, y = NULL)+
  geom_hline(yintercept  = 13, color = "green") + 
  geom_hline(yintercept = 27, color = "yellow")+
  theme_minimal() + 
  theme(
     axis.ticks.x = element_blank(),
     axis.text.x = element_blank()  ) +
  facet_wrap(~ hypercategory)

```

Time for regression:

Across all boroughs, do the restaurant  of the same cuisine type have the same expected score? 
```{r}
fit1 <- lm(score ~ factor(borough) * factor(hypercategory),
           data = df)
```

Bronx is the baseline borough and African is the baseline hypercategory. 

```{r}
co <- coef(fit1)
se <- sqrt(diag(vcov(fit1)))

keep <- c("factor(borough)Brooklyn",
          "factor(borough)Manhattan",
          "factor(borough)Queens",
          "factor(borough)Staten Island")

arm::coefplot(co[keep], sds = se[keep],
              col.pts = "blue")
```



Going to ignore other coefficient plots for now, becuase I don't know how to work it. But, I think we can do a block design ANOVA? 

```{r}

sample_block_design <- function(df, n_per_cell) {
  cell_counts <- df %>%
    count(hypercategory, borough, name = "n")

  insufficient <- cell_counts %>%
    filter(n < n_per_cell)
  
  if (nrow(insufficient) > 0) {
    msg <- paste0(
      "Not enough observations in the following hypercategory × borough cells:\n",
      paste0(
        "  - (", insufficient$hypercategory, ", ", insufficient$borough, 
        "): n = ", insufficient$n,
        collapse = "\n"
      )
    )
    stop(msg)
  }
  
  df %>%
    group_by(hypercategory, borough) %>%
    slice_sample(n = n_per_cell, replace = FALSE) %>%
    ungroup()
}
```
Ok, so we should probably drop Staten Island and Dietary Style, since those are the two most sparsely populated categories.

```{r, include = FALSE}
df <- df %>%
  filter(borough != "Staten Island") %>%
  filter(hypercategory != "Dietary Style")


sampled_df <- sample_block_design(df, n_per_cell = 30)
display_anova_table <- function(df) {
  with(df, table(hypercategory, borough))
}
```


```{r}
display_anova_table(df)

anova_model <- aov(score ~ hypercategory * borough, data = sampled_df)
```

```{r}
summary(anova_model)

```"""
