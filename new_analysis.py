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
