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
