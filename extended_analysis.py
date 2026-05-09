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
