# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Telco churn: exploratory analysis
#
# How does the churn rate vary with contract, tenure, price and services? Every rate comes with a
# 95% Wilson interval, because some groups are small.
#
# Data: IBM Telco Customer Churn sample dataset (Kaggle: blastchar/telco-customer-churn). The company is
# fictional and the data's license is unclear (see `docs/data_source.md`), so this notebook shows only
# aggregate statistics, never customer rows. `churn` = 1 means the customer left within the last month.
# Differences between groups are associations, not causes.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.figsize": (9, 4.5), "axes.grid": True, "grid.alpha": 0.3})

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda_utils import rate_table, wilson_interval  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "telco_clean.parquet"
IMG_DIR = ROOT / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(IMG_DIR / name, dpi=150, bbox_inches="tight")


def show(table: pd.DataFrame) -> None:
    out = table.drop(columns=["churn_rate", "ci_low", "ci_high"]).copy()
    out["churn_rate_pct"] = (100 * table["churn_rate"]).round(1)
    out["ci_low_pct"] = (100 * table["ci_low"]).round(1)
    out["ci_high_pct"] = (100 * table["ci_high"]).round(1)
    print(out.to_string())


def error_bars(table: pd.DataFrame):
    rates = 100 * table["churn_rate"].to_numpy()
    err = np.vstack([rates - 100 * table["ci_low"].to_numpy(), 100 * table["ci_high"].to_numpy() - rates])
    return rates, err


# %%
df = pd.read_parquet(DATA_PATH)
overall = float(df["churn"].mean())
low, high = wilson_interval(int(df["churn"].sum()), len(df))

# The notebook must agree with the audit (docs/raw_data_audit.txt).
assert len(df) == 7_043
assert int(df["churn"].sum()) == 1_869
contracts = df["contract"].value_counts()
assert contracts["Month-to-month"] == 3_875 and contracts["One year"] == 1_473 and contracts["Two year"] == 1_695
assert int((df["internet_service"] == "No").sum()) == 1_526

print(
    f"customers: {len(df):,} | churned: {int(df['churn'].sum()):,} | churn rate: {100 * overall:.2f}% "
    f"(95% interval {100 * float(low):.1f}% to {100 * float(high):.1f}%)"
)


# %% [markdown]
# ## 1. Plan and billing: contract, internet service, payment method, paperless billing

# %%
def plot_rates(ax, table: pd.DataFrame, title: str) -> None:
    labels = [f"{idx} (n={n:,})" for idx, n in zip(table.index, table["customers"])]
    rates, err = error_bars(table)
    ax.barh(labels, rates, xerr=err, capsize=3)
    ax.axvline(100 * overall, color="red", linestyle="--", label=f"overall {100 * overall:.1f}%")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("churn rate (%), 95% interval")
    ax.legend(loc="lower right")


features = ["contract", "internet_service", "payment_method", "paperless_billing"]
tables = {col: rate_table(df, col).sort_values("churn_rate", ascending=False) for col in features}

fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for ax, (col, table) in zip(axes.ravel(), tables.items()):
    assert table["customers"].sum() == len(df)
    print(f"\n{col}")
    show(table)
    plot_rates(ax, table, col)
save(fig, "01_churn_by_plan_and_billing.png")
plt.show()

# %% [markdown]
# ## 2. Tenure
#
# Bucket "0" holds the 11 customers with tenure 0 (their `total_charges` was blank in the raw file).

# %%
bins = [-1, 0, 6, 12, 24, 48, 72]
labels = ["0", "1-6", "7-12", "13-24", "25-48", "49-72"]
eda = df.assign(tenure_bucket=pd.cut(df["tenure"], bins=bins, labels=labels))

tenure_table = rate_table(eda, "tenure_bucket")
assert tenure_table["customers"].sum() == len(df)
show(tenure_table)

rates, err = error_bars(tenure_table)
fig, ax = plt.subplots()
ax.bar(range(len(tenure_table)), rates, yerr=err, capsize=3)
ax.axhline(100 * overall, color="red", linestyle="--", label=f"overall {100 * overall:.1f}%")
ax.set_xticks(range(len(tenure_table)))
ax.set_xticklabels([f"{i}\n(n={n:,})" for i, n in zip(tenure_table.index.astype(str), tenure_table["customers"])])
ax.set_xlabel("tenure (months, inferred unit)")
ax.set_ylabel("churn rate (%), 95% interval")
ax.set_title("Churn rate by tenure")
ax.legend()
save(fig, "02_churn_by_tenure.png")
plt.show()

# %% [markdown]
# Contract type and tenure are tangled together, because month-to-month customers tend to have shorter
# tenure. This table shows the churn rate (%) and customers for each combination.

# %%
by_contract = eda.groupby(["tenure_bucket", "contract"], observed=True)["churn"].agg(
    customers="size", churn_rate_pct="mean"
)
by_contract["churn_rate_pct"] = (100 * by_contract["churn_rate_pct"]).round(1)
print(by_contract.unstack("contract").to_string())

# %% [markdown]
# ## 3. Monthly charges (five equal-sized groups)

# %%
quintile = pd.qcut(df["monthly_charges"], 5, labels=False, duplicates="drop")
eda = df.assign(charge_quintile=quintile)
ranges = eda.groupby("charge_quintile")["monthly_charges"].agg(lowest="min", highest="max")
charge_table = rate_table(eda, "charge_quintile").join(ranges)
assert charge_table["customers"].sum() == len(df)
charge_table.index = [
    f"Q{int(i) + 1}: {lo:.0f}-{hi:.0f}"
    for i, lo, hi in zip(charge_table.index, charge_table["lowest"], charge_table["highest"])
]
show(charge_table.drop(columns=["lowest", "highest"]))

rates, err = error_bars(charge_table)
fig, ax = plt.subplots()
ax.bar(range(len(charge_table)), rates, yerr=err, capsize=3)
ax.axhline(100 * overall, color="red", linestyle="--", label=f"overall {100 * overall:.1f}%")
ax.set_xticks(range(len(charge_table)))
ax.set_xticklabels(charge_table.index)
ax.set_xlabel("monthly charges group (lowest to highest)")
ax.set_ylabel("churn rate (%), 95% interval")
ax.set_title("Churn rate by monthly charges")
ax.legend()
save(fig, "03_churn_by_monthly_charges.png")
plt.show()

print("\nmedian monthly charges by internet service (price and service type are linked):")
print(df.groupby("internet_service")["monthly_charges"].median().round(2).to_string())

# %% [markdown]
# ## 4. Add-on services, among customers with internet service only
#
# Customers without internet cannot have these add-ons, so including them would mix two different groups.

# %%
internet = df[df["internet_service"] != "No"]
assert len(internet) == 7_043 - 1_526
addons = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]

rows = []
for col in addons:
    for level in ["Yes", "No"]:
        sub = internet[internet[col] == level]
        k, n = int(sub["churn"].sum()), len(sub)
        lo, hi = wilson_interval(k, n)
        rows.append({
            "service": col, "has_service": level, "customers": n, "churned": k,
            "churn_rate": k / n, "ci_low": float(lo), "ci_high": float(hi),
        })
addon_table = pd.DataFrame(rows)
assert (addon_table.groupby("service")["customers"].sum() == len(internet)).all()

printable = addon_table.copy()
for c in ["churn_rate", "ci_low", "ci_high"]:
    printable[c] = (100 * printable[c]).round(1)
print(printable.to_string(index=False))

x = np.arange(len(addons))
width = 0.38
fig, ax = plt.subplots(figsize=(10, 5))
for i, level in enumerate(["Yes", "No"]):
    sub = addon_table[addon_table["has_service"] == level].set_index("service").loc[addons]
    rates, err = error_bars(sub)
    ax.bar(x + (i - 0.5) * width, rates, width, yerr=err, capsize=3, label=f"has service: {level}")
ax.set_xticks(x)
ax.set_xticklabels(addons, rotation=20, ha="right")
ax.set_ylabel("churn rate (%), 95% interval")
ax.set_title("Churn rate by add-on service (internet customers only)")
ax.legend()
save(fig, "04_churn_by_addon_services.png")
plt.show()

# %% [markdown]
# ## 5. How related are tenure and charges?

# %%
print(df[["tenure", "monthly_charges", "total_charges"]].corr().round(3).to_string())