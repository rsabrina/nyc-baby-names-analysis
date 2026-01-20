import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv("data/Popular_Baby_Names.csv")

# -----------------------------
# Select Relevant Columns
# -----------------------------
df = data[["Year of Birth", "Gender", "Count", "Rank"]].dropna()

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
print("Summary Statistics:")
print(df.describe())

# -----------------------------
# Correlation Analysis
# -----------------------------
corr_count_rank = df["Count"].corr(df["Rank"])
corr_year_count = df["Year of Birth"].corr(df["Count"])
corr_year_rank = df["Year of Birth"].corr(df["Rank"])

print("\nCorrelation Results:")
print(f"Count vs Rank: {corr_count_rank:.3f}")
print(f"Year vs Count: {corr_year_count:.3f}")
print(f"Year vs Rank: {corr_year_rank:.3f}")

# -----------------------------
# Visualizations
# -----------------------------
plt.figure()
plt.hist(df["Count"], bins=30)
plt.title("Distribution of Baby Name Counts")
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(df["Rank"], bins=30)
plt.title("Distribution of Baby Name Ranks")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.show()

gender_counts = df["Gender"].value_counts()

plt.figure()
gender_counts.plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Records")
plt.show()

plt.figure()
gender_counts.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Gender Proportion")
plt.show()

# -----------------------------
# Regression Analysis
# -----------------------------
X = df["Count"]
y = df["Rank"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print("\nRegression Summary:")
print(model.summary())

# -----------------------------
# Extended Analysis:
# Gender / Ethnicity / Year Effects
# -----------------------------

data = pd.read_csv("data/Popular_Baby_Names.csv")
cols = [c for c in ["Year of Birth", "Gender", "Ethnicity", "Count", "Rank"] if c in data.columns]
df = data[cols].dropna()

for c in ["Year of Birth", "Count", "Rank"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[c for c in ["Year of Birth", "Count", "Rank"] if c in df.columns])

if "Gender" in df.columns:
    corr_by_gender = df.groupby("Gender")[["Count", "Rank"]].corr().iloc[0::2, -1]
    print("\nCount vs Rank correlation by Gender:")
    print(corr_by_gender)

if "Ethnicity" in df.columns:
    print("\nTop 10 Ethnicities by record count:")
    print(df["Ethnicity"].value_counts().head(10))

X_cols = [c for c in ["Count", "Year of Birth", "Gender", "Ethnicity"] if c in df.columns]
X = pd.get_dummies(df[X_cols], drop_first=True)
X = X.apply(pd.to_numeric, errors="coerce").astype(float)
X = sm.add_constant(X)
y = pd.to_numeric(df["Rank"], errors="coerce").astype(float)

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]


model_multi = sm.OLS(y, X).fit()
print("\nMultiple Regression (Rank ~ Count + Year + Gender + Ethnicity) Summary:")
print(model_multi.summary())
