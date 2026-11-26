import pandas as pd

# Load the original dataset
df = pd.read_csv("../Data/Palo Alto Networks.csv")

# Basic dataset inspection
print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

career_cols = [
    "JobLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

print(df[career_cols].describe())
