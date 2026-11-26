from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score


# Project folder paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_DIR
OUTPUT_DIR = PROJECT_DIR / "Output"

INPUT_FILE = DATA_DIR / "Palo Alto Networks.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Required dataset columns

REQUIRED_COLUMNS = [
    "Department",
    "JobRole",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "TrainingTimesLastYear",
    "YearsWithCurrManager",
    "TotalWorkingYears",
    "Attrition",
    "JobSatisfaction",
    "JobInvolvement",
]


# ==========================================
# Load and Validate the Dataset
# ==========================================

def load_data():
    # Check whether the dataset file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    # Read the CSV file into a Pandas DataFrame

    df = pd.read_csv(INPUT_FILE)

    # Check whether all required columns are available

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    # Stop execution if any required columns are missing

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Display dataset loading information

    print("Dataset loaded successfully!")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # Return the validated dataset
    return df
   

# ==========================================
# Feature Engineering - Career Metrics
# ==========================================

def create_career_metrics(df):
    """
    Calculate career-related metrics and identify new joiners.
    """

    # Identify employees who have just joined
    df["IsNewJoiner"] = (
        df["YearsAtCompany"] == 0
    ).astype(int)

    # Replace zero tenure with NaN to avoid division by zero

    tenure = df["YearsAtCompany"].replace(0, np.nan)

    # Calculate Promotion Gap Ratio
    # Formula: YearsSinceLastPromotion / YearsAtCompany

    df["PromotionGapRatio"] = (
        df["YearsSinceLastPromotion"] / tenure
    ).fillna(0)

    # Calculate Role Stagnation Index
    # Formula: YearsInCurrentRole / YearsAtCompany

    df["RoleStagnationIndex"] = (
        df["YearsInCurrentRole"] / tenure
    ).fillna(0)

    # Calculate Training Intensity Score
    # Use a minimum denominator of 1 for new joiners

    df["TrainingIntensityScore"] = (
        df["TrainingTimesLastYear"]
        / df["YearsAtCompany"].clip(lower=1)
    )

    # Calculate Manager Stability Indicator
    # Project-defined proxy: manager tenure / company tenure

    df["ManagerStabilityIndicator"] = (
        df["YearsWithCurrManager"] / tenure
    ).fillna(0)

    # Display the first five rows of the calculated metrics

    metric_columns = [
        "IsNewJoiner",
        "PromotionGapRatio",
        "RoleStagnationIndex",
        "TrainingIntensityScore",
        "ManagerStabilityIndicator",
    ]

    print("\nCareer metrics created successfully!")
    print(df[metric_columns].head())

    # Return the updated dataset
    return df

# ==========================================
# Validate Career Metrics
# ==========================================

def validate_career_metrics(df):
    cols = [
        "PromotionGapRatio",
        "RoleStagnationIndex",
        "TrainingIntensityScore",
        "ManagerStabilityIndicator"
    ]

    print("\nFeature Engineering Validation")
    print("Records:", len(df))
    print("New joiners:", df["IsNewJoiner"].sum())

    print("\nMissing values:")
    print(df[cols].isna().sum())

    print("\nCareer metrics summary:")
    print(df[cols].describe().round(2).to_string())


# ==========================================
# Data Preprocessing
# ==========================================

CAREER_FEATURES = [
    "PromotionGapRatio",
    "RoleStagnationIndex",
    "TrainingIntensityScore",
    "ManagerStabilityIndicator",
]

NORM_FEATURES = [col + "_Norm" for col in CAREER_FEATURES]


def preprocess_data(df):
    """
    Normalize career features, encode categorical columns,
    and identify extreme late-career outliers.
    """

    # Identify extreme late-career outliers using IQR

    q1 = df["TotalWorkingYears"].quantile(0.25)
    q3 = df["TotalWorkingYears"].quantile(0.75)
    upper_limit = q3 + 1.5 * (q3 - q1)

    is_outlier = df["TotalWorkingYears"] > upper_limit

    print("\nOutlier limit:", upper_limit)
    print("Outliers identified:", is_outlier.sum())
    
    print("\nOutlier TotalWorkingYears summary:")
    print(df.loc[is_outlier, ["Age", "TotalWorkingYears", "YearsAtCompany"]].describe())    
    

    # Save outlier records separately for review

    df[is_outlier].to_csv(
        OUTPUT_DIR / "outliers_removed.csv",
        index=False
    )

    # Exclude identified outliers from the processed copy

    df = df[~is_outlier].copy()

    # Normalize career features into new columns

    scaler = MinMaxScaler()
    df[NORM_FEATURES] = scaler.fit_transform(
        df[CAREER_FEATURES]
    )

    # Encode categorical columns in a separate copy

    df_model = pd.get_dummies(
        df,
        columns=["Department", "JobRole"],
        dtype=int
    )

    print("Rows after preprocessing:", len(df))
    print("Encoded dataset shape:", df_model.shape)

    print("\nNormalized feature ranges:")
    print(
        df[NORM_FEATURES]
        .agg(["min", "max"])
        .round(2)
        .to_string()
    )

    return df, df_model

# Run the dataset loading and career metrics workflow
if __name__ == "__main__":
    df = load_data()
    df = create_career_metrics(df)
    validate_career_metrics(df)
    df, df_model = preprocess_data(df)

 # Prepare normalized career features for clustering

    clustering_features = [
        "PromotionGapRatio_Norm",
        "RoleStagnationIndex_Norm",
        "TrainingIntensityScore_Norm",
        "ManagerStabilityIndicator_Norm"
    ]

    X = df[clustering_features].copy()

    print("\nClustering data shape:", X.shape)

# Evaluate the optimal number of clusters

    print("\nK-Means cluster evaluation:")

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)

        print(f"K = {k}, Silhouette Score = {score:.3f}")

    # Apply K-Means with five career clusters

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["CareerCluster"] = kmeans.fit_predict(X)

    print("\nK-Means Cluster Sizes:")
    print(df["CareerCluster"].value_counts().sort_index())        

 # Apply Hierarchical Clustering for validation

    hierarchical = AgglomerativeClustering(n_clusters=5)

    df["HierarchicalCluster"] = hierarchical.fit_predict(X)

    print("\nHierarchical Cluster Sizes:")
    print(df["HierarchicalCluster"].value_counts().sort_index())

    # Compare K-Means and Hierarchical clustering

    ari_score = adjusted_rand_score(
        df["CareerCluster"],
        df["HierarchicalCluster"]
    )

    print(f"\nAdjusted Rand Index (ARI): {ari_score:.3f}")

 # Analyze career profiles for each K-Means cluster
 
    profile_columns = [
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "TrainingTimesLastYear",
        "YearsWithCurrManager"
    ]

    print("\nCareer Profile Summary by Cluster:")
    print(
        df.groupby("CareerCluster")[profile_columns]
        .mean()
        .round(2)
        .to_string()
    )

    label_profile = df.groupby("CareerCluster")[
            [
                "YearsAtCompany",
                "YearsSinceLastPromotion",
                "YearsWithCurrManager",
                "PromotionGapRatio",
                "RoleStagnationIndex",
            ]
        ].mean()

    cluster_labels = {}
    remaining = list(label_profile.index)

    def assign(cluster_id, label):
            cluster_labels[cluster_id] = label
            remaining.remove(cluster_id)

    assign(
            label_profile.loc[remaining, "YearsAtCompany"].idxmin(),
            "Early-career explorers"
        )
    assign(
            label_profile.loc[remaining, "YearsSinceLastPromotion"].idxmax(),
            "Promotion-stalled employees"
        )
    assign(
            label_profile.loc[remaining, "YearsWithCurrManager"].idxmin(),
            "High-risk stagnation profiles"
        )
    assign(
            (
                label_profile.loc[remaining, "PromotionGapRatio"]
                + label_profile.loc[remaining, "RoleStagnationIndex"]
            ).idxmin(),
            "Fast-track performers"
        )
    assign(remaining[0], "Stable long-term contributors")

    df["CareerClusterLabel"] = df["CareerCluster"].map(cluster_labels)

    print("\nCareer Profile Counts:")
    print(df["CareerClusterLabel"].value_counts())    

    
    # ==========================================
    # Promotion Gap Risk and Career Stagnation Risk
    # ==========================================

    df["PromotionGapScore"] = pd.cut(
        df["YearsSinceLastPromotion"],
        bins=[-1, 2, 5, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    print("\nPromotion Gap Score:")
    print(df["PromotionGapScore"].value_counts().sort_index())

    def stagnation_risk(row):
        if row["IsNewJoiner"] == 1:
            return "Low"

        if (
            row["PromotionGapScore"] == "High"
            or row["CareerClusterLabel"] == "Promotion-stalled employees"
        ):
            return "High"

        if (
            row["PromotionGapScore"] == "Medium"
            and row["YearsInCurrentRole"] >= 5
        ):
            return "High"

        if row["CareerClusterLabel"] == "High-risk stagnation profiles":
            return (
                "High"
                if row["YearsInCurrentRole"] >= 5
                else "Medium"
            )

        if (
            row["PromotionGapScore"] == "Medium"
            or row["YearsInCurrentRole"] >= 5
        ):
            return "Medium"

        return "Low"

    df["CareerStagnationRisk"] = df.apply(
        stagnation_risk,
        axis=1
    )

    print("\nCareer Stagnation Risk:")
    print(df["CareerStagnationRisk"].value_counts())

    print("\nCareer Stagnation Risk by Cluster:")
    print(
        pd.crosstab(
            df["CareerClusterLabel"],
            df["CareerStagnationRisk"]
        ).to_string()
    )


    # ==========================================
    # Training Need Indicator
    # ==========================================

    df["TrainingNeedIndicator"] = pd.cut(
        df["TrainingTimesLastYear"],
        bins=[-1, 1, 2, float("inf")],
        labels=["High", "Medium", "Low"]
    )

    print("\nTraining Need Indicator:")
    print(df["TrainingNeedIndicator"].value_counts())

    # ==========================================
    # Manager Stability Impact
    # ==========================================

    df["ManagerStabilityImpact"] = pd.cut(
        df["ManagerStabilityIndicator"],
        bins=[-0.01, 0.3, 0.7, float("inf")],
        labels=["Unstable", "Moderate", "Stable"]
    ).astype(object)

    df.loc[
        df["IsNewJoiner"] == 1,
        "ManagerStabilityImpact"
    ] = "New joiner"

    print("\nManager Stability Impact:")
    print(df["ManagerStabilityImpact"].value_counts())


    # ==========================================
    # Retention Opportunity Identification
    # Identifies engaged employees who have not
    # left the company but show career stagnation.
    # ==========================================

    # Eligibility: employee has not left and is
    # not classified as disengaged

    df["RetentionEligible"] = (
        (df["Attrition"] == 0)
        & (df["JobSatisfaction"] >= 3)
        & (df["JobInvolvement"] >= 3)
    )

    # Retention Opportunity Identification

    

    df["RetentionOpportunityIndex"] = np.select(
        [
            df["RetentionEligible"]
            & (df["CareerStagnationRisk"] == "High"),

            df["RetentionEligible"]
            & (df["CareerStagnationRisk"] == "Medium")
        ],
        [2, 1],
        default=0
    )

    df["RetentionOpportunity"] = df["RetentionOpportunityIndex"].map({
        2: "High priority",
        1: "Medium priority",
        0: "No immediate opportunity"
    })

    print("\nRetention Opportunity Distribution:")
    print(df["RetentionOpportunity"].value_counts())

    print("\nRetention Eligibility:")
    print(df["RetentionEligible"].value_counts())
   
    # =================================
    # Suggested Action
    # =================================

    def suggested_action(row):
        # No action if there is no retention opportunity
        if row["RetentionOpportunityIndex"] == 0:
            return "No action"

        # Promotion review for high promotion gap
        if row["PromotionGapScore"] == "High":
            return "Promotion review"

        # Job rotation for employees in the same role for 5+ years
        if row["YearsInCurrentRole"] >= 5:
            return "Job rotation"

        # Training plan for high training needs
        if row["TrainingNeedIndicator"] == "High":
            return "Training plan"

        # Default recommendation
        return "Career conversation"

    df["SuggestedAction"] = df.apply(suggested_action, axis=1)

    print("\nSuggested Action:")
    print(df["SuggestedAction"].value_counts())

    # ==========================================
    # Export Files for Power BI
    # ==========================================

    df.insert(0, "EmployeeID", df.index + 1)

    cluster_profile = df.groupby("CareerClusterLabel").agg(
        Employees=("EmployeeID", "count"),
        Avg_YearsAtCompany=("YearsAtCompany", "mean"),
        Avg_YearsInCurrentRole=("YearsInCurrentRole", "mean"),
        Avg_YearsSinceLastPromotion=("YearsSinceLastPromotion", "mean"),
        Avg_TrainingTimesLastYear=("TrainingTimesLastYear", "mean"),
        Avg_YearsWithCurrManager=("YearsWithCurrManager", "mean"),
    ).round(2).reset_index()

    df_model = pd.get_dummies(
        df, columns=["Department", "JobRole"], dtype=int
    )

    df.to_csv(OUTPUT_DIR / "career_output.csv", index=False)
    cluster_profile.to_csv(OUTPUT_DIR / "cluster_profile.csv", index=False)
    df_model.to_csv(OUTPUT_DIR / "career_model_encoded.csv", index=False)

    print("\nExported to:", OUTPUT_DIR)
    print("career_output.csv:", df.shape)
    print("cluster_profile.csv:", cluster_profile.shape)
    print("career_model_encoded.csv:", df_model.shape)
    
     


