import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def prepare_ecommerce_data(ecom_path, ip_map_path, output_dir="data/processed"):
    df = pd.read_csv(ecom_path)
    ip_map = pd.read_csv(ip_map_path)

    # Parse timestamps and engineer temporal features
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    # Drop invalid rows
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Convert IP addresses to integers
    df["ip_address"] = df["ip_address"].astype(str).str.strip()
    df["ip_int"] = pd.to_numeric(df["ip_address"], errors="coerce")
    df.dropna(subset=["ip_int"], inplace=True)
    df["ip_int"] = df["ip_int"].astype(int)

    ip_map["lower_bound_ip_address"] = pd.to_numeric(ip_map["lower_bound_ip_address"], errors="coerce").astype("Int64")
    ip_map["upper_bound_ip_address"] = pd.to_numeric(ip_map["upper_bound_ip_address"], errors="coerce").astype("Int64")

    def map_country(ip):
        match = ip_map[
            (ip_map["lower_bound_ip_address"] <= ip) &
            (ip_map["upper_bound_ip_address"] >= ip)
        ]
        return match["country"].values[0] if not match.empty else "Unknown"

    df["country"] = df["ip_int"].apply(map_country)

    # 👉 Transaction frequency & velocity
    df.sort_values(["user_id", "purchase_time"], inplace=True)
    df["user_transaction_count"] = df.groupby("user_id")["user_id"].transform("count")
    df["device_transaction_count"] = df.groupby("device_id")["device_id"].transform("count")

    df["prev_transaction_time"] = df.groupby("user_id")["purchase_time"].shift(1)
    df["time_since_last_txn"] = (df["purchase_time"] - df["prev_transaction_time"]).dt.total_seconds()

    if df["time_since_last_txn"].isna().all():
        df["time_since_last_txn"] = 0
    else:
        fallback_value = df["time_since_last_txn"].median(skipna=True)
        df["time_since_last_txn"] = df["time_since_last_txn"].fillna(fallback_value)

    # Encode categorical features
    df = pd.get_dummies(df, columns=["source", "browser", "sex", "country"], drop_first=True)

    # Drop unused and high-cardinality columns
    df.drop(columns=[
        "signup_time", "purchase_time", "ip_address", "ip_int",
        "user_id", "device_id", "prev_transaction_time"
    ], inplace=True)

    df.to_csv(f"{output_dir}/ecom_full_cleaned.csv", index=False)
    return df


def prepare_bank_data(bank_path, output_dir="data/processed"):
    df = pd.read_csv(bank_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
    df.to_csv(f"{output_dir}/bank_full_cleaned.csv", index=False)
    return df


def split_and_balance(df, target_col, prefix, output_dir="data/processed"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove non-numeric features before SMOTE
    X = X.drop(columns=X.select_dtypes(exclude=[np.number]).columns.tolist())

    # Final check for NaNs
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Save splits
    X_train_bal.to_csv(f"{output_dir}/{prefix}_X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/{prefix}_X_test.csv", index=False)
    y_train_bal.to_csv(f"{output_dir}/{prefix}_y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/{prefix}_y_test.csv", index=False)

    return X_train_bal, X_test, y_train_bal, y_test
