#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert your OI monitor log into a proper ML dataset (training_data.csv)
Compatible with XGBoost ML training script.
"""

import pandas as pd
import re
from datetime import datetime

# ----------------------------------------------------------
# CONFIG — CHANGE THIS TO YOUR ACTUAL LOG FILE NAME
# ----------------------------------------------------------
LOG_FILE = "OI_data_NIFTY_with_ML.txt"    # <-- CHANGE THIS
OUTPUT_FILE = "training_data.csv"
# ----------------------------------------------------------

rows = []

# Regex to extract fields
pattern = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),"
    r"ATM=(?P<atm>\d+),"
    r"CallOI=(?P<calloi>\d+),"
    r"PutOI=(?P<putoi>\d+),"
    r"TotalOI=(?P<totaloi>\d+),"
    r"Diff=(?P<diff>-?\d+),"
    r"PCR=(?P<pcr>\d+\.\d+),"
    r"Sentiment=(?P<sentiment>\w+)"
    r"(,ML=(?P<mlsig>\w+))?"
    r"(,ML_prob=(?P<mlprob>\d+\.\d+))?"
)

print(f"[INFO] Reading log file: {LOG_FILE}")

with open(LOG_FILE, "r") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue

        data = m.groupdict()
        ts = datetime.strptime(data["ts"], "%Y-%m-%d %H:%M:%S")

        row = {
            "time": ts,
            "atm": int(data["atm"]),
            "total_call_oi": int(data["calloi"]),
            "total_put_oi": int(data["putoi"]),
            "total_oi": int(data["totaloi"]),
            "diff": int(data["diff"]),
            "pcr": float(data["pcr"]),
            "sentiment": data["sentiment"],
            "ml_signal": data.get("mlsig"),
            "ml_prob": float(data["mlprob"]) if data.get("mlprob") else None,
        }

        rows.append(row)

# Build dataframe
df = pd.DataFrame(rows)
df = df.sort_values("time")
df = df.reset_index(drop=True)

# ----------------------------------------------------------
# Compute derived ML features (matching your training script)
# ----------------------------------------------------------
df["pcr_change"] = df["pcr"].diff()
df["oi_diff_change"] = df["diff"].diff()

# Spot/VWAP unavailable in log — set placeholder
df["spot"] = df["atm"].astype(float)  # ATM as proxy (better than nothing)
df["vwap"] = df["spot"].rolling(5).mean()  # simple approx
df["spot_vwap_diff"] = df["spot"] - df["vwap"]

df["pcr_roc_3"] = df["pcr"].rolling(3).apply(lambda x: x.iloc[-1] - x.iloc[0])
df["spot_change"] = df["spot"].diff()

# CE/PE IV missing in log — set placeholders
df["ce_iv"] = 0.0
df["pe_iv"] = 0.0

df = df.dropna()

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)

print("[SUCCESS] Created training_data.csv")
print(f"[INFO] Rows: {len(df)}")
print("[INFO] Now run your XGBoost training script.")
