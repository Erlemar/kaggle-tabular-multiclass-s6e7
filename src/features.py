import sys
from pathlib import Path
import numpy as np
import pandas as pd

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C

def generate_features():
    print("Loading data...")
    train = C.load_train()
    test = C.load_test()
    folds = C.load_folds()

    df = pd.concat([train.drop(columns=[C.TARGET]), test], axis=0, ignore_index=True)
    
    numeric_cols = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"]
    cat_cols = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]

    features = pd.DataFrame({C.ID: df[C.ID]})

    print("1. Missing indicators...")
    for col in df.columns:
        if df[col].isna().sum() > 0 and col != C.ID:
            features[f"{col}_isnull"] = df[col].isna().astype(np.int8)

    print("2. Categorical features as category...")
    for c in cat_cols:
        features[c] = df[c].fillna("Missing").astype("category")
        
    print("3. Frequency/Count Encoding...")
    for c in cat_cols:
        counts = df[c].fillna("Missing").value_counts(dropna=False)
        features[f"{c}_count"] = df[c].fillna("Missing").map(counts).astype(np.float32)
        
    print("4. Numerics (raw)...")
    for c in numeric_cols:
        features[c] = df[c].astype(np.float32)
        
    print("5. Ratios...")
    eps = 1e-5
    features["ratio_step_calorie"] = (df["step_count"] / (df["calorie_expenditure"] + eps)).astype(np.float32)
    features["ratio_step_bmi"] = (df["step_count"] / (df["bmi"] + eps)).astype(np.float32)
    features["ratio_exercise_calorie"] = (df["exercise_duration"] / (df["calorie_expenditure"] + eps)).astype(np.float32)
    features["ratio_sleep_exercise"] = (df["sleep_duration"] / (df["exercise_duration"] + eps)).astype(np.float32)
    features["ratio_water_exercise"] = (df["water_intake"] / (df["exercise_duration"] + eps)).astype(np.float32)
    features["ratio_water_bmi"] = (df["water_intake"] / (df["bmi"] + eps)).astype(np.float32)
    
    print("6. Group aggregations...")
    group_cats = ["gender", "stress_level", "diet_type"]
    for gc in group_cats:
        for num in numeric_cols:
            grouped = df.groupby(gc)[num]
            features[f"{num}_mean_by_{gc}"] = df[gc].map(grouped.mean()).astype(np.float32)
            features[f"{num}_std_by_{gc}"] = df[gc].map(grouped.std()).astype(np.float32)
            features[f"{num}_min_by_{gc}"] = df[gc].map(grouped.min()).astype(np.float32)
            features[f"{num}_max_by_{gc}"] = df[gc].map(grouped.max()).astype(np.float32)

    print("7. OOF Target Encoding...")
    train_ids = train[C.ID].to_numpy()
    test_ids = test[C.ID].to_numpy()
    
    labels = C.class_labels(train)
    y_idx = C.encode_target(train[C.TARGET].to_numpy(), labels)
    
    fold_df = folds.set_index(C.ID)
    train_folds = fold_df.loc[train_ids, "fold"].to_numpy()
    n_folds = train_folds.max() + 1
    
    for c in cat_cols:
        for class_idx, class_name in enumerate(labels):
            col_name = f"TE_{c}_{class_name}"
            train_te = np.zeros(len(train), dtype=np.float32)
            test_te = np.zeros((len(test), n_folds), dtype=np.float32)
            
            bin_y = (y_idx == class_idx).astype(np.float32)
            prior = bin_y.mean()
            
            for f in range(n_folds):
                val_idx = train_folds == f
                trn_idx = train_folds != f
                
                trn_cat = train.iloc[trn_idx][c].fillna("Missing")
                val_cat = train.iloc[val_idx][c].fillna("Missing")
                tst_cat = test[c].fillna("Missing")
                
                stats = pd.DataFrame({"cat": trn_cat, "y": bin_y[trn_idx]}).groupby("cat")["y"].agg(["sum", "count"])
                smoothing = 10
                stats["smooth"] = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
                
                mapping = stats["smooth"].to_dict()
                
                train_te[val_idx] = val_cat.map(mapping).fillna(prior).astype(np.float32)
                test_te[:, f] = tst_cat.map(mapping).fillna(prior).astype(np.float32)
                
            train_te[train_folds == -1] = prior
            
                
            feat_train = pd.Series(train_te, index=train_ids)
            feat_test = pd.Series(test_te.mean(axis=1), index=test_ids)
            
            feat_all = pd.concat([feat_train, feat_test], axis=0).reindex(features[C.ID])
            features[col_name] = feat_all.to_numpy()

    print("Saving features...")

        
    feat_train_df = features[features[C.ID].isin(train_ids)].copy()
    feat_test_df = features[features[C.ID].isin(test_ids)].copy()
    
    feat_train_df = feat_train_df.set_index(C.ID).reindex(train_ids).reset_index()
    feat_test_df = feat_test_df.set_index(C.ID).reindex(test_ids).reset_index()
    
    out_dir = C.ART
    feat_train_df.to_parquet(out_dir / "features_train.parquet", index=False)
    feat_test_df.to_parquet(out_dir / "features_test.parquet", index=False)
    
    print(f"Done! Train features: {feat_train_df.shape}, Test features: {feat_test_df.shape}")

if __name__ == '__main__':
    generate_features()
