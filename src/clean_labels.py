import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C

def clean_labels():
    print("Loading data for confident learning...")
    train = C.load_train()
    
    # We will use raw features to find noisy labels
    features = [c for c in train.columns if c not in [C.ID, C.TARGET]]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    
    X = train[features].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        
    labels = C.class_labels(train)
    y_idx = C.encode_target(train[C.TARGET].to_numpy(), labels)
    
    print(f"Training 5-fold CV to compute OOF probabilities on {len(train)} rows...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros((len(train), len(labels)))
    
    params = {
        "objective": "multiclass",
        "num_class": len(labels),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42
    }
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y_idx)):
        print(f"Fold {fold+1}/5...")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_idx[tr_idx], y_idx[va_idx]
        
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        
        booster = lgb.train(
            params, dtr, num_boost_round=1000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        oof_probs[va_idx] = booster.predict(X_va)
        
    print("Computing noise...")
    # Find the probability assigned to the true class
    true_class_probs = oof_probs[np.arange(len(train)), y_idx]
    
    # Threshold for noise: if the true class probability is very low
    threshold = 0.05
    is_noisy = true_class_probs < threshold
    
    noisy_count = is_noisy.sum()
    print(f"Found {noisy_count} noisy labels ({noisy_count/len(train)*100:.2f}%)")
    
    clean_ids = train.loc[~is_noisy, C.ID].tolist()
    
    out_path = C.ART / "clean_ids.parquet"
    pd.DataFrame({C.ID: clean_ids}).to_parquet(out_path, index=False)
    print(f"Saved {len(clean_ids)} clean IDs to {out_path}")

if __name__ == "__main__":
    clean_labels()
