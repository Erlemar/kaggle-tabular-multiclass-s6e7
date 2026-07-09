import pandas as pd

df = pd.read_csv("/kaggle/input/competitions/playground-series-s6e7/train.csv")
df.head()

numeric_cols = df.select_dtypes(include="number").drop("id", axis=1).columns
numeric_cols = [c for c in numeric_cols if c != 'id']
catgorical_cols = df.select_dtypes(include="object").columns

print(numeric_cols)
print(catgorical_cols)

df.describe().T

from sklearn.preprocessing import MinMaxScaler

def feature_engineer(df, scaler=None):
    target = 'health_condition'
    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [c for c in numeric_cols if c != "id"]
    _norm = df.copy()
    if scaler:
        _norm[numeric_cols] = scaler.transform(_norm[numeric_cols])
    else:
        scaler = MinMaxScaler()
        _norm[numeric_cols] = scaler.fit_transform(_norm[numeric_cols])

    assert int(_norm[numeric_cols].min().sum()) == 0, "improper normalization"

    stress = df['stress_level'].map({'low': 1.0, 'medium': 2.0, 'high': 3.0})
    activity = df['physical_activity_level'].map({
        "sedentary": 1.0,
        "moderate": 2.0,
        "active": 3.0
    })
    df['stress_x_activity'] = stress * activity
    # df['bmi_activity_ratio'] = df['bmi'] / activity
    df['bmi_x_activity'] = df['bmi'] * activity
    # df['bmi_x_calorie'] = df['bmi'] * df['calorie_expenditure']
    # # df['stress_bmi_ratio'] = df['stress'] / df['bmi']
    # # df['stress_activity_sleep'] = df['stress_x_activity'] / df['sleep_duration']
    # # df['sleep_quality'] = df['sleep_quality'].map({
    # #     "poor": 1.0,
    # #     "average": 2.0,
    # #     "good": 3.0
    # # })
    # # df['sleep_deprived'] = 7.5 - df['sleep_duration']
    df['sleep_stress_ratio'] = df['sleep_duration'] / stress
    # # df['fitness'] = (0.6 * _norm['step_count'] +
    # #                  0.3 * _norm['exercise_duration'] +
    # #                  0.1 * _norm['calorie_expenditure'])
    # df['life'] = df['heart_rate'] * df['bmi'] / (df['exercise_duration'] * df['step_count'])

    # df.drop(columns=['gender','diet_type', 'smoking_alcohol'], inplace=True)

    # df['life_quality'] = df['fitness'] / df['life']

    return df

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit_transform(df[numeric_cols])
train = feature_engineer(df, scaler)
train.head()

from sklearn.model_selection import StratifiedKFold

target = "health_condition"
features = [c for c in train.columns if c not in (target, "id")]

X = train[features].copy()
y = train[target]

# LightGBM handles categoricals natively via the pandas "category" dtype
cat_cols = X.select_dtypes(exclude="number").columns
for c in cat_cols:
    X[c] = X[c].fillna("missing")
    X[c] = X[c].astype("category")

num_cols = X.select_dtypes(include="number").columns[1:]
for c in num_cols:
    X[c] = X[c].fillna(X[c].median())

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier
import numpy as np

train_scores = []
valid_scores = []

for train_idx, valid_idx in skf.split(X, y):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    lgb_model = LGBMClassifier(random_state=42,
                        num_leaves=31,
                        max_depth=-1,
                        learning_rate=0.1, 
                        n_estimators=100, 
                        objective='multiclass',
                        verbose=-1, 
                        n_jobs=-1,
                        class_weight="balanced")
    lgb_model.fit(X_train, 
                  y_train)

    lgb_train_preds = lgb_model.predict(X_train)
    lgb_val_preds = lgb_model.predict(X_valid)

    train_scores += [balanced_accuracy_score(y_train, lgb_train_preds)]
    valid_scores += [balanced_accuracy_score(y_valid, lgb_val_preds)]

print(f"train_score: {np.mean(train_scores)}")
print(f"vaild_score: {np.mean(valid_scores)}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_valid, lgb_val_preds, labels=list(y_valid.unique()), normalize='all')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(y_valid.unique())).plot()

from lightgbm import plot_importance
import matplotlib.pyplot as plt

plot_importance(
    lgb_model,
    importance_type="gain",
    max_num_features=30,
    figsize=(8,10)
)

plt.show()

import pandas as pd
test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e7/test.csv")
test = feature_engineer(test, scaler=scaler)

target = "health_condition"
features = [c for c in test.columns if c not in (target, "id")]

X_test = test[features].copy()

# LightGBM handles categoricals natively via the pandas "category" dtype
cat_cols_test = X_test.select_dtypes(exclude="number").columns
for c in cat_cols_test:
    X_test[c] = X_test[c].fillna("missing")
    X_test[c] = X_test[c].astype("category")

num_cols_test = X_test.select_dtypes(include="number").columns[1:]
for c in num_cols_test:
    X_test[c] = X_test[c].fillna(X[c].median())

lgb_submission = pd.DataFrame({
    "id": test["id"],
    target: lgb_model.predict(X_test).ravel(),
})

lgb_submission.to_csv("submission.csv", index=False)
lgb_submission
