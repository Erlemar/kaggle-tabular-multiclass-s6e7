import sys
from pathlib import Path
_p = Path(__file__).resolve()
sys.path.insert(0, str(_p.parent / "src"))
import common as C

C.log_experiment(
    name="baseline_prior",
    family="prior",
    cv=0.85867,
    cv_std=0.0,
    n_feat=0,
    notes="Class prior baseline (at-risk)"
)
