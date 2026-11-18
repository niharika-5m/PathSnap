"""Model zoo used in the comparison. Each entry records its family and the
initial (untuned) configuration so it can be reported."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED

# max_rows: training rows are randomly subsampled to this many for slow models
# (recorded in the results so it can be reported).
MODEL_SPECS = {
    "logistic_regression": dict(
        family="Linear / probabilistic", max_rows=None,
        params=dict(C=1.0, max_iter=1000),
        build=lambda p: make_pipeline(StandardScaler(), LogisticRegression(**p))),
    "gaussian_nb": dict(
        family="Probabilistic (generative)", max_rows=None,
        params=dict(var_smoothing=1e-9),
        build=lambda p: make_pipeline(StandardScaler(), GaussianNB(**p))),
    "knn": dict(
        family="Instance-based", max_rows=200_000,
        params=dict(n_neighbors=25, weights="distance", n_jobs=-1),
        build=lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p))),
    "decision_tree": dict(
        family="Tree-based", max_rows=None,
        params=dict(max_depth=12, min_samples_leaf=50, random_state=SEED),
        build=lambda p: DecisionTreeClassifier(**p)),
    "random_forest": dict(
        family="Ensemble (bagging)", max_rows=600_000,
        params=dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED),
        build=lambda p: RandomForestClassifier(**p)),
    "extra_trees": dict(
        family="Ensemble (randomised trees)", max_rows=600_000,
        params=dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED),
        build=lambda p: ExtraTreesClassifier(**p)),
    "hist_gradient_boosting": dict(
        family="Boosting", max_rows=None,
        params=dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED),
        build=lambda p: HistGradientBoostingClassifier(**p)),
    "lightgbm": dict(
        family="Boosting", max_rows=None,
        params=dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1),
        build=lambda p: LGBMClassifier(**p)),
    "xgboost": dict(
        family="Boosting", max_rows=None,
        params=dict(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        build=lambda p: XGBClassifier(**p)),
    "mlp": dict(
        family="Neural network", max_rows=400_000,
        params=dict(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                    max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
                    random_state=SEED),
        build=lambda p: make_pipeline(StandardScaler(), MLPClassifier(**p))),
}


def build_model(name, params=None):
    spec = MODEL_SPECS[name]
    return spec["build"](params if params is not None else spec["params"])


def search_space(name, trial):
    """Hyperparameter search space used by the Bayesian optimisation."""
    base = dict(MODEL_SPECS[name]["params"])
    if name == "lightgbm":
        base.update(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    elif name == "xgboost":
        base.update(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    elif name == "hist_gradient_boosting":
        base.update(
            max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        )
    elif name in ("random_forest", "extra_trees"):
        base.update(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 8, 40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=trial.suggest_float("max_features", 0.2, 1.0),
        )
    elif name == "mlp":
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        depth = trial.suggest_int("depth", 1, 3)
        base.update(
            hidden_layer_sizes=tuple([width] * depth),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        )
    else:
        raise SystemExit(f"no search space defined for {name}")
    return base
