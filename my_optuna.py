import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Objective function for Optuna
def objective(trial, X_train, y_train):
    # Suggest hyperparameters to search
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])  # Changed to categorical
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)  # Node
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)  # Node
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5)  # Node
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.1)  # Node
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=10)  # Number of trees
    max_depth = trial.suggest_int("max_depth", 1, 50)  # Maximum depth of trees
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])  # Features to consider at each split
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])  # Whether to use bootstrap samples

    # Create the model with suggested hyperparameters
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42,
        n_jobs=-1
    )

    # Evaluate using cross-validation (with ROC-AUC as scoring metric)
    roc_auc = cross_val_score(rf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1).mean()

    return roc_auc
