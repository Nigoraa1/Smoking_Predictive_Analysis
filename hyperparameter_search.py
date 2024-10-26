from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def randomized_search_rf(X_train, y_train, scoring='roc_auc', cv=5, n_iter=50):
    param_dist = {
        'n_estimators': np.arange(50, 301, 50),
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': np.arange(2, 16, 3),
        'min_samples_leaf': np.arange(1, 9, 2),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        scoring=scoring,
        n_iter=n_iter,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
