"""
Machine Learning Classifiers for Voice Pathology Detection.
Includes:
1. Binary Classification: Random Forest on Jitter, Shimmer, CPP.
2. Multiclass Classification: Distance-weighted k-NN with StandardScaler on 6-feature vector.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_binary_random_forest(
    features_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate a Random Forest classifier for binary discrimination (Healthy vs Pathological).

    Features used: ['Jitter', 'Shimmer', 'CPP']
    Target: 'Target' (0 = Healthy, 1 = Pathological)
    """
    X = features_df[['Jitter', 'Shimmer', 'CPP']]
    y = features_df['Target']

    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Predictions
    y_test_pred = clf.predict(X_test)
    y_full_pred = clf.predict(X_clean)

    acc_test = accuracy_score(y_test, y_test_pred)
    acc_full = accuracy_score(y, y_full_pred)
    cm_full = confusion_matrix(y, y_full_pred)

    importances = dict(zip(['Jitter', 'Shimmer', 'CPP'], clf.feature_importances_))

    return {
        "model": clf,
        "test_accuracy": acc_test,
        "full_accuracy": acc_full,
        "confusion_matrix": cm_full,
        "feature_importances": importances,
        "classification_report": classification_report(y, y_full_pred, target_names=["Healthy", "Pathological"])
    }


def train_multiclass_knn(
    spectral_df: pd.DataFrame,
    n_neighbors: int = 5,
    cv_folds: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate a k-NN pipeline with scaling for 3-class pathology discrimination:
    ['hyperkinetic dysphonia', 'hypokinetic dysphonia', 'reflux laryngitis'].

    Features: ['LowFreq', 'MidFreq', 'HighFreq', 'Jitter', 'Shimmer', 'CPP']
    """
    feature_cols = ['LowFreq', 'MidFreq', 'HighFreq', 'Jitter', 'Shimmer', 'CPP']
    X = spectral_df[feature_cols]
    y = spectral_df['Diagnosis']

    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.20, random_state=random_state, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    # 10-fold cross validation on full dataset
    cv_scores = cross_val_score(pipeline, X_clean, y, cv=cv_folds, scoring='accuracy')

    return {
        "pipeline": pipeline,
        "test_accuracy": test_acc,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_min": float(cv_scores.min()),
        "cv_max": float(cv_scores.max()),
        "classification_report": classification_report(y_test, y_test_pred)
    }
