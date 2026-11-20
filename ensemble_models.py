"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def _as_array(values: Any) -> np.ndarray:
    """Поддерживает как numpy-массивы, так и таблицы pandas."""
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def build_candidate_models(random_seed: int) -> dict[str, Any]:
    """Создаёт модели с небольшим числом параметров для сравнения."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=8,
            subsample=0.8,
            random_state=random_seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=5.0,
            random_state=random_seed,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.03,
            random_state=random_seed,
        ),
        "MLPClassifier": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1.0,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_seed,
            ),
        ),
    }


def calculate_metrics(
    target: Any,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Считает метрики для положительного класса 1."""
    target_array = _as_array(target)
    return {
        "roc_auc": float(roc_auc_score(target_array, probabilities)),
        "accuracy": float(accuracy_score(target_array, predictions)),
        "precision": float(
            precision_score(target_array, predictions, zero_division=0)
        ),
        "recall": float(recall_score(target_array, predictions, zero_division=0)),
        "f1": float(f1_score(target_array, predictions, zero_division=0)),
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Выбирает ансамбль на validation и считает метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    train_features = _as_array(data.train_features)
    train_target = _as_array(data.train_target)
    test_features = _as_array(data.test_features)
    test_target = _as_array(data.test_target)

    fit_features, validation_features, fit_target, validation_target = (
        train_test_split(
            train_features,
            train_target,
            test_size=0.2,
            random_state=data.random_seed + 1,
            stratify=train_target,
        )
    )

    best_model_name = ""
    best_validation_auc = -1.0
    for model_name, model in build_candidate_models(data.random_seed).items():
        model.fit(fit_features, fit_target)
        probabilities = model.predict_proba(validation_features)[:, 1]
        validation_auc = roc_auc_score(validation_target, probabilities)
        if validation_auc > best_validation_auc:
            best_model_name = model_name
            best_validation_auc = float(validation_auc)

    selected_model = build_candidate_models(data.random_seed)[best_model_name]
    selected_model.fit(train_features, train_target)
    probabilities = selected_model.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(test_target, probabilities, predictions)

    return EnsembleResult(
        model_name=best_model_name,
        probabilities=probabilities,
        predictions=predictions,
        **metrics,
    )
