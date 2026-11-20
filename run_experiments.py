"""Запускает сравнение ансамблей на немецком кредитном датасете."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ensemble_models import (
    build_candidate_models,
    calculate_metrics,
    run_ensemble_experiment,
)
from grader_contracts.ensembles import EnsembleInput


DATA_DIR = Path(__file__).parent
RANDOM_SEED = 42


def main() -> None:
    """Обучает модели, выводит сравнение и сохраняет измеренные метрики."""
    credit_data = pd.read_csv(DATA_DIR / "german.csv", sep=";")
    target = credit_data["Creditability"].to_numpy()
    features = credit_data.drop(columns="Creditability").to_numpy()

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=target,
    )
    experiment_input = EnsembleInput(
        train_features=train_features,
        train_target=train_target,
        test_features=test_features,
        test_target=test_target,
        random_seed=RANDOM_SEED,
    )

    fit_features, validation_features, fit_target, validation_target = (
        train_test_split(
            train_features,
            train_target,
            test_size=0.2,
            random_state=RANDOM_SEED + 1,
            stratify=train_target,
        )
    )

    model_results = {}
    validation_scores = {}
    for model_name, model in build_candidate_models(RANDOM_SEED).items():
        model.fit(fit_features, fit_target)
        validation_probabilities = model.predict_proba(validation_features)[:, 1]
        validation_scores[model_name] = float(
            roc_auc_score(validation_target, validation_probabilities)
        )

        model.fit(train_features, train_target)
        test_probabilities = model.predict_proba(test_features)[:, 1]
        test_predictions = (test_probabilities >= 0.5).astype(int)
        model_results[model_name] = {
            "validation_roc_auc": validation_scores[model_name],
            **calculate_metrics(test_target, test_probabilities, test_predictions),
        }

    selected_result = run_ensemble_experiment(experiment_input)
    chosen_validation_auc = validation_scores[selected_result.model_name]

    results = {
        "data_split": {
            "rows": int(len(credit_data)),
            "features": int(features.shape[1]),
            "test_size": 0.2,
            "random_seed": RANDOM_SEED,
            "stratified": True,
        },
        "selection": {
            "criterion": "ROC-AUC on a validation split from training data",
            "validation_size_of_training": 0.2,
            "validation_random_seed": RANDOM_SEED + 1,
            "selected_model": selected_result.model_name,
            "selected_validation_roc_auc": chosen_validation_auc,
        },
        "experiments": model_results,
        "selected_model_test_metrics": {
            "roc_auc": selected_result.roc_auc,
            "accuracy": selected_result.accuracy,
            "precision": selected_result.precision,
            "recall": selected_result.recall,
            "f1": selected_result.f1,
        },
    }

    output_path = DATA_DIR / "experiment_results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
