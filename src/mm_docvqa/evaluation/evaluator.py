from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm_docvqa.domain.schemas import DocumentSample, PredictionRecord
from mm_docvqa.evaluation.metrics import anls_score, exact_match_score


@dataclass(slots=True)
class SampleEvalResult:
    """
    Evaluation result for one sample/question.
    """

    question_id: str
    question: str
    prediction: str
    answers: list[str]
    exact_match: float
    anls: float
    is_missing_prediction: bool = False


@dataclass(slots=True)
class EvalSummary:
    """
    Aggregated evaluation summary for a whole dataset split.
    """

    n_samples: int
    n_predicted: int
    n_missing: int
    exact_match: float
    anls: float
    per_sample: list[SampleEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_predicted": self.n_predicted,
            "n_missing": self.n_missing,
            "exact_match": self.exact_match,
            "anls": self.anls,
        }


def _build_prediction_map(
    predictions: list[PredictionRecord] | list[dict[str, Any]],
) -> dict[str, str]:
    """
    Convert predictions into a map:
    question_id -> prediction_text
    """
    pred_map: dict[str, str] = {}

    for item in predictions:
        if isinstance(item, PredictionRecord):
            pred_map[item.question_id] = item.prediction
        elif isinstance(item, dict):
            question_id = str(item["question_id"]).strip()
            prediction = str(item["prediction"]).strip()
            pred_map[question_id] = prediction
        else:
            raise TypeError(f"Unsupported prediction type: {type(item)}")

    return pred_map


def evaluate_predictions(
    samples: list[DocumentSample],
    predictions: list[PredictionRecord] | list[dict[str, Any]],
    anls_threshold: float = 0.5,
) -> EvalSummary:
    """
    Evaluate a batch of predictions against samples.

    Inputs:
    - samples: ground-truth samples
    - predictions: model outputs

    Output:
    - aggregated summary
    - per-sample results
    """
    pred_map = _build_prediction_map(predictions)

    per_sample: list[SampleEvalResult] = []
    em_total = 0.0
    anls_total = 0.0
    n_missing = 0

    for sample in samples:
        pred_text = pred_map.get(sample.question_id, "")
        is_missing = sample.question_id not in pred_map

        if is_missing:
            n_missing += 1

        em = exact_match_score(pred_text, sample.answers)
        anls = anls_score(pred_text, sample.answers, threshold=anls_threshold)

        em_total += em
        anls_total += anls

        per_sample.append(
            SampleEvalResult(
                question_id=sample.question_id,
                question=sample.question,
                prediction=pred_text,
                answers=sample.answers,
                exact_match=em,
                anls=anls,
                is_missing_prediction=is_missing,
            )
        )

    n_samples = len(samples)
    n_predicted = n_samples - n_missing

    return EvalSummary(
        n_samples=n_samples,
        n_predicted=n_predicted,
        n_missing=n_missing,
        exact_match=(em_total / n_samples) if n_samples else 0.0,
        anls=(anls_total / n_samples) if n_samples else 0.0,
        per_sample=per_sample,
    )