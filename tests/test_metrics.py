from mm_docvqa.domain.schemas import DocumentSample, PredictionRecord
from mm_docvqa.evaluation.evaluator import evaluate_predictions
from mm_docvqa.evaluation.metrics import (
    anls_score,
    best_answer_similarity,
    exact_match_score,
    levenshtein_distance,
    normalized_levenshtein_similarity,
)
from mm_docvqa.evaluation.normalize import normalize_answers, normalize_text


def test_normalize_text() -> None:
    assert normalize_text("  P. Carter  ") == "p. carter"
    assert normalize_text("Review   of   existing") == "review of existing"
    assert normalize_text(None) == ""


def test_normalize_answers() -> None:
    answers = ["  P. Carter ", "p. carter", ""]
    assert normalize_answers(answers) == ["p. carter", "p. carter"]


def test_levenshtein_distance() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("carter", "carter") == 0


def test_normalized_levenshtein_similarity() -> None:
    assert normalized_levenshtein_similarity("abc", "abc") == 1.0
    sim = normalized_levenshtein_similarity("carter", "carte")
    assert 0.0 < sim < 1.0


def test_exact_match_score_case_insensitive() -> None:
    pred = "p. carter"
    refs = ["P. Carter", "p. carter"]
    assert exact_match_score(pred, refs) == 1.0


def test_exact_match_score_failure() -> None:
    pred = "john smith"
    refs = ["P. Carter"]
    assert exact_match_score(pred, refs) == 0.0


def test_best_answer_similarity() -> None:
    pred = "p carter"
    refs = ["P. Carter", "p. carter"]
    sim = best_answer_similarity(pred, refs)
    assert sim > 0.5


def test_anls_score_exact() -> None:
    pred = "1/8/93"
    refs = ["1/8/93"]
    assert anls_score(pred, refs) == 1.0


def test_anls_score_soft_match() -> None:
    pred = "p carter"
    refs = ["P. Carter"]
    score = anls_score(pred, refs, threshold=0.5)
    assert score > 0.5


def test_evaluate_predictions() -> None:
    samples = [
        DocumentSample(
            dataset_name="docvqa",
            split="train",
            question_id="337",
            question="what is the date mentioned in this letter?",
            image_path="documents/xnbl0037_1.png",
            answers=["1/8/93"],
            question_types=["handwritten", "form"],
            doc_numeric_id=279,
            doc_id="xnbl0037",
            page_no=1,
        ),
        DocumentSample(
            dataset_name="docvqa",
            split="train",
            question_id="338",
            question="what is the contact person name mentioned in letter?",
            image_path="documents/xnbl0037_1.png",
            answers=["P. Carter", "p. carter"],
            question_types=["handwritten", "form"],
            doc_numeric_id=279,
            doc_id="xnbl0037",
            page_no=1,
        ),
    ]

    predictions = [
        PredictionRecord(question_id="337", prediction="1/8/93", dataset_name="docvqa"),
        PredictionRecord(question_id="338", prediction="p carter", dataset_name="docvqa"),
    ]

    summary = evaluate_predictions(samples, predictions)

    assert summary.n_samples == 2
    assert summary.n_predicted == 2
    assert summary.n_missing == 0
    assert summary.exact_match < 1.0
    assert summary.anls > 0.5
    assert len(summary.per_sample) == 2


def test_evaluate_predictions_with_missing_prediction() -> None:
    samples = [
        DocumentSample(
            dataset_name="docvqa",
            split="train",
            question_id="337",
            question="what is the date mentioned in this letter?",
            image_path="documents/xnbl0037_1.png",
            answers=["1/8/93"],
        )
    ]

    predictions: list[PredictionRecord] = []

    summary = evaluate_predictions(samples, predictions)

    assert summary.n_samples == 1
    assert summary.n_predicted == 0
    assert summary.n_missing == 1
    assert summary.exact_match == 0.0
    assert summary.anls == 0.0
    assert summary.per_sample[0].is_missing_prediction is True