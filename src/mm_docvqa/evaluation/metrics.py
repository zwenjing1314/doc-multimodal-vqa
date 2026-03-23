from __future__ import annotations

from mm_docvqa.evaluation.normalize import normalize_answers, normalize_text


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute classic Levenshtein edit distance between two strings.

    Edit distance = minimum number of insertions, deletions,
    or substitutions needed to turn a into b.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (0 if char_a == char_b else 1)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = current_row

    return prev_row[-1]


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    """
    Convert edit distance into similarity score in [0, 1].

    similarity = 1 - dist / max(len(a), len(b))
    """
    if a == b:
        return 1.0

    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0

    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)


def exact_match_score(prediction: str, answers: list[str]) -> float:
    """
    Return 1.0 if normalized prediction exactly matches
    any normalized reference answer, else 0.0.
    """
    pred = normalize_text(prediction)  # 预测答案， 是一个字符串
    refs = normalize_answers(answers)  # 标准答案， 是一个列表

    if not refs:
        return 0.0  # 没有标准答案，返回0.0

    return 1.0 if pred in set(refs) else 0.0  # 预测答案是否在标准答案中


def best_answer_similarity(prediction: str, answers: list[str]) -> float:
    """
    Return the best normalized Levenshtein similarity
    between prediction and any reference answer.
    """
    pred = normalize_text(prediction)
    refs = normalize_answers(answers)

    if not refs:
        return 0.0

    best = 0.0
    for ref in refs:
        sim = normalized_levenshtein_similarity(pred, ref)
        if sim > best:
            best = sim
    return best


def anls_score(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """
    Average Normalized Levenshtein Similarity style score.

    We compute the best similarity between prediction and references.
    If the similarity is below threshold, score becomes 0.

    This makes the metric tolerant to small OCR-like errors,
    but still rejects clearly wrong answers.
    """
    best_sim = best_answer_similarity(prediction, answers)
    return best_sim if best_sim >= threshold else 0.0
