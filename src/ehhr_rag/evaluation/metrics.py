import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero_metric = (0, 0, 0)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_predictions(data: list[dict[str, Any]], answer_name: str = "final_answer", gold_answer_name: str = "answers", context_name: str = "context"):
    metrics = {"em": 0, "f1": 0, "prec": 0, "recall": 0, "succ": 0}
    count = 0
    succ_count = 0
    skip_succ_answers = {"yes", "no", "noanswer"}
    for dp in data:
        pred = dp.get(answer_name, "")
        if not isinstance(pred, str):
            pred = ""
        gold_answers = dp.get(gold_answer_name, []) or []
        if not gold_answers:
            continue
        should_skip_succ = all(str(gold).lower().strip() in skip_succ_answers for gold in gold_answers if gold)
        succ = 0
        if not should_skip_succ:
            context = dp.get(context_name, "")
            if context and gold_answers:
                for gold in gold_answers:
                    if gold and gold in context:
                        succ = 1
                        break
            metrics["succ"] += succ
            succ_count += 1
        best = {"f1": -1.0, "em": 0.0, "prec": 0.0, "recall": 0.0}
        for gold in gold_answers:
            em = exact_match_score(pred, gold)
            f1, prec, recall = f1_score(pred, gold)
            if f1 > best["f1"]:
                best = {"f1": f1, "em": float(em), "prec": prec, "recall": recall}
        metrics["em"] += best["em"]
        metrics["f1"] += best["f1"]
        metrics["prec"] += best["prec"]
        metrics["recall"] += best["recall"]
        count += 1
    metrics["em"] = metrics["em"] / count if count > 0 else 0.0
    metrics["f1"] = metrics["f1"] / count if count > 0 else 0.0
    metrics["prec"] = metrics["prec"] / count if count > 0 else 0.0
    metrics["recall"] = metrics["recall"] / count if count > 0 else 0.0
    metrics["succ"] = metrics["succ"] / succ_count if succ_count > 0 else 0.0
    return {key: round(value, 3) for key, value in metrics.items()}
