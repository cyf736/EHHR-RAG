import argparse
import json
import os
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ehhr_rag.config import dataset_db_dir, dataset_outputs_dir, dataset_raw_dir
from ehhr_rag.llm import get_token_stats, reset_token_stats
from ehhr_rag.logging_utils import logger
from ehhr_rag.reasoning.evidence_aware import EvidenceAwareReasoner
from ehhr_rag.retrieval.retriever import MultiHypergraphRetriever


def _load_questions(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return [item for item in data if isinstance(item, dict) and item.get("question")]


def _load_existing_results(output_json: str) -> Dict[str, Any]:
    if not os.path.exists(output_json):
        return {"by_id": {}, "list": []}
    try:
        with open(output_json, encoding="utf-8") as f:
            results_list = json.load(f)
        by_id = {item.get("id"): item for item in results_list if isinstance(item, dict)}
        return {"by_id": by_id, "list": results_list}
    except Exception as exc:
        logger.info("warning: could not load existing results from %s: %s", output_json, exc)
        return {"by_id": {}, "list": []}


def process_question(item: Dict[str, Any], retriever=None, max_loops: int = 8) -> Dict[str, Any]:
    question = item["question"]
    answers = item.get("answers", [])
    qid = item.get("id") or item.get("question_id")
    reasoner = EvidenceAwareReasoner(original_query=question, retriever=retriever, max_loops=max_loops)
    reasoner.execute()
    results = reasoner.get_results()
    results.update({"id": qid, "question": question, "answers": answers})
    return results


def _to_answer_list(answers: Any) -> List[str]:
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list):
        return []
    cleaned = []
    for ans in answers:
        s = str(ans or "").strip()
        if s:
            cleaned.append(s)
    return cleaned


def _normalize_answer(s: Any) -> str:
    text = str(s or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def _f1_score(prediction: str, ground_truth: str):
    normalized_prediction = _normalize_answer(prediction)
    normalized_ground_truth = _normalize_answer(ground_truth)
    zero_metric = (0.0, 0.0, 0.0)
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


def _exact_match_score(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _compute_prompt_succ_per_result(result: Dict[str, Any]) -> Dict[str, Any]:
    answers = _to_answer_list(result.get("answers", []))
    node_prompts = result.get("node_prompts", {}) or {}
    prompt_texts: List[str] = []
    for prompt_items in node_prompts.values():
        if not isinstance(prompt_items, list):
            continue
        for item in prompt_items:
            if isinstance(item, dict):
                prompt_input = str(item.get("prompt_input", "") or "").strip()
                response = str(item.get("response", "") or "").strip()
                combined = f"{prompt_input}\n{response}".strip()
            else:
                combined = str(item or "").strip()
            if combined:
                prompt_texts.append(combined)
    found = False
    matched_answer = ""
    matched_prompt_index = -1
    for idx, prompt_text in enumerate(prompt_texts):
        for answer in answers:
            if answer and answer in prompt_text:
                found = True
                matched_answer = answer
                matched_prompt_index = idx
                break
        if found:
            break
    result["succ"] = bool(found)
    result["succ_hit_answer"] = matched_answer
    result["succ_hit_prompt_index"] = matched_prompt_index
    result["succ_prompt_count"] = len(prompt_texts)
    return result


def _score_result(result: Dict[str, Any]) -> Dict[str, Any]:
    pred = str(result.get("final_answer", "") or "").strip()
    gold_answers = _to_answer_list(result.get("answers", []))
    best = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0, "gold": gold_answers[0] if gold_answers else ""}
    for gold in gold_answers:
        em = float(_exact_match_score(pred, gold))
        f1, prec, recall = _f1_score(pred, gold)
        if f1 > best["f1"] or (f1 == best["f1"] and em > best["em"]):
            best = {"em": em, "f1": f1, "prec": prec, "recall": recall, "gold": gold}
    result["score_detail"] = {
        "best_gold": best["gold"],
        "em": best["em"],
        "f1": best["f1"],
        "prec": best["prec"],
        "recall": best["recall"],
    }
    return result


def _evaluate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0, "succ": 0.0}
    count = 0
    succ_total = 0
    skip_succ_answers = {"yes", "no", "noanswer"}
    for item in results:
        if not isinstance(item, dict):
            continue
        item = _score_result(item)
        detail = item.get("score_detail", {}) or {}
        metrics["em"] += float(detail.get("em", 0.0) or 0.0)
        metrics["f1"] += float(detail.get("f1", 0.0) or 0.0)
        metrics["prec"] += float(detail.get("prec", 0.0) or 0.0)
        metrics["recall"] += float(detail.get("recall", 0.0) or 0.0)
        count += 1
        answers = _to_answer_list(item.get("answers", []))
        should_skip_succ = all(_normalize_answer(gold) in skip_succ_answers for gold in answers if gold)
        if not should_skip_succ:
            succ_total += 1
            if bool(item.get("succ", False)):
                metrics["succ"] += 1.0
    if count > 0:
        metrics["em"] /= count
        metrics["f1"] /= count
        metrics["prec"] /= count
        metrics["recall"] /= count
    metrics["succ"] = (metrics["succ"] / succ_total) if succ_total > 0 else 0.0
    return {key: round(value, 3) for key, value in metrics.items()}


def _build_error_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct_count = 0
    errors: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    for item in results:
        item = _score_result(item)
        detail = item.get("score_detail", {}) or {}
        em = bool(detail.get("em", 0.0))
        if em:
            correct_count += 1
            continue
        reason = _classify_error_reason(item)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        errors.append(
            {
                "id": item.get("id"),
                "question": item.get("question", ""),
                "prediction": item.get("final_answer", ""),
                "gold_answers": _to_answer_list(item.get("answers", [])),
                "best_gold": detail.get("gold", ""),
                "f1": round(float(detail.get("f1", 0.0) or 0.0), 3),
                "reason": reason,
                "sub_question_count": len(item.get("sub_questions", []) or []),
                "evidence_count": len(item.get("evidence_pool", []) or []),
                "retrieval_doc_count": len(item.get("retrieval_pool", []) or []),
            }
        )
    return {"correct_count": correct_count, "wrong_count": len(errors), "reason_counts": reason_counts, "errors": errors}


def _classify_error_reason(result: Dict[str, Any]) -> str:
    prediction = str(result.get("final_answer", "") or "").strip().lower()
    sub_questions = result.get("sub_questions", []) or []
    evidence_pool = result.get("evidence_pool", []) or []
    final_nodes = [sq for sq in sub_questions if sq.get("is_final")]
    final_ready = any(sq.get("ans") == "ready" for sq in final_nodes)
    final_high = any(sq.get("ans") == "ready" and sq.get("sup") == "high" for sq in final_nodes)
    if prediction in {"", "none", "unknown"}:
        if evidence_pool and not final_ready:
            return "final_subquestion_never_became_ready"
        if final_ready and not final_high:
            return "final_subquestion_support_insufficient"
        if len(sub_questions) >= 4:
            return "subquestion_expansion_did_not_unlock_answer"
        return "no_final_answer_generated"
    if final_ready and not final_high:
        return "answered_before_strong_final_support"
    if final_nodes and not final_ready:
        return "answer_generated_with_unresolved_final_subquestion"
    return "wrong_answer_despite_completed_reasoning"


def _generate_report(
    output_json: str,
    total_questions: int,
    processed_new: int,
    existing_count: int,
    start_time: float,
    end_time: float,
    report_path: str | None = None,
) -> Dict[str, Any]:
    with open(output_json, encoding="utf-8") as f:
        results = json.load(f)
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        results[idx] = _compute_prompt_succ_per_result(item)
        results[idx] = _score_result(results[idx])
    metrics = _evaluate_results(results)
    error_analysis = _build_error_analysis(results)
    token_stats = get_token_stats().get_stats()
    total_time = end_time - start_time
    report = {
        "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "total_questions": total_questions,
            "processed_new": processed_new,
            "existing_count": existing_count,
            "total_in_output": len(results),
        },
        "timing": {
            "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_seconds": round(total_time, 2),
            "total_minutes": round(total_time / 60, 2),
            "avg_seconds_per_question": round(total_time / processed_new, 3) if processed_new > 0 else 0,
        },
        "token_usage": {
            "total_prompt_tokens": token_stats["total_prompt_tokens"],
            "total_completion_tokens": token_stats["total_completion_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "api_call_count": token_stats["api_call_count"],
            "cache_hit_count": token_stats["cache_hit_count"],
            "api_call_without_cache": token_stats["api_call_without_cache"],
        },
        "evaluation": metrics,
        "correctness": {
            "correct_count": error_analysis["correct_count"],
            "wrong_count": error_analysis["wrong_count"],
            "accuracy": round(error_analysis["correct_count"] / len(results), 3) if results else 0.0,
        },
        "error_analysis": error_analysis,
        "output_file": output_json,
    }
    if report_path is None:
        base, _ = os.path.splitext(output_json)
        report_path = base + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Report file: %s", report_path)
    return report


def run_on_file(input_json: str, output_json: str | None = None, max_workers: int = 4, max_loops: int = 8, generate_report: bool = True):
    reset_token_stats()
    start_time = time.time()
    questions = _load_questions(input_json)
    if not questions:
        logger.info("no valid questions found in %s", input_json)
        return
    if output_json is None:
        base, _ = os.path.splitext(input_json)
        output_json = base + "_evidence_aware_answers.json"
    existing = _load_existing_results(output_json)
    existing_by_id = existing["by_id"]
    existing_list = existing["list"]
    new_questions = [item for item in questions if (item.get("id") or item.get("question_id")) not in existing_by_id]
    try:
        retriever = MultiHypergraphRetriever(base_dir=dataset_db_dir())
        logger.info("initialized shared retriever")
    except Exception as exc:
        logger.warning("could not initialize retriever: %s", exc)
        retriever = None
    new_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_question, item, retriever, max_loops): item for item in new_questions}
        total = len(future_to_item)
        done = 0
        for fut in as_completed(future_to_item):
            src_item = future_to_item[fut]
            done += 1
            qid = src_item.get("id") or src_item.get("question_id")
            try:
                new_results.append(fut.result())
                status = "OK"
            except Exception as exc:
                new_results.append(
                    {"id": qid, "question": src_item.get("question"), "answers": src_item.get("answers", []), "error": str(exc)}
                )
                status = "ERR"
            logger.info("[%s] [%4d/%4d] ID=%s", status, done, total, qid)
    all_results = existing_list + new_results
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    end_time = time.time()
    if generate_report:
        _generate_report(output_json, len(questions), len(new_results), len(existing_list), start_time, end_time)


def main():
    parser = argparse.ArgumentParser(description="Run EHHR evidence-aware reasoning on a QA file.")
    parser.add_argument("--dataset", default="hotpot")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-loops", type=int, default=8)
    args = parser.parse_args()
    qa_file = Path(args.qa_file) if args.qa_file else dataset_raw_dir(args.dataset) / f"{args.dataset}_1000_sampled_qa.json"
    output = Path(args.output) if args.output else dataset_outputs_dir(args.dataset) / f"{args.dataset}_1000_ehhr.json"
    run_on_file(str(qa_file), str(output), max_workers=args.max_workers, max_loops=args.max_loops)


if __name__ == "__main__":
    main()
