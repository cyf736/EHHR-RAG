import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ehhr_rag.config import dataset_db_dir, prompt_dir
from ehhr_rag.llm import generate_from_prompt_template
from ehhr_rag.logging_utils import logger

QUESTION_STATE_NOT_READY = "not-ready"
QUESTION_STATE_READY = "ready"

SUPPORT_NONE = "none"
SUPPORT_LOW = "low"
SUPPORT_HIGH = "high"

MODE_NONE = "none"
MODE_ENTITY = "Entity"
MODE_SEMANTIC = "Semantic"
MODE_THEME = "Theme"

SUPPORT_PRIORITY = {
    SUPPORT_NONE: 0,
    SUPPORT_LOW: 1,
    SUPPORT_HIGH: 2,
}


@dataclass
class EvidenceAwareSubQuestion:
    id: str
    text: str
    pred: List[str] = field(default_factory=list)
    ans: str = QUESTION_STATE_NOT_READY
    sup: str = SUPPORT_NONE
    mode: str = MODE_ENTITY
    hist: List[str] = field(default_factory=list)
    value: str = ""
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalDocument:
    doc_id: str
    text: str
    mode: str
    source_question_id: str
    source_question_text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    evidence_id: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceAwareReasoner:
    def __init__(
        self,
        original_query: str,
        retriever=None,
        base_dir: Optional[str] = None,
        max_loops: int = 8,
        retrieval_pool_capacity: int = 30,
        max_llm_retries: int = 3,
        final_low_answer_patience: int = 2,
        max_helper_questions: int = 2,
    ):
        self.original_query = original_query
        self.prompt_template_dir = str(prompt_dir())
        self.max_loops = max_loops
        self.retrieval_pool_capacity = retrieval_pool_capacity
        self.max_llm_retries = max_llm_retries
        self.final_low_answer_patience = final_low_answer_patience
        self.max_helper_questions = max_helper_questions

        self.retriever = retriever
        if self.retriever is None:
            from ehhr_rag.retrieval.retriever import MultiHypergraphRetriever

            if base_dir is None:
                base_dir = dataset_db_dir()
            self.retriever = MultiHypergraphRetriever(base_dir=base_dir)

        self.sub_questions: List[EvidenceAwareSubQuestion] = []
        self.retrieval_pool: List[RetrievalDocument] = []
        self.evidence_pool: List[EvidenceItem] = []
        self.loop_history: List[Dict[str, Any]] = []
        self.node_prompts: Dict[str, List[Dict[str, str]]] = {}
        self.final_answer: str = "unknown"

        self._doc_counter = 0
        self._evidence_counter = 0
        self._sub_question_counter = 0
        self._helper_question_counter = 0

    def execute(self) -> str:
        self.sub_questions = self._initialize_sub_questions()
        logger.info("Evidence-aware reasoning initialized with %s sub-questions", len(self.sub_questions))
        final_low_streak = 0

        for loop_idx in range(1, self.max_loops + 1):
            logger.info("Evidence-aware reasoning loop %s/%s", loop_idx, self.max_loops)

            selected = self._select_target_sub_questions()
            if selected:
                self._update_retrieval_modes(selected)
                retrieved_docs = self._retrieve_for_sub_questions(selected)
                self._append_retrieval_docs(retrieved_docs)
                extracted_evidences = self._extract_evidence_from_docs(retrieved_docs)
                if extracted_evidences:
                    self._append_evidences(extracted_evidences)
                    self._update_sub_question_evidence_view()
                    if not self._all_sub_questions_ready():
                        self._update_sub_question_readiness()
                else:
                    logger.info("No new evidence extracted in loop %s; skip state updates", loop_idx)
            else:
                logger.info("No ready low-support sub-questions were selected in loop %s", loop_idx)

            self.loop_history.append(
                {
                    "loop": loop_idx,
                    "selected_question_ids": [sq.id for sq in selected],
                    "retrieval_pool_size": len(self.retrieval_pool),
                    "evidence_pool_size": len(self.evidence_pool),
                    "sub_questions": [sq.to_dict() for sq in self.sub_questions],
                }
            )

            if self._is_final_question_ready():
                answer = self._answer_final_question()
                if self._is_valid_answer(answer):
                    self.final_answer = answer
                    return self.final_answer
                fallback_answer = self._get_final_subquestion_value()
                if self._is_valid_answer(fallback_answer):
                    self.final_answer = fallback_answer
                    return self.final_answer
                if self._all_sub_questions_ready_high():
                    new_sub_questions = self._generate_additional_sub_questions()
                    if new_sub_questions:
                        self._merge_new_sub_questions(new_sub_questions)
                        continue

            if self._is_final_question_ready_low():
                final_low_streak += 1
            else:
                final_low_streak = 0

            if final_low_streak >= self.final_low_answer_patience:
                answer = self._answer_final_question()
                if self._is_valid_answer(answer):
                    self.final_answer = answer
                    return self.final_answer
                fallback_answer = self._get_final_subquestion_value()
                if self._is_valid_answer(fallback_answer):
                    self.final_answer = fallback_answer
                    return self.final_answer

            if not self._is_final_question_ready():
                new_sub_questions = self._generate_additional_sub_questions()
                if new_sub_questions:
                    self._merge_new_sub_questions(new_sub_questions)

        answer = self._answer_final_question()
        if self._is_valid_answer(answer):
            self.final_answer = answer
            return self.final_answer

        answer = self._answer_final_question_flexible()
        if self._is_valid_answer(answer):
            self.final_answer = answer
            return self.final_answer

        fallback_answer = self._get_final_subquestion_value()
        if self._is_valid_answer(fallback_answer):
            self.final_answer = fallback_answer
            return self.final_answer

        self.final_answer = "unknown"
        return self.final_answer

    def get_results(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "final_answer": self.final_answer,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "retrieval_pool": [doc.to_dict() for doc in self.retrieval_pool],
            "evidence_pool": [evi.to_dict() for evi in self.evidence_pool],
            "node_prompts": self.node_prompts,
            "loop_history": self.loop_history,
            "graph_dict": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "nodes": {
                sq.id: {
                    "node_id": sq.id,
                    "question": sq.text,
                    "input_source": "<SEP>".join(sq.pred),
                    "output_variable": sq.value,
                    "node_type": "sub_question",
                    "is_final": sq.is_final,
                    "is_aggregate": False,
                    "answer": sq.value,
                    "dependencies": [],
                    "query_dict": {
                        "entity_list": sq.pred,
                        "sentence_text": sq.text,
                    },
                    "state": {
                        "ans": sq.ans,
                        "sup": sq.sup,
                        "mode": sq.mode,
                        "hist": sq.hist,
                    },
                }
                for sq in self.sub_questions
            },
            "execution_order": [sq.id for sq in self.sub_questions],
            "edges": [],
        }

    def _initialize_sub_questions(self) -> List[EvidenceAwareSubQuestion]:
        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "initialize_sub_questions.txt")
        prompt_input = [self.original_query]
        items = self._call_json_prompt(node_id="initializer", prompt_path=prompt_path, prompt_input=prompt_input, expect_list=True)

        sub_questions: List[EvidenceAwareSubQuestion] = []
        for item in items:
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            question_id = str(item.get("id", "") or "").strip() or self._next_sub_question_id()
            pred = self._normalize_predicates(item.get("pred", []), owner_id=question_id)
            ans = self._normalize_question_state(item.get("ans", QUESTION_STATE_NOT_READY))
            is_final = bool(item.get("is_final", False))
            mode = MODE_ENTITY if pred else MODE_THEME

            sub_questions.append(
                EvidenceAwareSubQuestion(
                    id=question_id,
                    text=text,
                    pred=pred,
                    ans=ans,
                    sup=SUPPORT_NONE,
                    mode=mode,
                    hist=[],
                    value="",
                    is_final=is_final,
                )
            )

        if not sub_questions:
            fallback_id = self._next_sub_question_id()
            sub_questions.append(
                EvidenceAwareSubQuestion(
                    id=fallback_id,
                    text=self.original_query,
                    pred=[],
                    ans=QUESTION_STATE_READY,
                    sup=SUPPORT_NONE,
                    mode=MODE_THEME,
                    hist=[],
                    value="",
                    is_final=True,
                )
            )
        elif not any(sq.ans == QUESTION_STATE_READY for sq in sub_questions):
            sub_questions[0].ans = QUESTION_STATE_READY
            logger.info("No ready sub-question was produced during initialization; force %s to ready", sub_questions[0].id)

        for sq in sub_questions:
            self._sub_question_counter = max(self._sub_question_counter, self._safe_int_suffix(sq.id))

        return sub_questions

    def _select_target_sub_questions(self) -> List[EvidenceAwareSubQuestion]:
        candidates = [
            sq
            for sq in self.sub_questions
            if sq.ans == QUESTION_STATE_READY and sq.sup in {SUPPORT_NONE, SUPPORT_LOW}
            and not self._is_sub_question_retrieval_exhausted(sq)
        ]
        if not candidates:
            return []

        min_support = min(SUPPORT_PRIORITY.get(sq.sup, 99) for sq in candidates)
        return [sq for sq in candidates if SUPPORT_PRIORITY.get(sq.sup, 99) == min_support]

    def _update_retrieval_modes(self, selected: Sequence[EvidenceAwareSubQuestion]) -> None:
        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "select_retrieval_mode.txt")
        prompt_input = [
            self.original_query,
            json.dumps([sq.to_dict() for sq in selected], ensure_ascii=False, indent=2),
            json.dumps([sq.to_dict() for sq in self.sub_questions], ensure_ascii=False, indent=2),
        ]
        mode_items = self._call_json_prompt(node_id="mode_selector", prompt_path=prompt_path, prompt_input=prompt_input, expect_list=True)

        selected_by_id = {sq.id: sq for sq in selected}
        for item in mode_items:
            question_id = str(item.get("id", "") or "").strip()
            sq = selected_by_id.get(question_id)
            if sq is None:
                continue
            chosen_mode = self._choose_retrieval_mode_for_selected(sq=sq, proposed_mode=item.get("mode", sq.mode))
            sq.mode = chosen_mode
            if chosen_mode not in sq.hist:
                sq.hist.append(chosen_mode)

        for sq in selected:
            if sq.mode == MODE_NONE:
                chosen_mode = self._choose_retrieval_mode_for_selected(sq=sq, proposed_mode=MODE_NONE)
                sq.mode = chosen_mode
                if chosen_mode not in sq.hist:
                    sq.hist.append(chosen_mode)

    def _retrieve_for_sub_questions(self, selected: Sequence[EvidenceAwareSubQuestion]) -> List[RetrievalDocument]:
        all_docs: List[RetrievalDocument] = []
        for sq in selected:
            if sq.mode == MODE_NONE:
                continue
            docs = self._retrieve_single_sub_question(sq)
            all_docs.extend(docs)
        return all_docs

    def _retrieve_single_sub_question(self, sq: EvidenceAwareSubQuestion) -> List[RetrievalDocument]:
        query_dict = {
            "entity_list": sq.pred,
            "sentence_text": sq.text,
            "atomic_text": self._build_atomic_text(sq.text, sq.pred),
            "seed_sentence_ids": [doc.doc_id for doc in self.retrieval_pool if doc.doc_id.startswith("<fact>")],
        }

        docs: List[RetrievalDocument] = []
        if sq.mode == MODE_ENTITY:
            result = asyncio.run(self.retriever.query_predicate_structure(query_dict))
            sentence_ids = list(result.get("sentence_ids", []) or [])
            score_map = dict(result.get("sentence_scores", {}) or {})
            for sentence_id in sentence_ids:
                fact_data = asyncio.run(self.retriever.fact_vdb.get_by_id(sentence_id))
                if not fact_data:
                    continue
                text = str(fact_data.get("text", "") or "").strip()
                if not text:
                    continue
                docs.append(
                    RetrievalDocument(
                        doc_id=sentence_id,
                        text=text,
                        mode=MODE_ENTITY,
                        source_question_id=sq.id,
                        source_question_text=sq.text,
                        meta={"score": float(score_map.get(sentence_id, 0.0) or 0.0)},
                    )
                )
        elif sq.mode == MODE_SEMANTIC:
            result = asyncio.run(self.retriever.query_sentence_expansion(query_dict))
            clue_texts = list(result.get("clue_texts", []) or [])
            clue_scores = dict(result.get("clue_scores", {}) or {})
            for clue_text in clue_texts:
                cleaned = str(clue_text or "").strip()
                if not cleaned:
                    continue
                doc_id = self._next_doc_id("semantic")
                docs.append(
                    RetrievalDocument(
                        doc_id=doc_id,
                        text=cleaned,
                        mode=MODE_SEMANTIC,
                        source_question_id=sq.id,
                        source_question_text=sq.text,
                        meta={"score": float(clue_scores.get(cleaned, 0.0) or 0.0)},
                    )
                )
        elif sq.mode == MODE_THEME:
            result = asyncio.run(self.retriever.query_topic_expansion(query_dict))
            chunk_ids = list(result.get("chunk_ids", []) or [])
            score_map = dict(result.get("chunk_scores", {}) or {})
            for chunk_id in chunk_ids:
                chunk_data = asyncio.run(self.retriever.chunk_vdb.get_by_id(chunk_id))
                if not chunk_data:
                    continue
                chunk_text = str(chunk_data.get("text", "") or "").strip()
                chunk_title = str(chunk_data.get("document_title", "") or "").strip()
                if not chunk_text:
                    continue
                text = f"[{chunk_title}] {chunk_text}" if chunk_title else chunk_text
                docs.append(
                    RetrievalDocument(
                        doc_id=chunk_id,
                        text=text,
                        mode=MODE_THEME,
                        source_question_id=sq.id,
                        source_question_text=sq.text,
                        meta={"score": float(score_map.get(chunk_id, 0.0) or 0.0)},
                    )
                )

        return self._dedupe_retrieval_docs(docs)

    def _append_retrieval_docs(self, docs: Sequence[RetrievalDocument]) -> None:
        if not docs:
            return

        existing_keys = {(doc.doc_id, doc.source_question_id, doc.text) for doc in self.retrieval_pool}
        for doc in docs:
            key = (doc.doc_id, doc.source_question_id, doc.text)
            if key not in existing_keys:
                self.retrieval_pool.append(doc)
                existing_keys.add(key)

        self._prune_retrieval_pool()

    def _extract_evidence_from_docs(self, docs: Sequence[RetrievalDocument]) -> List[EvidenceItem]:
        if not docs:
            return []

        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "extract_evidence.txt")
        prompt_input = [
            self.original_query,
            json.dumps([sq.to_dict() for sq in self.sub_questions], ensure_ascii=False, indent=2),
            json.dumps([doc.text for doc in docs], ensure_ascii=False, indent=2),
            json.dumps([evi.text for evi in self.evidence_pool], ensure_ascii=False, indent=2),
        ]
        evidence_items = self._call_json_prompt(node_id="evidence_extractor", prompt_path=prompt_path, prompt_input=prompt_input, expect_list=True)

        evidences: List[EvidenceItem] = []
        for item in evidence_items:
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            evidence_id = str(item.get("evidence_id", "") or "").strip() or self._next_evidence_id()
            evidences.append(EvidenceItem(evidence_id=evidence_id, text=text))
        return evidences

    def _append_evidences(self, evidences: Sequence[EvidenceItem]) -> None:
        if not evidences:
            return
        existing_keys = {e.text for e in self.evidence_pool}
        for evidence in evidences:
            key = evidence.text
            if key not in existing_keys:
                self.evidence_pool.append(evidence)
                existing_keys.add(key)

    def _update_sub_question_evidence_view(self) -> None:
        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "update_subquestion_evidence_view.txt")
        prompt_input = [
            self.original_query,
            json.dumps([sq.to_dict() for sq in self.sub_questions], ensure_ascii=False, indent=2),
            json.dumps([evi.text for evi in self.evidence_pool], ensure_ascii=False, indent=2),
        ]
        state_items = self._call_json_prompt(node_id="evidence_view_updater", prompt_path=prompt_path, prompt_input=prompt_input, expect_list=True)

        sub_questions_by_id = {sq.id: sq for sq in self.sub_questions}
        for item in state_items:
            question_id = str(item.get("id", "") or "").strip()
            sq = sub_questions_by_id.get(question_id)
            if sq is None:
                continue
            sq.sup = self._normalize_support(item.get("sup", sq.sup))
            sq.value = str(item.get("value", sq.value) or "").strip()
            sq.pred = self._normalize_predicates(item.get("pred", sq.pred), owner_id=sq.id)

    def _update_sub_question_readiness(self) -> None:
        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "update_subquestion_readiness.txt")
        prompt_input = [
            self.original_query,
            json.dumps([sq.to_dict() for sq in self.sub_questions], ensure_ascii=False, indent=2),
            json.dumps([evi.text for evi in self.evidence_pool], ensure_ascii=False, indent=2),
        ]
        state_items = self._call_json_prompt(node_id="readiness_updater", prompt_path=prompt_path, prompt_input=prompt_input, expect_list=True)

        sub_questions_by_id = {sq.id: sq for sq in self.sub_questions}
        for item in state_items:
            question_id = str(item.get("id", "") or "").strip()
            sq = sub_questions_by_id.get(question_id)
            if sq is None:
                continue
            new_ans = self._normalize_question_state(item.get("ans", sq.ans))
            new_text = str(item.get("text", sq.text) or "").strip()
            new_pred = self._normalize_predicates(item.get("pred", sq.pred), owner_id=sq.id)

            if sq.ans == QUESTION_STATE_READY:
                sq.ans = QUESTION_STATE_READY
            else:
                if new_ans == QUESTION_STATE_READY:
                    if new_text:
                        sq.text = new_text
                    if new_pred:
                        sq.pred = new_pred
                sq.ans = new_ans

    def _generate_additional_sub_questions(self) -> List[EvidenceAwareSubQuestion]:
        if self._helper_question_counter >= self.max_helper_questions:
            return []
        if not self.retrieval_pool:
            return []
        active_ready_nodes = [sq for sq in self.sub_questions if sq.ans == QUESTION_STATE_READY and sq.sup in {SUPPORT_NONE, SUPPORT_LOW}]
        if active_ready_nodes and not all(self._is_sub_question_retrieval_exhausted(sq) for sq in active_ready_nodes):
            return []

        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "generate_additional_subquestions.txt")
        prompt_input = [
            self.original_query,
            json.dumps([sq.to_dict() for sq in self.sub_questions], ensure_ascii=False, indent=2),
            json.dumps([evi.to_dict() for evi in self.evidence_pool], ensure_ascii=False, indent=2),
        ]
        items = self._call_json_prompt(node_id="subquestion_generator", prompt_path=prompt_path, prompt_input=prompt_input, expect_list=True)

        new_sub_questions: List[EvidenceAwareSubQuestion] = []
        for item in items:
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            normalized_text = self._normalize_text(text)
            if any(self._normalize_text(sq.text) == normalized_text for sq in self.sub_questions + new_sub_questions):
                continue
            new_question_id = self._next_helper_question_id()
            pred = self._normalize_predicates(item.get("pred", []), owner_id=new_question_id)
            if not pred or any(self._is_placeholder_anchor(p) for p in pred):
                continue
            mode = MODE_ENTITY if pred else MODE_THEME
            new_sub_questions.append(
                EvidenceAwareSubQuestion(
                    id=new_question_id,
                    text=text,
                    pred=pred,
                    ans=QUESTION_STATE_READY,
                    sup=SUPPORT_NONE,
                    mode=mode,
                    hist=[],
                    value="",
                    is_final=False,
                )
            )
        return new_sub_questions

    def _is_sub_question_retrieval_exhausted(self, sq: EvidenceAwareSubQuestion) -> bool:
        retrievable_modes = {MODE_ENTITY, MODE_SEMANTIC, MODE_THEME}
        tried_modes = {mode for mode in sq.hist if mode in retrievable_modes}
        return tried_modes == retrievable_modes

    def _merge_new_sub_questions(self, new_sub_questions: Sequence[EvidenceAwareSubQuestion]) -> None:
        if not new_sub_questions:
            return
        self.sub_questions.extend(new_sub_questions)
        logger.info("Added %s new sub-questions", len(new_sub_questions))

    def _is_final_question_ready(self) -> bool:
        return any(sq.is_final and sq.ans == QUESTION_STATE_READY and sq.sup == SUPPORT_HIGH for sq in self.sub_questions)

    def _is_final_question_ready_low(self) -> bool:
        return any(sq.is_final and sq.ans == QUESTION_STATE_READY and sq.sup == SUPPORT_LOW for sq in self.sub_questions)

    def _all_sub_questions_ready(self) -> bool:
        return bool(self.sub_questions) and all(sq.ans == QUESTION_STATE_READY for sq in self.sub_questions)

    def _all_sub_questions_ready_high(self) -> bool:
        return bool(self.sub_questions) and all(sq.ans == QUESTION_STATE_READY and sq.sup == SUPPORT_HIGH for sq in self.sub_questions)

    def _get_final_subquestion_value(self) -> str:
        for sq in self.sub_questions:
            if sq.is_final:
                value = str(sq.value or "").strip()
                if value:
                    return value
        return ""

    def _answer_final_question(self) -> str:
        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "answer_final_question.txt")
        prompt_input = [
            self.original_query,
            json.dumps([sq.to_dict() for sq in self.sub_questions], ensure_ascii=False, indent=2),
            json.dumps([evi.text for evi in self.evidence_pool], ensure_ascii=False, indent=2),
            json.dumps([doc.text for doc in self.retrieval_pool], ensure_ascii=False, indent=2),
        ]
        raw = self._call_text_prompt(node_id="final_answer", prompt_path=prompt_path, prompt_input=prompt_input)
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        parsed = self._extract_json(raw, expect_list=False, allow_empty=True)
        if isinstance(parsed, dict):
            return str(parsed.get("answer", "") or "").strip() or "unknown"
        return "unknown"

    def _answer_final_question_flexible(self) -> str:
        prompt_path = os.path.join(self.prompt_template_dir, "evidence_aware", "answer_final_question_flexible.txt")
        prompt_input = [
            self.original_query,
            json.dumps([evi.text for evi in self.evidence_pool], ensure_ascii=False, indent=2),
            json.dumps([doc.text for doc in self.retrieval_pool], ensure_ascii=False, indent=2),
        ]
        raw = self._call_text_prompt(node_id="final_answer_flexible", prompt_path=prompt_path, prompt_input=prompt_input)
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        parsed = self._extract_json(raw, expect_list=False, allow_empty=True)
        if isinstance(parsed, dict):
            return str(parsed.get("answer", "") or "").strip() or "unknown"
        return "unknown"

    def _prune_retrieval_pool(self) -> None:
        if len(self.retrieval_pool) <= self.retrieval_pool_capacity:
            return

        if not self.evidence_pool:
            self.retrieval_pool = self.retrieval_pool[-self.retrieval_pool_capacity:]
            return

        doc_texts = [doc.text for doc in self.retrieval_pool]
        evidence_texts = [evi.text for evi in self.evidence_pool if evi.text.strip()]
        if not evidence_texts:
            self.retrieval_pool = self.retrieval_pool[-self.retrieval_pool_capacity:]
            return

        doc_embeddings = self._embed_texts(doc_texts)
        evidence_embeddings = self._embed_texts(evidence_texts)
        if doc_embeddings is None or evidence_embeddings is None:
            self.retrieval_pool = self.retrieval_pool[-self.retrieval_pool_capacity:]
            return

        scores: List[tuple[int, float]] = []
        for idx, doc_vec in enumerate(doc_embeddings):
            best = 0.0
            for evidence_vec in evidence_embeddings:
                sim = self._cosine_similarity(doc_vec, evidence_vec)
                if sim > best:
                    best = sim
            scores.append((idx, best))

        keep_indices = {
            idx
            for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)[: self.retrieval_pool_capacity]
        }
        self.retrieval_pool = [doc for idx, doc in enumerate(self.retrieval_pool) if idx in keep_indices]

    def _embed_texts(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        cleaned = [str(text or "").strip() for text in texts]
        cleaned = [text for text in cleaned if text]
        if not cleaned:
            return None
        try:
            result = asyncio.run(self.retriever.fact_vdb.embedding_func(cleaned))
            embeddings = result.get("embeddings", [])
            if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                return embeddings.astype(np.float32)
            if embeddings:
                return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            logger.warning("Embedding texts for retrieval pruning failed: %s", exc)
        return None

    def _dedupe_retrieval_docs(self, docs: Sequence[RetrievalDocument]) -> List[RetrievalDocument]:
        seen = set()
        deduped: List[RetrievalDocument] = []
        for doc in docs:
            key = (doc.doc_id, doc.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    def _normalize_support(self, value: Any) -> str:
        value_str = str(value or "").strip()
        if value_str in {SUPPORT_NONE, SUPPORT_LOW, SUPPORT_HIGH}:
            return value_str
        return SUPPORT_NONE

    def _normalize_question_state(self, value: Any) -> str:
        value_str = str(value or "").strip()
        if value_str in {QUESTION_STATE_READY, QUESTION_STATE_NOT_READY}:
            return value_str
        return QUESTION_STATE_NOT_READY

    def _normalize_mode(self, value: Any, predicates: Sequence[str]) -> str:
        value_str = str(value or "").strip()
        if value_str in {MODE_NONE, MODE_ENTITY, MODE_SEMANTIC, MODE_THEME}:
            return value_str
        return MODE_ENTITY if predicates else MODE_THEME

    def _choose_retrieval_mode_for_selected(self, sq: EvidenceAwareSubQuestion, proposed_mode: Any) -> str:
        proposed = self._normalize_mode(proposed_mode, sq.pred)
        retrievable_modes = [MODE_ENTITY, MODE_SEMANTIC, MODE_THEME]

        if proposed in retrievable_modes and proposed not in sq.hist:
            return proposed

        preferred_order: List[str] = [MODE_ENTITY, MODE_SEMANTIC, MODE_THEME]
        for mode in preferred_order:
            if mode not in sq.hist:
                return mode

        return preferred_order[-1]

    def _normalize_predicates(self, value: Any, owner_id: str = "") -> List[str]:
        if isinstance(value, list):
            items = value
        elif isinstance(value, str):
            items = [part.strip() for part in value.split(",")]
        else:
            items = []

        normalized = []
        for item in items:
            cleaned = str(item or "").strip()
            cleaned = self._normalize_placeholder_anchor(cleaned, owner_id=owner_id)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def _normalize_placeholder_anchor(self, anchor: str, owner_id: str = "") -> str:
        text = str(anchor or "").strip()
        if not text:
            return ""

        prefixed_match = re.fullmatch(r"(sq_\d+)_x(\d+):([A-Za-z0-9_\- ]+)", text)
        if prefixed_match:
            return f"{prefixed_match.group(1)}_x{prefixed_match.group(2)}:{prefixed_match.group(3).strip()}"

        bare_match = re.fullmatch(r"x(\d+):([A-Za-z0-9_\- ]+)", text)
        if bare_match:
            owner = owner_id.strip() if owner_id else "sq_unknown"
            return f"{owner}_x{bare_match.group(1)}:{bare_match.group(2).strip()}"

        return text

    def _is_placeholder_anchor(self, anchor: str) -> bool:
        text = str(anchor or "").strip()
        return bool(
            re.fullmatch(r"(sq_\d+|hq_\d+)_x\d+:[A-Za-z0-9_\- ]+", text)
            or re.fullmatch(r"x\d+:[A-Za-z0-9_\- ]+", text)
        )

    def _build_atomic_text(self, question: str, predicates: Sequence[str]) -> str:
        text = str(question or "").lower()
        for pred in predicates:
            text = text.replace(str(pred).lower(), " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or str(question or "").strip()

    def _extract_json(self, text: str, expect_list: bool, allow_empty: bool = False) -> Any:
        content = str(text or "").strip()
        if not content:
            return [] if expect_list else {}

        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced_match:
            content = fenced_match.group(1).strip()

        parsed = self._try_parse_json(content)
        if parsed is None:
            bracket_match = re.search(r"\[\s*.*\]", content, re.DOTALL)
            brace_match = re.search(r"\{\s*.*\}", content, re.DOTALL)
            if expect_list and bracket_match:
                parsed = self._try_parse_json(bracket_match.group(0))
            elif (not expect_list) and brace_match:
                parsed = self._try_parse_json(brace_match.group(0))
            elif bracket_match:
                parsed = self._try_parse_json(bracket_match.group(0))
            elif brace_match:
                parsed = self._try_parse_json(brace_match.group(0))

        if parsed is None:
            if allow_empty:
                return [] if expect_list else {}
            raise ValueError(f"Cannot parse JSON from LLM output: {text}")

        if expect_list:
            if isinstance(parsed, list):
                return parsed
            if allow_empty:
                return []
            raise ValueError(f"Expected JSON list but got: {type(parsed).__name__}")

        if isinstance(parsed, dict):
            return parsed
        if allow_empty:
            return {}
        raise ValueError(f"Expected JSON object but got: {type(parsed).__name__}")

    def _call_json_prompt(self, node_id: str, prompt_path: str, prompt_input: Sequence[str], expect_list: bool) -> Any:
        last_error: Optional[Exception] = None
        retry_instruction = ""
        for attempt in range(1, self.max_llm_retries + 1):
            actual_input = list(prompt_input)
            if retry_instruction:
                actual_input = list(actual_input) + [retry_instruction]
            raw = generate_from_prompt_template(actual_input, prompt_path, use_cache=False)
            self._record_prompt(node_id, prompt_path, actual_input, raw)
            try:
                return self._extract_json(raw, expect_list=expect_list)
            except Exception as exc:
                last_error = exc
                retry_instruction = (
                    "Previous output could not be parsed or violated the required schema. "
                    "Retry and return valid JSON only. Do not add commentary, markdown fences, or extra text."
                )
                logger.warning("%s prompt parse failed on attempt %s/%s: %s", node_id, attempt, self.max_llm_retries, exc)
        if last_error is not None:
            raise last_error
        return [] if expect_list else {}

    def _call_text_prompt(self, node_id: str, prompt_path: str, prompt_input: Sequence[str]) -> str:
        raw = generate_from_prompt_template(prompt_input, prompt_path, use_cache=False)
        self._record_prompt(node_id, prompt_path, prompt_input, raw)
        return raw

    def _try_parse_json(self, text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r",(\s*[}\]])", r"\1", text)
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None

    def _record_prompt(self, node_id: str, prompt_path: str, prompt_input: Sequence[str], response: str) -> None:
        self.node_prompts.setdefault(node_id, []).append(
            {
                "template": os.path.basename(prompt_path),
                "prompt_input": "\n\n".join(str(item) for item in prompt_input),
                "response": response,
            }
        )

    def _next_doc_id(self, prefix: str) -> str:
        self._doc_counter += 1
        return f"<{prefix}-doc-{self._doc_counter}>"

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"evidence_{self._evidence_counter}"

    def _next_sub_question_id(self) -> str:
        self._sub_question_counter += 1
        return f"sq_{self._sub_question_counter}"

    def _next_helper_question_id(self) -> str:
        self._helper_question_counter += 1
        return f"hq_{self._helper_question_counter}"

    def _safe_int_suffix(self, question_id: str) -> int:
        match = re.search(r"(\d+)$", str(question_id or ""))
        if not match:
            return 0
        return int(match.group(1))

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
        return cleaned

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def _is_valid_answer(self, answer: str) -> bool:
        answer_text = str(answer or "").strip()
        if not answer_text:
            return False
        answer_lower = answer_text.lower()
        invalid_exact = {"none", "cannot answer", "cannot find", "don't know", "unknown", "no information"}
        if answer_lower in invalid_exact:
            return False
        return True
