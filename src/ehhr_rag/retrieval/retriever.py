import asyncio
import math
import os
import re
import threading
from typing import Any

import numpy as np
import torch

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None

from ehhr_rag.config import dataset_db_dir
from ehhr_rag.logging_utils import logger
from ehhr_rag.storage.json_kv_storage import JsonKVStorage
from ehhr_rag.storage.nano_vector_storage import NanoVectorStorage
from ehhr_rag.storage.networkx_graph_storage import NetworkXStorage


class MultiHypergraphRetriever:
    def __init__(self, base_dir: str | None = None):
        if base_dir is None:
            base_dir = str(dataset_db_dir())
        self.base_dir = base_dir

        self.embedding_model_name = "BAAI/bge-m3"

        self.entity_vdb = NanoVectorStorage(namespace="layer_micro_entities", base_dir=base_dir)
        self.predicate_vdb = NanoVectorStorage(namespace="layer_micro_predicates", base_dir=base_dir)
        self.fact_vdb = NanoVectorStorage(namespace="layer_micro_facts", base_dir=base_dir)
        self.micro_kv = JsonKVStorage(namespace="layer_micro", base_dir=base_dir)

        self.topic_vdb = NanoVectorStorage(namespace="layer_macro_topics", base_dir=base_dir)
        self.chunk_vdb = NanoVectorStorage(namespace="layer_macro_chunks", base_dir=base_dir)

        for vdb in [self.entity_vdb, self.predicate_vdb, self.fact_vdb, self.topic_vdb, self.chunk_vdb]:
            vdb.embedding_model = self.embedding_model_name

        self.micro_graph = NetworkXStorage(namespace="layer_micro", base_dir=base_dir)
        self.macro_graph = NetworkXStorage(namespace="layer_macro", base_dir=base_dir)

        self.entity_top_k = 10
        self.fact_retrieve_k = 15
        self.fact_return_k = 5

        self.topic_top_k = 10
        self.chunk_top_k = 15
        self.topic_chunk_top_k = 5

        self.enable_fact_rerank = True
        self.fact_rerank_top_m = 50
        self.reranker_model_name = "BAAI/bge-reranker-v2-m3"
        self.reranker_use_fp16 = True
        self._reranker = None
        self._reranker_init_lock = threading.Lock()
        self._reranker_infer_lock = threading.Lock()
        self._chunk_sentence_index_cache: dict[str, dict[int, str]] | None = None
        self._fact_text_cache: dict[str, str] = {}
        self._entity_text_cache: dict[str, str] = {}
        self._predicate_text_cache: dict[str, str] = {}
        self._reranker_disabled = False

        logger.info("MultiHypergraphRetriever initialized, base_dir: %s", base_dir)
        logger.info("Retriever embedding backend: HuggingFace (%s)", self.embedding_model_name)

    async def query_predicate_structure(self, dict_query: dict[str, Any]) -> dict[str, Any]:
        """Entity-first retrieval with similarity expansion and hyperedge reranking."""
        entity_list = dict_query.get("entity_list", [])
        atomic_text = dict_query.get("atomic_text", "")
        sentence_text = str(dict_query.get("sentence_text", "") or "")
        similarity_threshold = 0.6
        use_rerank = bool(dict_query.get("use_rerank", self.enable_fact_rerank))
        rerank_top_m = int(dict_query.get("rerank_top_m", self.fact_rerank_top_m))

        cleaned_entities = [e.strip() for e in entity_list if str(e).strip()]
        logger.info(
            "Start predicate-structure retrieval: entity_list=%s, atomic_text=%s, theta=%s, use_rerank=%s, rerank_top_m=%s",
            cleaned_entities,
            atomic_text,
            similarity_threshold,
            use_rerank,
            rerank_top_m,
        )

        if not cleaned_entities:
            return {
                "sentence_ids": [],
                "sentence_scores": {},
                "related_entity_ids": [],
                "related_entity_scores": {},
                "related_predicate_ids": [],
                "related_predicate_scores": {},
            }

        seed_entity_ids = set()
        for entity in cleaned_entities:
            entity_results = await self.entity_vdb.query(query=entity, top_k=self.entity_top_k)
            for item in entity_results:
                entity_id = item.get("id")
                if entity_id:
                    seed_entity_ids.add(entity_id)

        expanded_entity_ids = await self._expand_entities_via_similarity_hyperedges(seed_entity_ids)
        candidate_entity_ids = seed_entity_ids | expanded_entity_ids

        candidate_entity_scores = await self._compute_candidate_entity_similarities(
            candidate_entity_ids=candidate_entity_ids,
            query_entities=cleaned_entities,
        )

        filtered_entity_scores = {
            entity_id: sim_score
            for entity_id, sim_score in candidate_entity_scores.items()
            if sim_score >= similarity_threshold
        }

        if not filtered_entity_scores:
            logger.info("Predicate-structure retrieval finished: no usable entities after thresholding")
            return {
                "sentence_ids": [],
                "sentence_scores": {},
                "related_entity_ids": [],
                "related_entity_scores": {},
                "related_predicate_ids": [],
                "related_predicate_scores": {},
            }

        fact_to_hit_entities = {}
        for entity_id in filtered_entity_scores:
            neighbors = await self.micro_graph.get_neighbors(entity_id)
            if not neighbors:
                continue
            hit_facts = [nid for nid in neighbors if nid.startswith("<fact>")]
            if not hit_facts:
                continue
            for fact_id in hit_facts:
                if fact_id not in fact_to_hit_entities:
                    fact_to_hit_entities[fact_id] = set()
                fact_to_hit_entities[fact_id].add(entity_id)

        if not fact_to_hit_entities:
            logger.info("Predicate-structure retrieval finished: no hit facts")
            return {
                "sentence_ids": [],
                "sentence_scores": {},
                "related_entity_ids": list(filtered_entity_scores.keys()),
                "related_entity_scores": filtered_entity_scores,
                "related_predicate_ids": [],
                "related_predicate_scores": {},
            }

        predicate_scores = {}
        fact_scores = {}
        query_predicate_embedding = await self._embed_text(atomic_text, self.predicate_vdb)
        predicate_embedding_cache: dict[str, np.ndarray] = {}

        for fact_id, hit_entities in fact_to_hit_entities.items():
            hit_entity_sim_sum = sum(filtered_entity_scores.get(ent_id, 0.0) for ent_id in hit_entities)

            predicate_similarity = 0.0
            fact_neighbors = await self.micro_graph.get_neighbors(fact_id)
            predicate_ids = [nid for nid in (fact_neighbors or []) if nid.startswith("<predicate>")]
            if predicate_ids and query_predicate_embedding is not None:
                predicate_id = predicate_ids[0]
                predicate_embedding = predicate_embedding_cache.get(predicate_id)
                if predicate_embedding is None:
                    predicate_text = await self._get_predicate_text_by_id(predicate_id)
                    if predicate_text:
                        predicate_embedding = await self._embed_text(predicate_text, self.predicate_vdb)
                        if predicate_embedding is not None:
                            predicate_embedding_cache[predicate_id] = predicate_embedding
                if predicate_embedding is not None:
                    predicate_similarity = self._cosine_similarity(query_predicate_embedding, predicate_embedding)
                    predicate_scores[predicate_id] = max(predicate_scores.get(predicate_id, 0.0), predicate_similarity)

            fact_scores[fact_id] = hit_entity_sim_sum * (1.0 + math.log10(1.0 + predicate_similarity))

        sorted_facts = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
        top_facts = sorted_facts[:self.fact_return_k]

        if use_rerank and sentence_text.strip() and sorted_facts:
            rerank_m = max(self.fact_return_k, rerank_top_m)
            rerank_candidates = sorted_facts[:rerank_m]
            reranked_facts = await self._rerank_fact_candidates(sentence_text, rerank_candidates)
            if reranked_facts:
                top_facts = reranked_facts[:self.fact_return_k]

        top_fact_ids = [fact_id for fact_id, _ in top_facts]
        logger.info("Predicate-structure retrieval finished: returned %s facts", len(top_fact_ids))

        return {
            "sentence_ids": top_fact_ids,
            "sentence_scores": dict(top_facts),
            "related_entity_ids": list(filtered_entity_scores.keys()),
            "related_entity_scores": filtered_entity_scores,
            "related_predicate_ids": list(predicate_scores.keys()),
            "related_predicate_scores": predicate_scores,
        }

    def _get_reranker(self):
        if self._reranker_disabled:
            return None
        if FlagReranker is None:
            logger.warning("FlagEmbedding is not installed; rerank is disabled.")
            return None
        if self._reranker is None:
            with self._reranker_init_lock:
                if self._reranker is None:
                    prefer_fp16 = bool(self.reranker_use_fp16 and torch.cuda.is_available())
                    try:
                        self._reranker = FlagReranker(self.reranker_model_name, use_fp16=prefer_fp16)
                    except Exception as e:
                        try:
                            self._reranker = FlagReranker(self.reranker_model_name, use_fp16=False)
                        except Exception as e2:
                            logger.warning("Reranker init failed, disabling rerank: %s | fallback: %s", e, e2)
                            self._reranker_disabled = True
                            return None
        return self._reranker

    def _reranker_compute_score(self, reranker, pairs: list[list[str]], normalize: bool = True):
        with self._reranker_infer_lock:
            return reranker.compute_score(pairs, normalize=normalize)

    async def _get_fact_text_by_id(self, fact_id: str) -> str:
        if fact_id in self._fact_text_cache:
            return self._fact_text_cache[fact_id]

        fact_data = await self.fact_vdb.get_by_id(fact_id)
        if fact_data:
            fact_text = str(fact_data.get("text", "") or "").strip()
            if fact_text:
                self._fact_text_cache[fact_id] = fact_text
                return fact_text

        graph_fact_data = await self.micro_graph.get_node(fact_id)
        if graph_fact_data:
            fact_text = str(graph_fact_data.get("text", "") or "").strip()
            if fact_text:
                self._fact_text_cache[fact_id] = fact_text
                return fact_text
        return ""

    async def _get_entity_text_by_id(self, entity_id: str) -> str:
        if entity_id in self._entity_text_cache:
            return self._entity_text_cache[entity_id]

        entity_data = await self.entity_vdb.get_by_id(entity_id)
        if entity_data:
            entity_text = str(entity_data.get("text", "") or "").strip()
            if entity_text:
                self._entity_text_cache[entity_id] = entity_text
                return entity_text

        graph_entity_data = await self.micro_graph.get_node(entity_id)
        if graph_entity_data:
            entity_text = str(graph_entity_data.get("text", "") or "").strip()
            if entity_text:
                self._entity_text_cache[entity_id] = entity_text
                return entity_text
        return ""

    async def _get_predicate_text_by_id(self, predicate_id: str) -> str:
        if predicate_id in self._predicate_text_cache:
            return self._predicate_text_cache[predicate_id]

        predicate_data = await self.predicate_vdb.get_by_id(predicate_id)
        if predicate_data:
            predicate_text = str(predicate_data.get("text", "") or "").strip()
            if predicate_text:
                self._predicate_text_cache[predicate_id] = predicate_text
                return predicate_text

        graph_predicate_data = await self.micro_graph.get_node(predicate_id)
        if graph_predicate_data:
            predicate_text = str(graph_predicate_data.get("text", "") or "").strip()
            if predicate_text:
                self._predicate_text_cache[predicate_id] = predicate_text
                return predicate_text
        return ""

    async def _rerank_fact_candidates(self, query_text: str, fact_candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        reranker = self._get_reranker()
        if reranker is None:
            return fact_candidates

        rerank_pairs = []
        rerank_fact_ids = []
        for fact_id, _ in fact_candidates:
            fact_text = await self._get_fact_text_by_id(fact_id)
            if not fact_text:
                continue
            rerank_pairs.append([query_text, fact_text])
            rerank_fact_ids.append(fact_id)

        if not rerank_pairs:
            return fact_candidates

        try:
            rerank_scores = self._reranker_compute_score(reranker, rerank_pairs, normalize=True)
            if isinstance(rerank_scores, (float, int)):
                rerank_scores = [float(rerank_scores)]
        except Exception as e:
            if "meta tensor" in str(e).lower():
                self._reranker_disabled = True
                self._reranker = None
            logger.warning("Rerank failed, falling back to base ordering: %s", e)
            return fact_candidates

        rerank_map = {}
        for fact_id, score in zip(rerank_fact_ids, rerank_scores):
            rerank_map[fact_id] = float(score)

        reranked = sorted(rerank_map.items(), key=lambda x: x[1], reverse=True)
        return reranked

    async def _expand_entities_via_similarity_hyperedges(self, seed_entity_ids: set[str]) -> set[str]:
        expanded_entity_ids = set()
        if not seed_entity_ids:
            return expanded_entity_ids

        for entity_id in seed_entity_ids:
            neighbors = await self.micro_graph.get_neighbors(entity_id)
            if not neighbors:
                continue
            similarity_hyperedges = [nid for nid in neighbors if nid.startswith("<similarity>")]
            for sim_id in similarity_hyperedges:
                sim_neighbors = await self.micro_graph.get_neighbors(sim_id)
                if not sim_neighbors:
                    continue
                for sim_neighbor in sim_neighbors:
                    if sim_neighbor.startswith("<entity>"):
                        expanded_entity_ids.add(sim_neighbor)
        return expanded_entity_ids

    async def _compute_candidate_entity_similarities(self, candidate_entity_ids: set[str], query_entities: list[str]) -> dict[str, float]:
        if not candidate_entity_ids or not query_entities:
            return {}

        query_embed_result = await self.entity_vdb.embedding_func(query_entities)
        query_embeddings = [np.asarray(emb, dtype=np.float32) for emb in query_embed_result.get("embeddings", [])]
        if not query_embeddings:
            return {}

        entity_id_text_pairs = []
        for entity_id in candidate_entity_ids:
            entity_text = await self._get_entity_text_by_id(entity_id)
            if entity_text:
                entity_id_text_pairs.append((entity_id, entity_text))

        entity_id_to_embedding = await self._embed_texts_by_id(entity_id_text_pairs, self.entity_vdb)
        if not entity_id_to_embedding:
            return {}

        candidate_scores = {}
        for entity_id, entity_embedding in entity_id_to_embedding.items():
            max_sim = max(self._cosine_similarity(query_emb, entity_embedding) for query_emb in query_embeddings)
            candidate_scores[entity_id] = max_sim

        return candidate_scores

    async def _embed_text(self, text: str, vdb: NanoVectorStorage) -> np.ndarray | None:
        if not text or not text.strip():
            return None
        try:
            embed_result = await vdb.embedding_func([text])
            embeddings = embed_result.get("embeddings", [])
            if embeddings is None:
                return None
            if isinstance(embeddings, np.ndarray):
                if embeddings.size == 0:
                    return None
            elif len(embeddings) == 0:
                return None
            return np.asarray(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning("Text embedding failed: %s", e)
            return None

    async def _embed_texts_by_id(self, id_text_pairs: list[tuple[str, str]], vdb: NanoVectorStorage) -> dict[str, np.ndarray]:
        if not id_text_pairs:
            return {}

        texts = [text for _, text in id_text_pairs]
        try:
            embed_result = await vdb.embedding_func(texts)
        except Exception as e:
            logger.warning("Batch text embedding failed: %s", e)
            return {}

        embeddings = embed_result.get("embeddings", [])
        if embeddings is None:
            return {}

        embedding_list = [np.asarray(emb, dtype=np.float32) for emb in embeddings]
        if not embedding_list:
            return {}

        valid_indices = embed_result.get("valid_indices", [])
        id_to_embedding: dict[str, np.ndarray] = {}

        if isinstance(valid_indices, list) and len(valid_indices) == len(embedding_list):
            for pos, local_idx in enumerate(valid_indices):
                if 0 <= int(local_idx) < len(id_text_pairs):
                    item_id = id_text_pairs[int(local_idx)][0]
                    id_to_embedding[item_id] = embedding_list[pos]
        else:
            for (item_id, _), embedding in zip(id_text_pairs, embedding_list):
                id_to_embedding[item_id] = embedding

        return id_to_embedding

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        vec_a_norm = np.linalg.norm(vec_a)
        vec_b_norm = np.linalg.norm(vec_b)
        if vec_a_norm == 0 or vec_b_norm == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (vec_a_norm * vec_b_norm))

    async def prompt_generate(
        self,
        sentence_ids: list[str] | None = None,
        clue_texts: list[str] | None = None,
        chunk_ids: list[str] | None = None,
    ) -> str:
        sentence_contents = []
        clue_contents = [str(text).strip() for text in (clue_texts or []) if str(text or '').strip()]
        chunk_contents = []

        for fact_id in sentence_ids or []:
            if str(fact_id).startswith('<fact>'):
                fact_data = await self.fact_vdb.get_by_id(fact_id)
                text = str((fact_data or {}).get('text', '') or '').strip()
                if text:
                    sentence_contents.append(text)

        for chunk_id in chunk_ids or []:
            if not str(chunk_id).startswith('<chunk>'):
                continue
            chunk_data = await self.chunk_vdb.get_by_id(chunk_id)
            if not chunk_data:
                continue
            text = str(chunk_data.get('text', '') or '').strip()
            title = str(chunk_data.get('document_title', '') or '').strip()
            if text:
                chunk_contents.append(f'[{title}]:{text}' if title else text)

        sections = []
        if sentence_contents:
            sections.append('The following are relevant fact knowledge:\n' +
                            '\n'.join(f'{i}. {text}' for i, text in enumerate(sentence_contents, 1)))
        if clue_contents:
            sections.append('The following are relevant reasoning clues:\n' +
                            '\n'.join(f'{i}. {text}' for i, text in enumerate(clue_contents, 1)))
        if chunk_contents:
            sections.append('The following are relevant passage knowledge:\n' +
                            '\n'.join(f'{i}. {text}' for i, text in enumerate(chunk_contents, 1)))
        return '\n\n'.join(sections)

    async def _get_fact_belong_infos(self, fact_id: str) -> list[dict[str, Any]]:
        relation = await self.micro_kv.get_by_id(fact_id) or {}
        if not isinstance(relation, dict):
            return []
        return [item for item in relation.get('belong', [])
                if isinstance(item, dict) and item.get('chunk_id') is not None and item.get('position') is not None]

    async def _get_chunk_sentence_index(self, chunk_id: str) -> dict[int, str]:
        if self._chunk_sentence_index_cache is None:
            self._chunk_sentence_index_cache = {}
        if chunk_id in self._chunk_sentence_index_cache:
            return self._chunk_sentence_index_cache[chunk_id]
        return {}

    async def _ensure_micro_sentence_index(self):
        if self._chunk_sentence_index_cache is not None:
            return
        index: dict[str, dict[int, str]] = {}
        for fact_id, relation in self.micro_kv._data.items():
            if not isinstance(relation, dict):
                continue
            for item in relation.get('belong', []):
                if item.get('chunk_id') is None or item.get('position') is None:
                    continue
                index.setdefault(item['chunk_id'], {})[int(item['position'])] = fact_id
        self._chunk_sentence_index_cache = index

    async def _build_seed_intervals(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        intervals = []
        for fact_id in fact_ids:
            for item in await self._get_fact_belong_infos(fact_id):
                pos = int(item['position'])
                intervals.append({'chunk_id': item['chunk_id'], 'start': max(0, pos - 1),
                                  'end': pos + 1, 'seed_fact_ids': [fact_id]})
        return intervals

    def _merge_intervals(self, intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for interval in intervals:
            grouped.setdefault(interval['chunk_id'], []).append(interval)
        merged = []
        for chunk_id, values in grouped.items():
            values = sorted(values, key=lambda x: (x['start'], x['end']))
            current = {key: (list(value) if isinstance(value, list) else value) for key, value in values[0].items()}
            for value in values[1:]:
                if value['start'] <= current['end']:
                    current['end'] = max(current['end'], value['end'])
                    for key, items in value.items():
                        if key in {'chunk_id', 'start', 'end'} or not isinstance(items, list):
                            continue
                        current.setdefault(key, [])
                        for item in items:
                            if item not in current[key]:
                                current[key].append(item)
                else:
                    merged.append(current)
                    current = {key: (list(item) if isinstance(item, list) else item) for key, item in value.items()}
            merged.append(current)
        return merged

    @staticmethod
    def _position_in_intervals(chunk_id: str, position: int, intervals: list[dict[str, Any]]) -> bool:
        return any(item['chunk_id'] == chunk_id and item['start'] <= position <= item['end'] for item in intervals)

    @staticmethod
    def _interval_key(interval: dict[str, Any]) -> tuple[str, int, int]:
        return interval['chunk_id'], int(interval['start']), int(interval['end'])

    def _find_source_intervals_for_seed_ids(
        self,
        source_seed_ids: set[str],
        merged_seed_intervals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            interval
            for interval in merged_seed_intervals
            if set(interval.get('seed_fact_ids', [])) & source_seed_ids
        ]

    async def _collect_sentence_expansion_candidates(self, seed_ids: list[str]) -> dict[str, dict[str, set[str]]]:
        candidates: dict[str, dict[str, set[str]]] = {}
        for seed_id in seed_ids:
            for entity_id in await self.micro_graph.get_neighbors(seed_id) or []:
                if not entity_id.startswith('<entity>'):
                    continue
                for fact_id in await self.micro_graph.get_neighbors(entity_id) or []:
                    if not fact_id.startswith('<fact>') or fact_id == seed_id:
                        continue
                    item = candidates.setdefault(fact_id, {'entity_ids': set(), 'source_seed_ids': set()})
                    item['entity_ids'].add(entity_id)
                    item['source_seed_ids'].add(seed_id)
        return candidates

    async def _get_interval_fact_ids(self, interval: dict[str, Any]) -> list[str]:
        await self._ensure_micro_sentence_index()
        chunk_index = (self._chunk_sentence_index_cache or {}).get(interval['chunk_id'], {})
        return [chunk_index[pos] for pos in sorted(chunk_index)
                if interval['start'] <= pos <= interval['end']]

    async def _render_interval_text(self, interval: dict[str, Any]) -> str:
        texts = []
        last_title = None
        for fact_id in await self._get_interval_fact_ids(interval):
            text = await self._get_fact_text_by_id(fact_id)
            if text:
                title, content = self._split_fact_title_and_content(text)
                if title is not None and content:
                    texts.append(content if title == last_title else f'[{title}]:{content}')
                    last_title = title
                else:
                    texts.append(text)
                    last_title = None
        return ' '.join(texts).strip()

    def _split_fact_title_and_content(self, fact_text: str) -> tuple[str | None, str]:
        match = re.match(r'^\[(.*?)\][：:](.*)$', str(fact_text or '').strip())
        if not match:
            return None, str(fact_text or '').strip()
        return match.group(1).strip(), match.group(2).strip()

    async def _format_structured_clue(
        self,
        source_interval: dict[str, Any],
        target_intervals: list[dict[str, Any]],
        entity_ids: list[str],
    ) -> str:
        source_text = await self._render_interval_text(source_interval)
        if not source_text or not target_intervals:
            return ''
        entity_texts = []
        for entity_id in entity_ids:
            entity_text = await self._get_entity_text_by_id(entity_id)
            if entity_text and entity_text not in entity_texts:
                entity_texts.append(entity_text)
        rendered_targets = []
        for index, target_interval in enumerate(target_intervals, start=2):
            target_text = await self._render_interval_text(target_interval)
            if target_text:
                rendered_targets.append(f'Background Clue {index}: {target_text}')
        if not rendered_targets:
            return ''
        parts = [f'Background Clue 1: {source_text}']
        if entity_texts:
            parts.append(f"Key Entities: {' '.join(entity_texts)}")
        parts.extend(rendered_targets)
        return '\n'.join(parts)

    async def _render_clue_entry(self, entry: dict[str, Any]) -> str:
        if entry.get('is_isolated'):
            return await self._render_interval_text(entry['source_interval'])
        return await self._format_structured_clue(
            source_interval=entry['source_interval'],
            target_intervals=entry.get('target_intervals', []),
            entity_ids=entry.get('entity_ids', []),
        )

    async def _rerank_clue_entries(self, query_text: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = []
        texts = []
        for entry in entries:
            text = await self._render_clue_entry(entry)
            if text:
                item = dict(entry)
                item['rendered_text'] = text
                prepared.append(item)
                texts.append(text)
        ranked = await self._rerank_text_candidates(query_text, texts)
        scores: dict[str, list[float]] = {}
        for text, score in ranked:
            scores.setdefault(text, []).append(float(score))
        for entry in prepared:
            available = scores.get(entry['rendered_text'], [])
            entry['score'] = available.pop(0) if available else 0.0
        return sorted(prepared, key=lambda x: x['score'], reverse=True)

    @staticmethod
    def _intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left['chunk_id'] != right['chunk_id']:
            return False
        return not (int(left['end']) < int(right['start']) or int(right['end']) < int(left['start']))

    def _merge_reranked_clue_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = []
        source_indexes: dict[tuple[str, int, int], int] = {}
        claimed_targets: list[tuple[dict[str, Any], int]] = []
        for entry in entries:
            source_key = entry['source_key']
            if source_key not in source_indexes:
                source_indexes[source_key] = len(merged)
                merged.append({'source_key': source_key, 'source_interval': entry['source_interval'],
                               'target_intervals': [], 'entity_ids': list(entry.get('entity_ids', [])),
                               'is_isolated': bool(entry.get('is_isolated', False)),
                               'score': float(entry.get('score', 0.0))})
            source_index = source_indexes[source_key]
            owner = merged[source_index]
            owner['score'] = max(owner['score'], float(entry.get('score', 0.0)))
            for entity_id in entry.get('entity_ids', []):
                if entity_id not in owner['entity_ids']:
                    owner['entity_ids'].append(entity_id)
            for target in entry.get('target_intervals', []):
                target_owner = next((index for claimed, index in claimed_targets
                                     if self._intervals_overlap(target, claimed)), None)
                if target_owner is None:
                    target_owner = source_index
                    claimed_targets.append((target, target_owner))
                target_entry = merged[target_owner]
                if self._interval_key(target) not in {self._interval_key(item) for item in target_entry['target_intervals']}:
                    target_entry['target_intervals'].append(target)
                for entity_id in entry.get('entity_ids', []):
                    if entity_id not in target_entry['entity_ids']:
                        target_entry['entity_ids'].append(entity_id)
        for entry in merged:
            entry['target_intervals'] = self._merge_intervals(entry['target_intervals'])
            entry['is_isolated'] = not entry['target_intervals']
        return merged

    @staticmethod
    def _extract_sentence_ids_from_clue_entry(entry: dict[str, Any]) -> list[str]:
        ids = []
        for fact_id in entry.get('source_interval', {}).get('seed_fact_ids', []):
            if fact_id not in ids:
                ids.append(fact_id)
        for target in entry.get('target_intervals', []):
            for fact_id in target.get('target_fact_ids', []):
                if fact_id not in ids:
                    ids.append(fact_id)
        return ids

    async def _rerank_text_candidates(self, query_text: str, texts: list[str]) -> list[tuple[str, float]]:
        unique_texts = list(dict.fromkeys(str(text or '').strip() for text in texts if str(text or '').strip()))
        if not unique_texts:
            return []
        reranker = self._get_reranker()
        if reranker is not None:
            try:
                scores = self._reranker_compute_score(reranker, [[query_text, text] for text in unique_texts], normalize=True)
                if isinstance(scores, (float, int)):
                    scores = [scores]
                return sorted(zip(unique_texts, [float(score) for score in scores]), key=lambda x: x[1], reverse=True)
            except Exception as exc:
                if 'meta tensor' in str(exc).lower():
                    self._reranker_disabled = True
                    self._reranker = None
                logger.warning('Clue rerank failed, falling back to embedding similarity: %s', exc)
        query_embedding = await self._embed_text(query_text, self.fact_vdb)
        if query_embedding is None:
            return [(text, 0.0) for text in unique_texts]
        result = await self.fact_vdb.embedding_func(unique_texts)
        embeddings = result.get('embeddings', [])
        return sorted([(text, self._cosine_similarity(query_embedding, np.asarray(embedding, dtype=np.float32)))
                       for text, embedding in zip(unique_texts, embeddings)], key=lambda x: x[1], reverse=True)

    async def _compute_query_sentence_similarity(self, sentence_ids: list[str], query_text: str) -> dict[str, float]:
        query_embedding = await self._embed_text(query_text, self.fact_vdb)
        if query_embedding is None or not sentence_ids:
            return {}
        pairs = []
        for sentence_id in sentence_ids:
            text = await self._get_fact_text_by_id(sentence_id)
            if text:
                pairs.append((sentence_id, text))
        embeddings = await self._embed_texts_by_id(pairs, self.fact_vdb)
        return {sentence_id: self._cosine_similarity(query_embedding, embedding)
                for sentence_id, embedding in embeddings.items()}

    async def query_sentence_expansion(self, dict_query: dict[str, Any]) -> dict[str, Any]:
        sentence_text = str(dict_query.get('sentence_text', '') or '')
        if not sentence_text.strip():
            return {'clue_texts': [], 'clue_scores': {}, 'clue_sentence_ids': [], 'seed_sentence_ids': [], 'expanded_sentence_ids': []}
        supplied = [sid for sid in list(dict_query.get('seed_sentence_ids', []) or []) if str(sid).startswith('<fact>')]
        recalled = [item.get('id') for item in await self.fact_vdb.query(query=sentence_text, top_k=self.fact_retrieve_k) if item.get('id')]
        seed_ids = list(dict.fromkeys(recalled + supplied))
        if not seed_ids:
            return {'clue_texts': [], 'clue_scores': {}, 'clue_sentence_ids': [], 'seed_sentence_ids': [], 'expanded_sentence_ids': []}

        await self._ensure_micro_sentence_index()
        seed_intervals = self._merge_intervals(await self._build_seed_intervals(seed_ids))
        candidates = await self._collect_sentence_expansion_candidates(seed_ids)
        candidate_ids = list(candidates)
        similarities = await self._compute_query_sentence_similarity(candidate_ids, sentence_text)
        expanded_ids = [sid for sid, _ in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:max(self.fact_return_k, self.fact_retrieve_k)]]

        path_targets_by_source = {}
        for target_id in expanded_ids:
            target_occurrences = [item for item in await self._get_fact_belong_infos(target_id)
                                  if not self._position_in_intervals(item['chunk_id'], int(item['position']), seed_intervals)]
            if not target_occurrences:
                continue
            source_ids = candidates[target_id]['source_seed_ids']
            source_intervals = self._find_source_intervals_for_seed_ids(source_ids, seed_intervals)
            for source in source_intervals:
                source_key = self._interval_key(source)
                path_targets_by_source.setdefault(source_key, {'source_interval': source, 'target_intervals': []})
                for occurrence in target_occurrences:
                    target_interval = {'chunk_id': occurrence['chunk_id'], 'start': max(0, int(occurrence['position']) - 1),
                                       'end': int(occurrence['position']) + 1, 'target_fact_ids': [target_id],
                                       'entity_ids': list(candidates[target_id]['entity_ids'])}
                    path_targets_by_source[source_key]['target_intervals'].append(target_interval)

        entries = []
        linked_sources = set()
        for source_key, path_info in path_targets_by_source.items():
            for target_interval in self._merge_intervals(path_info['target_intervals']):
                entries.append({'source_key': source_key, 'source_interval': path_info['source_interval'],
                                'target_intervals': [target_interval],
                                'entity_ids': list(target_interval.get('entity_ids', [])), 'is_isolated': False})
                linked_sources.add(source_key)
        for source in seed_intervals:
            if self._interval_key(source) not in linked_sources:
                entries.append({'source_key': self._interval_key(source), 'source_interval': source,
                                'target_intervals': [], 'entity_ids': [], 'is_isolated': True})

        ranked = await self._rerank_clue_entries(sentence_text, entries)
        top_entries = self._merge_reranked_clue_entries(ranked)[:self.fact_return_k]
        clues = []
        clue_scores = {}
        ids_by_clue = []
        for entry in top_entries:
            clue_text = await self._render_clue_entry(entry)
            if not clue_text:
                continue
            clues.append(clue_text)
            clue_scores[clue_text] = float(entry.get('score', 0.0))
            ids_by_clue.append(self._extract_sentence_ids_from_clue_entry(entry))
        clue_ids = list(dict.fromkeys(fact_id for ids in ids_by_clue for fact_id in ids))
        return {'clue_texts': clues, 'clue_scores': clue_scores,
                'clue_sentence_ids_by_clue': ids_by_clue, 'clue_sentence_ids': clue_ids,
                'seed_sentence_ids': seed_ids, 'expanded_sentence_ids': expanded_ids}

    async def query_topic_expansion(self, dict_query: dict[str, Any]) -> dict[str, Any]:
        sentence_text = str(dict_query.get("sentence_text", "") or "")
        topic_top_k = int(dict_query.get("topic_top_k", self.topic_top_k))
        chunk_top_k = int(dict_query.get("chunk_top_k", self.chunk_top_k))
        seed_fact_top_k = int(dict_query.get("seed_fact_top_k", self.fact_retrieve_k))
        input_seed_sentence_ids_raw = list(dict_query.get("seed_sentence_ids", []) or [])
        input_seed_sentence_ids = []
        for sid in input_seed_sentence_ids_raw:
            if isinstance(sid, list):
                for nested_sid in sid:
                    if nested_sid not in input_seed_sentence_ids:
                        input_seed_sentence_ids.append(nested_sid)
            else:
                if sid not in input_seed_sentence_ids:
                    input_seed_sentence_ids.append(sid)
        rerank_top_m = int(dict_query.get("topic_rerank_top_m", max(self.topic_chunk_top_k, self.chunk_top_k)))
        final_top_k = int(dict_query.get("final_top_k", self.topic_chunk_top_k))

        logger.info(
            "Start topic expansion retrieval: sentence_text=%s, topic_top_k=%s, chunk_top_k=%s, seed_fact_top_k=%s, input_seed_sentence_ids=%s",
            sentence_text,
            topic_top_k,
            chunk_top_k,
            seed_fact_top_k,
            len(input_seed_sentence_ids),
        )
        if not sentence_text.strip():
            return {"chunk_ids": [], "chunk_scores": {}, "related_topic_ids": [], "related_topic_scores": {}}

        if input_seed_sentence_ids:
            seed_fact_ids = [sid for sid in input_seed_sentence_ids if str(sid).startswith("<fact>")]
        else:
            seed_fact_results = await self.fact_vdb.query(query=sentence_text, top_k=seed_fact_top_k)
            seed_fact_ids = [item.get("id") for item in seed_fact_results if item.get("id")]
        seed_chunks_from_facts = set()
        for fact_id in seed_fact_ids:
            belong_infos = await self._get_fact_belong_infos(fact_id)
            for belong_item in belong_infos:
                chunk_id = belong_item.get("chunk_id")
                if chunk_id:
                    seed_chunks_from_facts.add(chunk_id)

        topic_results = await self.topic_vdb.query(query=sentence_text, top_k=topic_top_k)
        topic_id_to_score = {}
        for topic_result in topic_results:
            topic_id = topic_result.get("id")
            if not topic_id:
                continue
            score = float(topic_result.get("distance", 0.0))
            if topic_id not in topic_id_to_score or score > topic_id_to_score[topic_id]:
                topic_id_to_score[topic_id] = score

        topic_to_chunks = {}
        chunk_hit_topic_sims = {}
        for topic_id, topic_sim in topic_id_to_score.items():
            neighbors = await self.macro_graph.get_neighbors(topic_id)
            if not neighbors:
                continue
            topic_chunks = [nid for nid in neighbors if nid.startswith("<chunk>")]
            topic_to_chunks[topic_id] = topic_chunks
            for chunk_id in topic_chunks:
                if chunk_id not in chunk_hit_topic_sims:
                    chunk_hit_topic_sims[chunk_id] = []
                chunk_hit_topic_sims[chunk_id].append(topic_sim)
        seed_chunks_from_topics = {chunk_id for topic_chunks in topic_to_chunks.values() for chunk_id in topic_chunks}

        chunk_results = await self.chunk_vdb.query(query=sentence_text, top_k=chunk_top_k)
        seed_chunks_from_vector = set()
        chunk_vector_seed_scores = {}
        for chunk_result in chunk_results:
            chunk_id = chunk_result.get("id")
            if not chunk_id:
                continue
            score = float(chunk_result.get("distance", 0.0))
            seed_chunks_from_vector.add(chunk_id)
            chunk_vector_seed_scores[chunk_id] = score

        candidate_chunk_ids = seed_chunks_from_facts | seed_chunks_from_topics | seed_chunks_from_vector
        if not candidate_chunk_ids:
            return {
                "chunk_ids": [],
                "chunk_scores": {},
                "related_topic_ids": list(topic_id_to_score.keys()),
                "related_topic_scores": topic_id_to_score,
            }

        logger.info(
            "Seed merge stats: from_facts=%s, from_topics=%s, from_vector=%s, merged=%s",
            len(seed_chunks_from_facts),
            len(seed_chunks_from_topics),
            len(seed_chunks_from_vector),
            len(candidate_chunk_ids),
        )

        sim_q_chunks = await self._compute_query_chunk_similarity(list(candidate_chunk_ids), sentence_text)

        chunk_scores = {}
        for chunk_id in candidate_chunk_ids:
            sim_q_v = sim_q_chunks.get(chunk_id, 0.0)
            hit_topic_sims = chunk_hit_topic_sims.get(chunk_id, [])
            topic_enhance = 1.0 + math.log10(1.0 + sum(hit_topic_sims)) if hit_topic_sims else 1.0
            chunk_scores[chunk_id] = sim_q_v * topic_enhance

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        pre_rerank_chunks = sorted_chunks[:max(final_top_k, rerank_top_m)]
        reranked_chunks = await self._rerank_chunk_candidates(sentence_text, pre_rerank_chunks)
        top_chunks = reranked_chunks[:final_top_k]
        top_chunk_ids = [chunk_id for chunk_id, _ in top_chunks]

        logger.info("Topic expansion retrieval finished: returned %s chunks", len(top_chunk_ids))

        return {
            "chunk_ids": top_chunk_ids,
            "chunk_scores": dict(top_chunks),
            "related_topic_ids": list(topic_id_to_score.keys()),
            "related_topic_scores": topic_id_to_score,
            "seed_chunks_from_facts": list(seed_chunks_from_facts),
            "seed_chunks_from_topics": list(seed_chunks_from_topics),
            "seed_chunks_from_vector": list(seed_chunks_from_vector),
        }

    async def _rerank_chunk_candidates(self, query_text: str, chunk_candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        reranker = self._get_reranker()
        if reranker is None:
            return chunk_candidates

        rerank_pairs = []
        rerank_chunk_ids = []
        for chunk_id, _ in chunk_candidates:
            chunk_data = await self.chunk_vdb.get_by_id(chunk_id)
            if not chunk_data:
                continue
            chunk_text = str(chunk_data.get("text", "") or "").strip()
            chunk_title = str(chunk_data.get("document_title", "") or "").strip()
            if not chunk_text:
                continue
            candidate_text = f"[{chunk_title}]：{chunk_text}" if chunk_title else chunk_text
            rerank_pairs.append([query_text, candidate_text])
            rerank_chunk_ids.append(chunk_id)

        if not rerank_pairs:
            return chunk_candidates

        try:
            rerank_scores = self._reranker_compute_score(reranker, rerank_pairs, normalize=True)
            if isinstance(rerank_scores, (float, int)):
                rerank_scores = [float(rerank_scores)]
        except Exception as e:
            if "meta tensor" in str(e).lower():
                self._reranker_disabled = True
                self._reranker = None
            logger.warning("Chunk rerank failed, falling back to formula ordering: %s", e)
            return chunk_candidates

        rerank_map = {cid: float(score) for cid, score in zip(rerank_chunk_ids, rerank_scores)}

        reranked = []
        missing = []
        for chunk_id, base_score in chunk_candidates:
            if chunk_id in rerank_map:
                reranked.append((chunk_id, rerank_map[chunk_id]))
            else:
                missing.append((chunk_id, base_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked.extend(missing)
        return reranked

    async def _compute_query_chunk_similarity(self, chunk_ids: list[str], query_text: str) -> dict[str, float]:
        if not chunk_ids or not query_text:
            return {}

        q_embedding = await self._embed_text(query_text, self.chunk_vdb)
        if q_embedding is None:
            return {}

        chunk_id_text_pairs = []
        for chunk_id in chunk_ids:
            chunk_data = await self.chunk_vdb.get_by_id(chunk_id)
            if not chunk_data:
                continue
            chunk_text = str(chunk_data.get("text", "") or "").strip()
            chunk_title = str(chunk_data.get("document_title", "") or "").strip()
            if not chunk_text:
                continue
            content_text = f"{chunk_title}:{chunk_text}" if chunk_title else chunk_text
            chunk_id_text_pairs.append((chunk_id, content_text))

        chunk_id_to_vector = await self._embed_texts_by_id(chunk_id_text_pairs, self.chunk_vdb)
        if not chunk_id_to_vector:
            return {}

        q_norm = np.linalg.norm(q_embedding)
        if q_norm == 0:
            return {}

        sim_scores = {}
        for chunk_id, c_embedding in chunk_id_to_vector.items():
            c_norm = np.linalg.norm(c_embedding)
            if c_norm > 0:
                similarity = float(np.dot(q_embedding, c_embedding) / (q_norm * c_norm))
                sim_scores[chunk_id] = similarity

        return sim_scores
