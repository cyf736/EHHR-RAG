import asyncio
import os
import re

import nltk
import spacy
from nltk.corpus import stopwords
from tqdm.asyncio import tqdm as tqdm_async

from ehhr_rag.logging_utils import logger
from ehhr_rag.storage.json_kv_storage import JsonKVStorage
from ehhr_rag.storage.nano_vector_storage import NanoVectorStorage
from ehhr_rag.storage.networkx_graph_storage import NetworkXStorage
from ehhr_rag.text_utils import compute_md5_hash_id

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def clean_text(text):
    """Remove punctuation and stop words while keeping predicate structure."""
    if not text:
        return ""
    text = str(text)
    punctuation_pattern = r"[^\w\s]"
    text = re.sub(punctuation_pattern, " ", text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 0]
    return " ".join(filtered_words)


class MicroGraphBuilder:
    def __init__(self, db_base_dir: str, namespace: str = "layer_micro"):
        self.db_base_dir = db_base_dir
        self.namespace = namespace

        logger.info("Loading spaCy model (en_core_web_lg)...")
        self.nlp = spacy.load("en_core_web_lg")

        self.kg = NetworkXStorage(namespace=namespace, base_dir=db_base_dir)
        self.entity_vdb = NanoVectorStorage(namespace=f"{namespace}_entities", base_dir=db_base_dir)
        self.predicate_vdb = NanoVectorStorage(namespace=f"{namespace}_predicates", base_dir=db_base_dir)
        self.fact_vdb = NanoVectorStorage(namespace=f"{namespace}_facts", base_dir=db_base_dir)
        self.kv_db = JsonKVStorage(namespace=namespace, base_dir=db_base_dir)

        self.similarity_threshold = 0.85
        self.similarity_top_k = 3

    async def build_from_chunks(self, chunks: list[dict]):
        """Build the micro-fact hypergraph from chunk records."""
        all_entities = {}
        all_predicates = {}
        all_facts = {}
        sentence_relation_map = {}

        for chunk in tqdm_async(chunks, desc="Building micro-fact hypergraph"):
            chunk_text = chunk["chunk_text"]
            chunk_title = str(chunk.get("document_title", "")).strip()
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                chunk_id = compute_md5_hash_id(f"{chunk_title}:{chunk_text}", prefix="<chunk>")
            doc = self.nlp(chunk_text)
            chunk_sentences = []

            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                fact_text = f"[{chunk_title}]：{sent_text}"
                fact_id = compute_md5_hash_id(fact_text, prefix="<fact>")
                chunk_sentences.append((sent, sent_text, fact_id))

            for idx, (_, _, fact_id) in enumerate(chunk_sentences):
                if fact_id not in sentence_relation_map:
                    sentence_relation_map[fact_id] = {"belong": [], "pre": [], "next": []}
                if chunk_id:
                    belong_item = {"chunk_id": chunk_id, "position": idx}
                    if belong_item not in sentence_relation_map[fact_id]["belong"]:
                        sentence_relation_map[fact_id]["belong"].append(belong_item)

                if idx > 0:
                    pre_fact_id = chunk_sentences[idx - 1][2]
                    if pre_fact_id not in sentence_relation_map[fact_id]["pre"]:
                        sentence_relation_map[fact_id]["pre"].append(pre_fact_id)

                if idx < len(chunk_sentences) - 1:
                    next_fact_id = chunk_sentences[idx + 1][2]
                    if next_fact_id not in sentence_relation_map[fact_id]["next"]:
                        sentence_relation_map[fact_id]["next"].append(next_fact_id)

            for sent, sent_text, fact_id in chunk_sentences:
                fact_text = f"[{chunk_title}]：{sent_text}"

                if await self.kg.has_node(fact_id):
                    continue

                fact_data = {"role": "hyperedge", "type": "fact", "text": fact_text}
                await self.kg.upsert_node(fact_id, node_data=fact_data)

                all_facts[fact_id] = {"content": fact_text, "text": fact_text, "fact_id": fact_id}

                extracted_ents = []
                for ent in sent.ents:
                    ent_text = ent.text.strip()
                    if not ent_text:
                        continue
                    extracted_ents.append((ent_text, ent.label_))

                extracted_ents = sorted(extracted_ents, key=lambda x: len(x[0]), reverse=True)
                atomic_text = sent_text

                for ent_text, ent_label in extracted_ents:
                    ent_id = compute_md5_hash_id(ent_text, prefix="<entity>")
                    ent_data = {
                        "role": "node",
                        "type": "entity",
                        "text": ent_text,
                        "label": ent_label,
                    }
                    await self.kg.upsert_node(ent_id, node_data=ent_data)
                    await self.kg.upsert_edge(fact_id, ent_id, edge_data={"relation": "contains_entity"})

                    if ent_id not in all_entities:
                        all_entities[ent_id] = {"content": ent_text, "text": ent_text, "entity_id": ent_id}

                    atomic_text = re.sub(re.escape(ent_text), "", atomic_text)

                atomic_text = " ".join(atomic_text.split())
                cleaned_predicate = clean_text(atomic_text)

                if cleaned_predicate:
                    pred_id = compute_md5_hash_id(cleaned_predicate, prefix="<predicate>")
                    pred_data = {"role": "node", "type": "predicate", "text": cleaned_predicate}
                    await self.kg.upsert_node(pred_id, node_data=pred_data)
                    await self.kg.upsert_edge(fact_id, pred_id, edge_data={"relation": "contains_predicate"})

                    if pred_id not in all_predicates:
                        all_predicates[pred_id] = {
                            "content": cleaned_predicate,
                            "text": cleaned_predicate,
                            "predicate_id": pred_id,
                        }

        await self.kg.index_done_callback()

        if all_entities:
            logger.info("Vectorizing %s new entities...", len(all_entities))
            await self.entity_vdb.upsert(all_entities, need_embedding_list=False)
            await self.entity_vdb.index_done_callback()

        if all_predicates:
            logger.info("Vectorizing %s new predicates...", len(all_predicates))
            await self.predicate_vdb.upsert(all_predicates, need_embedding_list=False)
            await self.predicate_vdb.index_done_callback()

        if all_facts:
            logger.info("Vectorizing %s new fact sentences...", len(all_facts))
            await self.fact_vdb.upsert(all_facts, need_embedding_list=False)
            await self.fact_vdb.index_done_callback()

        if sentence_relation_map:
            await self._upsert_sentence_relations(sentence_relation_map)

        await self.build_similarity_hyperedges(seed_entity_ids=list(all_entities.keys()))

    async def _upsert_sentence_relations(self, relation_map: dict[str, dict]):
        """Persist sentence adjacency metadata."""
        overwrite_count = 0
        for fact_id, rel in relation_map.items():
            self.kv_db._data[fact_id] = {
                "belong": list(rel.get("belong", [])),
                "pre": list(rel.get("pre", [])),
                "next": list(rel.get("next", [])),
            }
            overwrite_count += 1

        await self.kv_db.index_done_callback()
        logger.info("Sentence relation map updated: %s rows", overwrite_count)

    async def build_similarity_hyperedges(self, seed_entity_ids: list[str] | None = None):
        logger.info("Computing entity similarity and building synonym hyperedges...")
        try:
            data = self.entity_vdb.client_storage.get("data", [])
            if not data:
                return

            entity_text_by_id = {}
            for item in data:
                entity_id = item.get("__id__")
                entity_text = item.get("content") or item.get("text")
                if entity_id and entity_text:
                    entity_text_by_id[entity_id] = str(entity_text)
            if len(entity_text_by_id) < 2:
                return

            if seed_entity_ids is not None:
                seed_entity_set = set(seed_entity_ids)
                if not seed_entity_set:
                    logger.info("No new entities in this run; skipping similarity hyperedges.")
                    return
                query_entity_ids = [entity_id for entity_id in entity_text_by_id.keys() if entity_id in seed_entity_set]
                if not query_entity_ids:
                    logger.info("Seed entities are not in the vector store; skipping similarity hyperedges.")
                    return
            else:
                query_entity_ids = list(entity_text_by_id.keys())

            query_entity_items = []
            batch_size = max(1, int(getattr(self.entity_vdb, "_max_batch_size", 64)))
            for start in range(0, len(query_entity_ids), batch_size):
                batch_ids = query_entity_ids[start:start + batch_size]
                batch_texts = [entity_text_by_id[entity_id] for entity_id in batch_ids]
                try:
                    result = await self.entity_vdb.embedding_func(batch_texts, embed_model=self.entity_vdb.embedding_model)
                except Exception as exc:
                    logger.warning("Entity similarity embedding batch failed, skipping batch: %s", exc)
                    continue

                embeddings = result.get("embeddings", [])
                valid_indices = result.get("valid_indices", [])
                if embeddings is None:
                    continue
                embedding_list = [emb for emb in embeddings]
                if not embedding_list:
                    continue

                if isinstance(valid_indices, list) and len(valid_indices) == len(embedding_list):
                    for pos, local_idx in enumerate(valid_indices):
                        if 0 <= int(local_idx) < len(batch_ids):
                            query_entity_items.append((batch_ids[int(local_idx)], embedding_list[pos]))
                else:
                    for entity_id, embedding in zip(batch_ids, embedding_list):
                        query_entity_items.append((entity_id, embedding))
            if len(query_entity_items) < 1:
                logger.info("No usable query entity vectors; skipping similarity hyperedges.")
                return

            entity_ids_set = set(entity_text_by_id.keys())
            seen_pairs = set()
            sim_count = 0
            existing_similarity_ids = set(await self.kg.find_nodes_by_prefix("<similarity>"))
            graph = self.kg.return_self_graph()

            max_concurrency = 12
            query_batch_size = 32
            semaphore = asyncio.Semaphore(max_concurrency)

            async def query_one_entity(item):
                id1, vector = item
                async with semaphore:
                    query_results = await self.entity_vdb.query(
                        query="",
                        top_k=self.similarity_top_k + 1,
                        query_embedding=vector,
                    )
                return id1, query_results

            progress_bar = tqdm_async(total=len(query_entity_items), desc="Building similarity candidates")
            for start_idx in range(0, len(query_entity_items), query_batch_size):
                batch_items = query_entity_items[start_idx:start_idx + query_batch_size]
                batch_tasks = [query_one_entity(item) for item in batch_items]
                batch_outputs = await asyncio.gather(*batch_tasks)
                progress_bar.update(len(batch_items))

                for id1, query_results in batch_outputs:
                    for res in query_results:
                        id2 = res.get("id")
                        if not id2 or id2 == id1 or id2 not in entity_ids_set:
                            continue
                        similarity_score = float(res.get("distance", 0.0))
                        if similarity_score < self.similarity_threshold:
                            continue
                        pair = tuple(sorted((id1, id2)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        sim_id_text = f"{pair[0]}_{pair[1]}"
                        sim_edge_id = compute_md5_hash_id(sim_id_text, prefix="<similarity>")
                        if sim_edge_id in existing_similarity_ids:
                            continue
                        existing_similarity_ids.add(sim_edge_id)
                        sim_data = {
                            "role": "hyperedge",
                            "type": "similarity",
                            "similarity_score": similarity_score,
                        }
                        graph.add_node(sim_edge_id, **sim_data)
                        graph.add_edge(sim_edge_id, pair[0], relation="similar_to")
                        graph.add_edge(sim_edge_id, pair[1], relation="similar_to")
                        sim_count += 1
            progress_bar.close()
            logger.info("Added %s similarity hyperedges.", sim_count)
            await self.kg.index_done_callback()

        except Exception as exc:
            logger.error("Error building similarity hyperedges: %s", exc)
