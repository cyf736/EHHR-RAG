import asyncio
import os

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm.asyncio import tqdm as tqdm_async

from ehhr_rag.config import prompt_dir
from ehhr_rag.llm import generate_from_prompt_template
from ehhr_rag.logging_utils import logger
from ehhr_rag.storage.nano_vector_storage import NanoVectorStorage
from ehhr_rag.storage.networkx_graph_storage import NetworkXStorage
from ehhr_rag.text_utils import compute_md5_hash_id

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MacroGraphBuilder:
    def __init__(self, db_base_dir: str, namespace: str = "layer_macro"):
        self.db_base_dir = db_base_dir
        self.namespace = namespace

        self.kg = NetworkXStorage(namespace=namespace, base_dir=db_base_dir)
        self.chunk_vdb = NanoVectorStorage(namespace=f"{namespace}_chunks", base_dir=db_base_dir)
        self.topic_vdb = NanoVectorStorage(namespace=f"{namespace}_topics", base_dir=db_base_dir)

        self.tau = 0.10
        self.min_components = 100
        self.samples_per_component = 25
        self.prompt_path = str(prompt_dir("indexing") / "summary_chunk.txt")

    async def build_from_chunks(self, chunks: list[dict]):
        """Build the macro-theme hypergraph from chunk records."""
        logger.info("Building macro-theme hypergraph with %s chunks.", len(chunks))

        chunk_records = {}
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = str(chunk.get("chunk_text", "") or "")
            doc_title = str(chunk.get("document_title", "") or "")
            if not chunk_text:
                continue

            chunk_records[chunk_id] = {
                "content": f"{doc_title}:{chunk_text}",
                "text": chunk_text,
                "document_title": doc_title,
            }

            if await self.kg.has_node(chunk_id):
                continue
            await self.kg.upsert_node(
                chunk_id,
                node_data={
                    "role": "node",
                    "type": "chunk",
                    "text": chunk_text,
                    "document_title": doc_title,
                },
            )

        await self.kg.index_done_callback()

        if not chunk_records:
            logger.error("No usable chunk text found; cannot cluster.")
            return

        ordered_chunk_ids = list(chunk_records.keys())
        chunk_contents = [chunk_records[cid]["content"] for cid in ordered_chunk_ids]
        batch_size = max(1, int(getattr(self.chunk_vdb, "_max_batch_size", 64)))

        chunk_id_to_embedding: dict[str, np.ndarray] = {}
        for start in range(0, len(chunk_contents), batch_size):
            batch_ids = ordered_chunk_ids[start:start + batch_size]
            batch_texts = chunk_contents[start:start + batch_size]
            try:
                result = await self.chunk_vdb.embedding_func(
                    batch_texts,
                    embed_model=self.chunk_vdb.embedding_model,
                )
            except Exception as exc:
                logger.warning("Chunk embedding batch failed, skipping batch: %s", exc)
                continue

            embeddings = result.get("embeddings", [])
            valid_indices = result.get("valid_indices", [])
            if embeddings is None:
                continue
            emb_list = [np.asarray(emb, dtype=np.float32) for emb in embeddings]
            if not emb_list:
                continue

            if isinstance(valid_indices, list) and len(valid_indices) == len(emb_list):
                for pos, local_idx in enumerate(valid_indices):
                    if 0 <= int(local_idx) < len(batch_ids):
                        chunk_id_to_embedding[batch_ids[int(local_idx)]] = emb_list[pos]
            else:
                for cid, emb in zip(batch_ids, emb_list):
                    chunk_id_to_embedding[cid] = emb

        vdb_upsert_data = []
        for chunk_id in ordered_chunk_ids:
            chunk_meta_item = chunk_records[chunk_id]
            vector = chunk_id_to_embedding.get(chunk_id)
            if vector is None:
                continue
            vdb_upsert_data.append(
                {
                    "__id__": chunk_id,
                    "__vector__": vector,
                    "content": chunk_meta_item["content"],
                    "text": chunk_meta_item["text"],
                    "document_title": chunk_meta_item["document_title"],
                }
            )
        if vdb_upsert_data:
            self.chunk_vdb._client.upsert(datas=vdb_upsert_data)
            await self.chunk_vdb.index_done_callback()
            logger.info("Chunk vectors written to VDB: %s/%s", len(vdb_upsert_data), len(ordered_chunk_ids))

        embeddings = []
        chunk_meta = {}
        filtered_chunk_ids = []
        for chunk_id in ordered_chunk_ids:
            vector = chunk_id_to_embedding.get(chunk_id)
            if vector is None:
                continue
            embeddings.append(vector)
            filtered_chunk_ids.append(chunk_id)
            chunk_meta[chunk_id] = {
                "text": chunk_records[chunk_id].get("text", ""),
                "document_title": chunk_records[chunk_id].get("document_title", ""),
            }
        ordered_chunk_ids = filtered_chunk_ids
        logger.info(
            "Macro clustering sample stats: chunks_input=%s, with_vector=%s, without_vector=%s",
            len(chunks),
            len(embeddings),
            max(0, len(chunks) - len(embeddings)),
        )

        X = np.array(embeddings)
        n_samples = X.shape[0]

        K = n_samples // self.samples_per_component
        K = max(self.min_components, K)
        K = min(K, n_samples)
        if K < 2:
            logger.warning(
                "Too few samples for meaningful clustering. n_samples=%s, chunks_input=%s, samples_per_component=%s, min_components=%s",
                n_samples,
                len(chunks),
                self.samples_per_component,
                self.min_components,
            )
            return
        effective_tau = max(self.tau, 1.0 / K)

        logger.info("Running GMM soft clustering (K=%s, tau=%.4f)...", K, effective_tau)
        gmm = GaussianMixture(n_components=K, random_state=42)
        gmm.fit(X)

        probs = gmm.predict_proba(X)

        topic_vdb_data = {}
        topic_id_by_cluster = {}
        covered_chunk_ids = set()

        for k in range(K):
            cluster_indices = np.where(probs[:, k] > effective_tau)[0]
            if len(cluster_indices) == 0:
                cluster_indices = np.where(np.argmax(probs, axis=1) == k)[0]
            if len(cluster_indices) == 0:
                continue

            top_n_indices = np.argsort(probs[:, k])[::-1][:10]
            core_texts = []
            core_titles = set()

            for idx in top_n_indices:
                chunk_id = ordered_chunk_ids[idx]
                text = chunk_meta.get(chunk_id, {}).get("text", "")
                title = chunk_meta.get(chunk_id, {}).get("document_title", "")
                core_texts.append(text)
                if title:
                    core_titles.add(title)

            combined_text = "\n---\n".join(core_texts)
            combined_title = "; ".join(core_titles) if core_titles else ""

            logger.info("Generating summary for topic cluster %s...", k)
            fallback_summary = f"Topic cluster {k} (fallback summary): {combined_title if combined_title else 'Untitled'}"
            try:
                if os.path.exists(self.prompt_path):
                    summary = generate_from_prompt_template([combined_title, combined_text], self.prompt_path)
                else:
                    logger.warning("Prompt template missing: %s; using fallback summary.", self.prompt_path)
                    summary = fallback_summary
            except Exception as exc:
                logger.error("Failed to generate summary: %s", exc)
                summary = fallback_summary

            if not isinstance(summary, str) or not summary.strip():
                logger.warning("Topic cluster %s summary is empty; using fallback summary.", k)
                summary = fallback_summary

            topic_id = compute_md5_hash_id(summary, prefix="<topic>")
            topic_data = {
                "role": "hyperedge",
                "type": "topic",
                "summary": summary,
            }
            await self.kg.upsert_node(topic_id, node_data=topic_data)
            topic_id_by_cluster[k] = topic_id

            topic_vdb_data[topic_id] = {
                "content": summary,
                "summary": summary,
                "topic_id": topic_id,
            }

            for idx in cluster_indices:
                member_chunk_id = ordered_chunk_ids[idx]
                edge_data = {"relation": "belongs_to_topic", "probability": float(probs[idx, k])}
                await self.kg.upsert_edge(topic_id, member_chunk_id, edge_data=edge_data)
                covered_chunk_ids.add(member_chunk_id)

        for idx, chunk_id in enumerate(ordered_chunk_ids):
            if chunk_id in covered_chunk_ids:
                continue
            best_k = int(np.argmax(probs[idx]))
            topic_id = topic_id_by_cluster.get(best_k)
            if topic_id is None:
                continue
            edge_data = {"relation": "belongs_to_topic", "probability": float(probs[idx, best_k])}
            await self.kg.upsert_edge(topic_id, chunk_id, edge_data=edge_data)
            covered_chunk_ids.add(chunk_id)

        logger.info("Topic clustering coverage: %s/%s", len(covered_chunk_ids), len(ordered_chunk_ids))

        await self.kg.index_done_callback()

        logger.info("Vectorizing new topic summaries...")
        if topic_vdb_data:
            await self.topic_vdb.upsert(topic_vdb_data, need_embedding_list=False)
            await self.topic_vdb.index_done_callback()

        logger.info("Macro-theme hypergraph construction completed.")
