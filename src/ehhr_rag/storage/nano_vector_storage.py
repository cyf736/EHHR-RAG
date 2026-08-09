from pathlib import Path
from typing import Any

import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm.asyncio import tqdm as tqdm_async

from ehhr_rag.config import EMBEDDING_BATCH_NUM, OLLAMA_HOST, dataset_db_dir
from ehhr_rag.llm import get_embedding_dim, hf_embedding
from ehhr_rag.logging_utils import logger

DEFAULT_HF_EMBEDDING_MODEL = "BAAI/bge-m3"


class NanoVectorStorage:
    def __init__(self, namespace: str, base_dir: str | Path | None = None):
        self.namespace = namespace
        self.embedding_model: str = DEFAULT_HF_EMBEDDING_MODEL
        self.embedding_func = hf_embedding
        self.ollama_host: str = OLLAMA_HOST
        self.embedding_dim: int = get_embedding_dim(self.embedding_func, self.embedding_model)
        self.cosine_better_than_threshold: float = 0.4
        resolved_base_dir = Path(base_dir) if base_dir is not None else dataset_db_dir()
        self._client_file_name = resolved_base_dir / f"vdb_{self.namespace}.json"
        self._max_batch_size = EMBEDDING_BATCH_NUM
        self._client = NanoVectorDB(self.embedding_dim, storage_file=str(self._client_file_name))
        self.query_top_k: int = 5

    async def upsert(self, data: dict[str, dict], need_embedding_list: bool = False) -> Any:
        logger.info("Inserting %s vectors to %s", len(data), self.namespace)
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [{"__id__": key, **value} for key, value in data.items()]
        contents = [value["content"] for value in data.values()]
        batches = [
            (contents[i : i + self._max_batch_size], list(range(i, min(i + self._max_batch_size, len(contents)))))
            for i in range(0, len(contents), self._max_batch_size)
        ]
        successful_data_list = []
        failed_indices_list = []
        pbar = tqdm_async(total=len(batches), desc="Processing embeddings in batches", unit="batch")
        for batch_contents, batch_indices in batches:
            try:
                result = await self.embedding_func(batch_contents, embed_model=self.embedding_model)
                embeddings = result["embeddings"]
                valid_indices = result["valid_indices"]
                for local_idx, global_idx in enumerate(batch_indices):
                    if local_idx in valid_indices:
                        valid_position = valid_indices.index(local_idx)
                        list_data[global_idx]["__vector__"] = embeddings[valid_position]
                        if need_embedding_list:
                            vector = embeddings[valid_position]
                            list_data[global_idx]["embedding"] = vector.tolist() if hasattr(vector, "tolist") else vector
                        successful_data_list.append(list_data[global_idx])
                    else:
                        failed_indices_list.append(global_idx)
                logger.info("Batch processed: %s/%s items embedded successfully", len(valid_indices), len(batch_contents))
            except Exception as exc:
                logger.error("Error processing batch with indices %s: %s", batch_indices, exc)
                failed_indices_list.extend(batch_indices)
            pbar.update(1)
        pbar.close()
        if successful_data_list:
            logger.info("Saving %s successfully embedded items to vector database", len(successful_data_list))
            results = self._client.upsert(datas=successful_data_list)
            logger.info("Successfully saved %s vectors to %s", len(successful_data_list), self.namespace)
            if failed_indices_list:
                logger.warning("Failed to embed %s items at indices: %s", len(failed_indices_list), failed_indices_list)
            return results
        logger.error("No items were successfully embedded. Failed: %s", len(failed_indices_list))
        return []

    async def query(self, query: str, top_k: int = 10, query_embedding=None):
        if query_embedding is not None:
            embedding = query_embedding
        else:
            result = await self.embedding_func([query], embed_model=self.embedding_model)
            embeddings = result["embeddings"]
            if embeddings.size == 0:
                logger.warning("Failed to embed query: %s", query)
                return []
            embedding = embeddings[0]
        results = self._client.query(query=embedding, top_k=top_k, better_than_threshold=self.cosine_better_than_threshold)
        return [{**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results]

    async def get_by_id(self, node_id: str):
        node_list = self._client.get([node_id])
        if node_list and len(node_list) != 0:
            return node_list[0]
        return None

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def index_done_callback(self):
        self._client.save()
