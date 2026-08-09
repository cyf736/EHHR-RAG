import asyncio
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import spacy
from tqdm import tqdm

from ehhr_rag.config import LLAMA_TOKENIZER_MODEL_NAME, dataset_db_dir, dataset_raw_dir
from ehhr_rag.indexing.macro_graph_builder import MacroGraphBuilder
from ehhr_rag.indexing.micro_graph_builder import MicroGraphBuilder
from ehhr_rag.llm import encode_string_by_llama, get_token_stats, reset_token_stats
from ehhr_rag.logging_utils import logger
from ehhr_rag.storage.networkx_graph_storage import NetworkXStorage
from ehhr_rag.text_utils import compute_md5_hash_id

FIXED_CHUNK_TOKEN_LIMIT = 512


def get_chunks_cache_path(csv_path: str, db_base_dir: str) -> str:
    """Build the cache path for chunked documents derived from a CSV file."""
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    cache_dir = os.path.join(db_base_dir, "chunks_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{csv_basename}_chunks.json")


def save_chunks_to_json(chunks: list, output_path: str) -> None:
    """Persist chunk metadata to a JSON cache file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)
    logger.info("Saved chunk cache to %s (%s chunks)", output_path, len(chunks))


def load_chunks_from_json(input_path: str) -> list | None:
    """Load cached chunks from JSON if the file exists and is readable."""
    if not os.path.exists(input_path):
        return None

    try:
        with open(input_path, "r", encoding="utf-8") as handle:
            chunks = json.load(handle)
        logger.info("Loaded chunk cache from %s (%s chunks)", input_path, len(chunks))
        return chunks
    except Exception as exc:
        logger.warning("Failed to load chunk cache from %s: %s", input_path, exc)
        return None


def is_chunk_cache_compatible(chunks: list, token_limit: int = FIXED_CHUNK_TOKEN_LIMIT) -> bool:
    """Validate whether cached chunks match the current fixed token limit."""
    if not isinstance(chunks, list) or not chunks:
        return False
    for item in chunks:
        token_count = int(item.get("token_count", 0) or 0)
        if token_count > token_limit:
            return False
    return True


def suppress_ollama_console_logs() -> None:
    logging.getLogger("ollama").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items(), key=lambda item: item[0]))


def get_graph_stats(db_base_dir: str, namespace: str) -> dict:
    graph = NetworkXStorage(namespace=namespace, base_dir=db_base_dir).return_self_graph()

    node_role = Counter()
    node_type = Counter()
    edge_relation = Counter()

    for _, data in graph.nodes(data=True):
        node_role[str(data.get("role", "unknown"))] += 1
        node_type[str(data.get("type", "unknown"))] += 1
    for _, _, data in graph.edges(data=True):
        edge_relation[str(data.get("relation", "unknown"))] += 1

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "node_roles": dict(node_role),
        "node_types": dict(node_type),
        "edge_relations": dict(edge_relation),
    }


def print_graph_construction_stats(db_base_dir: str) -> None:
    micro_stats = get_graph_stats(db_base_dir, "layer_micro")
    macro_stats = get_graph_stats(db_base_dir, "layer_macro")

    logger.info("=" * 80)
    logger.info("Hypergraph construction statistics")
    logger.info("=" * 80)
    logger.info(
        "Micro-fact hypergraph: nodes=%s, edges=%s, node_roles=(%s), node_types=(%s), edge_relations=(%s)",
        micro_stats["node_count"],
        micro_stats["edge_count"],
        _format_counter(Counter(micro_stats["node_roles"])),
        _format_counter(Counter(micro_stats["node_types"])),
        _format_counter(Counter(micro_stats["edge_relations"])),
    )
    logger.info(
        "Macro-theme hypergraph: nodes=%s, edges=%s, node_roles=(%s), node_types=(%s), edge_relations=(%s)",
        macro_stats["node_count"],
        macro_stats["edge_count"],
        _format_counter(Counter(macro_stats["node_roles"])),
        _format_counter(Counter(macro_stats["node_types"])),
        _format_counter(Counter(macro_stats["edge_relations"])),
    )


def count_sentences_from_chunks(chunks: list) -> int:
    return sum(chunk.get("sentence_count", 0) for chunk in chunks)


def generate_construction_report(
    db_base_dir: str,
    chunks: list,
    timing_stats: dict,
    output_path: str | None = None,
) -> dict:
    micro_stats = get_graph_stats(db_base_dir, "layer_micro")
    macro_stats = get_graph_stats(db_base_dir, "layer_macro")
    token_stats = get_token_stats().get_stats()
    total_sentences = count_sentences_from_chunks(chunks)

    report = {
        "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_summary": {
            "total_documents": len(set(chunk.get("document_title", "") for chunk in chunks)),
            "total_chunks": len(chunks),
            "total_sentences": total_sentences,
            "avg_sentences_per_chunk": round(total_sentences / len(chunks), 2) if chunks else 0,
        },
        "timing": {
            "total_seconds": round(timing_stats.get("total", 0), 2),
            "total_minutes": round(timing_stats.get("total", 0) / 60, 2),
            "micro_graph_seconds": round(timing_stats.get("micro", 0), 2),
            "macro_graph_seconds": round(timing_stats.get("macro", 0), 2),
        },
        "token_usage": {
            "total_prompt_tokens": token_stats["total_prompt_tokens"],
            "total_completion_tokens": token_stats["total_completion_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "api_call_count": token_stats["api_call_count"],
            "cache_hit_count": token_stats["cache_hit_count"],
            "api_call_without_cache": token_stats["api_call_without_cache"],
        },
        "micro_graph": {
            "node_count": micro_stats["node_count"],
            "edge_count": micro_stats["edge_count"],
            "node_types": micro_stats["node_types"],
            "edge_relations": micro_stats["edge_relations"],
        },
        "macro_graph": {
            "node_count": macro_stats["node_count"],
            "edge_count": macro_stats["edge_count"],
            "node_types": macro_stats["node_types"],
            "edge_relations": macro_stats["edge_relations"],
        },
        "db_base_dir": db_base_dir,
    }

    if output_path is None:
        output_path = os.path.join(db_base_dir, "construction_report.json")

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("[%s] Hypergraph construction report", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)
    logger.info("Data summary:")
    logger.info("  - documents: %s", report["data_summary"]["total_documents"])
    logger.info("  - chunks: %s", report["data_summary"]["total_chunks"])
    logger.info("  - sentences: %s", report["data_summary"]["total_sentences"])
    logger.info("  - avg sentences per chunk: %s", report["data_summary"]["avg_sentences_per_chunk"])
    logger.info("Timing:")
    logger.info(
        "  - total: %s minutes (%s seconds)",
        report["timing"]["total_minutes"],
        report["timing"]["total_seconds"],
    )
    logger.info("  - micro graph: %s seconds", report["timing"]["micro_graph_seconds"])
    logger.info("  - macro graph: %s seconds", report["timing"]["macro_graph_seconds"])
    logger.info("Token usage (LLM calls):")
    logger.info("  - prompt tokens: %s", report["token_usage"]["total_prompt_tokens"])
    logger.info("  - completion tokens: %s", report["token_usage"]["total_completion_tokens"])
    logger.info("  - total tokens: %s", report["token_usage"]["total_tokens"])
    logger.info(
        "  - API calls: %s (cache hits: %s)",
        report["token_usage"]["api_call_count"],
        report["token_usage"]["cache_hit_count"],
    )
    logger.info("Micro-fact hypergraph:")
    logger.info("  - nodes: %s", report["micro_graph"]["node_count"])
    logger.info("  - edges: %s", report["micro_graph"]["edge_count"])
    logger.info("  - node types: %s", report["micro_graph"]["node_types"])
    logger.info("  - edge relations: %s", report["micro_graph"]["edge_relations"])
    logger.info("Macro-theme hypergraph:")
    logger.info("  - nodes: %s", report["macro_graph"]["node_count"])
    logger.info("  - edges: %s", report["macro_graph"]["edge_count"])
    logger.info("  - node types: %s", report["macro_graph"]["node_types"])
    logger.info("  - edge relations: %s", report["macro_graph"]["edge_relations"])
    logger.info("Report file: %s", output_path)
    logger.info("=" * 80)

    return report


def chunk_documents_in_memory(
    nlp,
    df: pd.DataFrame,
    max_token_per_chunk: int = FIXED_CHUNK_TOKEN_LIMIT,
    model_name: str = LLAMA_TOKENIZER_MODEL_NAME,
) -> list:
    """Chunk documents in memory with a fixed token budget per chunk."""
    chunks_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        document_title = row.get("title", "")
        document_text = row.get("text", "")

        if pd.isna(document_text) or not isinstance(document_text, str):
            continue

        doc = nlp(document_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        if not sentences:
            continue

        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = encode_string_by_llama(sentence, model_name=model_name)
            sentence_token_count = len(sentence_tokens)

            if not current_chunk:
                current_chunk.append(sentence)
                current_token_count = sentence_token_count
            elif current_token_count + sentence_token_count <= max_token_per_chunk:
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
            else:
                chunk_text = " ".join(current_chunk)
                chunk_id = compute_md5_hash_id(f"{document_title}:{chunk_text}", prefix="<chunk>")
                chunks_list.append(
                    {
                        "document_title": document_title,
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "token_count": current_token_count,
                        "sentence_count": len(current_chunk),
                    }
                )
                current_chunk = [sentence]
                current_token_count = sentence_token_count

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = compute_md5_hash_id(f"{document_title}:{chunk_text}", prefix="<chunk>")
            chunks_list.append(
                {
                    "document_title": document_title,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "token_count": current_token_count,
                    "sentence_count": len(current_chunk),
                }
            )

    return chunks_list


async def build_hypergraph_pipeline(csv_path: str, db_base_dir: str, use_cached_chunks: bool = True):
    suppress_ollama_console_logs()
    logger.info("Starting dual-layer hypergraph construction (Micro-Fact & Macro-Theme)...")

    reset_token_stats()
    pipeline_start_time = time.time()

    chunks_cache_path = get_chunks_cache_path(csv_path, db_base_dir)
    chunks = None

    if use_cached_chunks:
        logger.info("Trying chunk cache at %s", chunks_cache_path)
        chunks = load_chunks_from_json(chunks_cache_path)
        if chunks is not None and not is_chunk_cache_compatible(chunks, FIXED_CHUNK_TOKEN_LIMIT):
            logger.info(
                "Chunk cache is incompatible with the current fixed token limit (<= %s). Regenerating chunks.",
                FIXED_CHUNK_TOKEN_LIMIT,
            )
            chunks = None

    if chunks is None:
        logger.info("Chunk cache not available. Generating chunks from the source CSV...")
        logger.info("Loading spaCy model for sentence segmentation...")
        nlp = spacy.load("en_core_web_lg")

        logger.info("Reading source CSV: %s", csv_path)
        df = pd.read_csv(csv_path)
        chunks = chunk_documents_in_memory(nlp, df, max_token_per_chunk=FIXED_CHUNK_TOKEN_LIMIT)
        logger.info("Fixed chunk token limit: %s", FIXED_CHUNK_TOKEN_LIMIT)
        logger.info("Generated %s chunks.", len(chunks))

        if use_cached_chunks:
            save_chunks_to_json(chunks, chunks_cache_path)

    micro_start = time.time()
    micro_builder = MicroGraphBuilder(db_base_dir=db_base_dir, namespace="layer_micro")
    await micro_builder.build_from_chunks(chunks)
    micro_time = time.time() - micro_start
    logger.info("Micro-fact hypergraph construction finished in %.2f seconds", micro_time)

    logger.info("Starting macro-theme hypergraph construction...")
    macro_start = time.time()
    macro_builder = MacroGraphBuilder(db_base_dir=db_base_dir, namespace="layer_macro")
    await macro_builder.build_from_chunks(chunks)
    macro_time = time.time() - macro_start
    logger.info("Macro-theme hypergraph construction finished in %.2f seconds", macro_time)

    total_time = time.time() - pipeline_start_time

    print_graph_construction_stats(db_base_dir)
    generate_construction_report(
        db_base_dir=db_base_dir,
        chunks=chunks,
        timing_stats={"total": total_time, "micro": micro_time, "macro": macro_time},
    )

    logger.info("Dual-layer hypergraph construction completed.")


if __name__ == "__main__":
    dataset_name = "hotpot"
    csv_file = dataset_raw_dir(dataset_name) / f"{dataset_name}_1000_sampled_ctx.csv"
    db_dir = dataset_db_dir(dataset_name)

    os.makedirs(db_dir, exist_ok=True)

    if os.path.exists(csv_file):
        asyncio.run(build_hypergraph_pipeline(str(csv_file), str(db_dir)))
    else:
        logger.error("Raw dataset file not found: %s", csv_file)
