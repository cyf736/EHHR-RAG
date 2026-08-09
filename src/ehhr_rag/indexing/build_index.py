import argparse
import asyncio
from pathlib import Path

from ehhr_rag.config import dataset_db_dir, dataset_raw_dir
from ehhr_rag.indexing.pipeline import build_hypergraph_pipeline
from ehhr_rag.logging_utils import logger


def main():
    parser = argparse.ArgumentParser(description="Build EHHR retrieval indexes from a dataset context CSV.")
    parser.add_argument("--dataset", required=True, help="Dataset name under dataset/.")
    parser.add_argument("--context-csv", default=None, help="Optional explicit context CSV path.")
    parser.add_argument("--no-cache", action="store_true", help="Disable chunk cache reuse.")
    args = parser.parse_args()

    context_csv = Path(args.context_csv) if args.context_csv else dataset_raw_dir(args.dataset) / f"{args.dataset}_1000_sampled_ctx.csv"
    db_dir = dataset_db_dir(args.dataset)
    db_dir.mkdir(parents=True, exist_ok=True)
    if not context_csv.exists():
        raise FileNotFoundError(f"Context CSV not found: {context_csv}")
    logger.info("Building indexes for dataset=%s from %s", args.dataset, context_csv)
    asyncio.run(build_hypergraph_pipeline(str(context_csv), str(db_dir), use_cached_chunks=not args.no_cache))


if __name__ == "__main__":
    main()
