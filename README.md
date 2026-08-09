# EHHR-RAG

A cleaned reproducibility release for the EHHR evidence-aware RAG pipeline.

## Layout

- `dataset/<name>/raw`: raw dataset files
- `dataset/<name>/db`: generated retrieval indexes
- `dataset/<name>/outputs`: generated prediction and report files
- `prompts/`: prompt templates used by indexing and evidence-aware reasoning
- `src/ehhr_rag/`: package source

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Build indexes from a dataset context CSV:

```bash
python -m ehhr_rag.indexing.build_index --dataset hotpot --context-csv dataset/hotpot/raw/hotpot_1000_sampled_ctx.csv
```

Run evidence-aware inference:

```bash
python -m ehhr_rag.evaluation.run_evidence_aware --dataset hotpot --qa-file dataset/hotpot/raw/hotpot_1000_sampled_qa.json --output dataset/hotpot/outputs/hotpot_1000_ehhr.json
```
