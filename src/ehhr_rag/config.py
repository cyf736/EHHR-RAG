import os
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"')


_load_env_file()

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

DATASET_ROOT = REPO_ROOT / "dataset"
PROMPTS_ROOT = REPO_ROOT / "prompts"
LOG_ROOT = REPO_ROOT / "logs"
LOG_ROOT.mkdir(exist_ok=True)

DEFAULT_DATASET = os.environ.get("EHHR_DATASET", "hotpot")

ENDPOINT_URL = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://gpt-wxy-east-us.openai.azure.com/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL_MODE = os.environ.get("LLM_MODEL_MODE", "ali")
LLM_NAME = os.environ.get("LLM_NAME", "qwen3-14b")
MAX_TOKEN = int(os.environ.get("MAX_TOKEN", "8000"))
MAX_INPUT_TOKEN = int(os.environ.get("MAX_INPUT_TOKEN", "3500"))

MAX_ONE_KNOWLEDGE_UNIT_TOKEN = int(os.environ.get("MAX_ONE_KNOWLEDGE_UNIT_TOKEN", "800"))
MAX_KNOWLEDGE_UNITS_TOKEN = int(os.environ.get("MAX_KNOWLEDGE_UNITS_TOKEN", "2048"))
LLAMA_TOKENIZER_MODEL_NAME = os.environ.get("LLAMA_TOKENIZER_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
LLAMA_EMBEDDING_MODEL_NAME = os.environ.get("LLAMA_EMBEDDING_MODEL_NAME", "bge-m3")

ALI_BASE_URL = os.environ.get("ALI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
ALI_API_KEY = os.environ.get("ALI_API_KEY", "")

VV_BASE_URL = os.environ.get("VV_BASE_URL", "https://api.vveai.com/v1/")
VV_API_KEY = os.environ.get("VV_API_KEY", "")

GPT_LOG_FILEPATH = LOG_ROOT / "gpt_log.txt"
SYSTEM_LOG_FILEPATH = LOG_ROOT / "system_log.txt"
CHAIN_LOG_FILEPATH = LOG_ROOT / "chain_log.txt"

EMBEDDING_BATCH_NUM = int(os.environ.get("EMBEDDING_BATCH_NUM", "32"))
GRAPH_FIELD_SEP = "<SEP>"

PCST_TOPK = int(os.environ.get("PCST_TOPK", "15"))
PCST_TOPK_E = int(os.environ.get("PCST_TOPK_E", "2"))
PCST_NUM_CLUSTERS = int(os.environ.get("PCST_NUM_CLUSTERS", "2"))
PCST_ALPHA = float(os.environ.get("PCST_ALPHA", "0.8"))
PCST_BETA = float(os.environ.get("PCST_BETA", "0.1"))
THINK_CHAIN_LOOPS = int(os.environ.get("THINK_CHAIN_LOOPS", "8"))


def dataset_dir(dataset_name: str | None = None) -> Path:
    return DATASET_ROOT / (dataset_name or DEFAULT_DATASET)


def dataset_raw_dir(dataset_name: str | None = None) -> Path:
    return dataset_dir(dataset_name) / "raw"

def dataset_db_dir(dataset_name: str | None = None) -> Path:
    return dataset_dir(dataset_name) / "db"


def dataset_outputs_dir(dataset_name: str | None = None) -> Path:
    return dataset_dir(dataset_name) / "outputs"


def prompt_dir(subdir: str | None = None) -> Path:
    return PROMPTS_ROOT if subdir is None else PROMPTS_ROOT / subdir
