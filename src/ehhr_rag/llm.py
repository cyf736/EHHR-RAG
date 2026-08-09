import asyncio
import inspect
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Callable, Generator, List, Union

import numpy as np
import ollama
import requests
import torch
from openai import APIConnectionError, RateLimitError, AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_fixed
from transformers import AutoTokenizer

from ehhr_rag.config import (
    ALI_API_KEY,
    ALI_BASE_URL,
    API_VERSION,
    ENDPOINT_URL,
    LLAMA_EMBEDDING_MODEL_NAME,
    LLAMA_TOKENIZER_MODEL_NAME,
    LLM_MODEL_MODE,
    LLM_NAME,
    MAX_TOKEN,
    OLLAMA_API_URL,
    REPO_ROOT,
    VV_API_KEY,
    VV_BASE_URL,
)
from ehhr_rag.logging_utils import gpt_logger, logger


@dataclass
class TokenStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    api_call_count: int = 0
    cache_hit_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_tokens(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens

    def add_api_call(self, cache_hit: bool = False):
        with self._lock:
            self.api_call_count += 1
            if cache_hit:
                self.cache_hit_count += 1

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "api_call_count": self.api_call_count,
                "cache_hit_count": self.cache_hit_count,
                "api_call_without_cache": self.api_call_count - self.cache_hit_count,
            }

    def reset(self):
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
            self.api_call_count = 0
            self.cache_hit_count = 0


GLOBAL_TOKEN_STATS = TokenStats()


def get_token_stats() -> TokenStats:
    return GLOBAL_TOKEN_STATS


def reset_token_stats():
    GLOBAL_TOKEN_STATS.reset()


CACHE_DB_PATH = REPO_ROOT / "dataset" / "llm_cache.db"
CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_cache_db():
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ali_api_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            UNIQUE(model, prompt)
        )
        """
    )
    conn.commit()
    conn.close()


init_cache_db()


def generate_gpt_log(prompt_input: Union[str, List[str]], prompt: str, output: str) -> str:
    return (
        "\n========================== START ===============================\n"
        "################## prompt_input ######################\n"
        f"{prompt_input}\n"
        "################## output ######################\n"
        f"{output}\n"
        "============================= END ================================\n\n\n"
    )


def generate_prompt(prompt_input: Union[str, List[str]], prompt_lib_file: str) -> str:
    if isinstance(prompt_input, str):
        prompt_input = [prompt_input]
    prompt_input = [str(item) for item in prompt_input]
    with open(prompt_lib_file, "r", encoding="utf-8") as handle:
        prompt = handle.read()
    for count, input_text in enumerate(prompt_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", input_text)
    if "<comment-block-marker>###</comment-block-marker>" in prompt:
        prompt = prompt.split("<comment-block-marker>###</comment-block-marker>")[1]
    return prompt.strip()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
def gpt_azure_request(prompt: str, model: str = LLM_NAME, max_tokens: int = MAX_TOKEN) -> str:
    client = AzureOpenAI(azure_endpoint=ENDPOINT_URL, api_key=os.environ.get("OPENAI_API_KEY", ""), api_version=API_VERSION)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def gpt_api_request(prompt: str, model: str = LLM_NAME, max_tokens: int = MAX_TOKEN) -> str:
    import openai

    openai.base_url = VV_BASE_URL
    openai.api_key = VV_API_KEY
    openai.default_headers = {"x-foo": "true"}
    completion = openai.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def gpt_ali_api_request(prompt: str, model: str = LLM_NAME, max_tokens: int = MAX_TOKEN, use_cache: bool = True) -> str:
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    if use_cache:
        cursor.execute("SELECT response FROM ali_api_cache WHERE model = ? AND prompt = ?", (model, prompt))
        result = cursor.fetchone()
        if result:
            conn.close()
            GLOBAL_TOKEN_STATS.add_api_call(cache_hit=True)
            logger.info("[gpt_ali_api_request] cache hit")
            return result[0]
    client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )
    prompt_tokens = res.usage.prompt_tokens if res.usage else 0
    completion_tokens = res.usage.completion_tokens if res.usage else 0
    GLOBAL_TOKEN_STATS.add_tokens(prompt_tokens, completion_tokens)
    GLOBAL_TOKEN_STATS.add_api_call(cache_hit=False)
    response_text = res.choices[0].message.content
    if use_cache:
        try:
            cursor.execute("INSERT INTO ali_api_cache (model, prompt, response) VALUES (?, ?, ?)", (model, prompt, response_text))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
    conn.close()
    return response_text


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
def gpt_ollama_request(
    prompt: str,
    model: str = LLM_NAME,
    max_tokens: int = MAX_TOKEN,
    temperature: float = 0.7,
    stream: bool = False,
    timeout: int = 120,
) -> Union[str, Generator[str, None, None]]:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stream": stream}
    response = requests.post(
        url=OLLAMA_API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        stream=stream,
        timeout=timeout,
    )
    response.raise_for_status()
    if not stream:
        return response.json().get("response", "")

    def stream_generator():
        for chunk in response.iter_lines():
            if not chunk:
                continue
            try:
                chunk_data = json.loads(chunk.decode("utf-8"))
                response_text = chunk_data.get("response", "")
                if response_text:
                    yield response_text
            except json.JSONDecodeError:
                continue

    return stream_generator()


def ollama_get_embedding_dim(embed_model: str = LLAMA_EMBEDDING_MODEL_NAME, **kwargs) -> int:
    ollama_client = ollama.Client(**kwargs)
    data = ollama_client.embeddings(model=embed_model, prompt="This is a dummy text for embedding dimension detection.")
    return len(data["embedding"])


DEFAULT_HF_EMBEDDING_MODEL = "BAAI/bge-m3"
HF_EMBEDDING_MODEL_CACHE = {}
HF_EMBEDDING_MODEL_CACHE_LOCK = threading.Lock()


def get_hf_embedding_model(embed_model: str, use_fp16: bool = True, cache_folder: str | None = None):
    cache_key = f"{embed_model}|fp16={use_fp16}|cache={cache_folder}"
    if cache_key not in HF_EMBEDDING_MODEL_CACHE:
        with HF_EMBEDDING_MODEL_CACHE_LOCK:
            if cache_key not in HF_EMBEDDING_MODEL_CACHE:
                model_kwargs = {}
                if use_fp16:
                    model_kwargs["torch_dtype"] = torch.bfloat16
                init_kwargs = {}
                if model_kwargs:
                    init_kwargs["model_kwargs"] = model_kwargs
                if cache_folder:
                    init_kwargs["cache_folder"] = cache_folder
                HF_EMBEDDING_MODEL_CACHE[cache_key] = SentenceTransformer(embed_model, **init_kwargs)
                if torch.cuda.is_available():
                    HF_EMBEDDING_MODEL_CACHE[cache_key].to("cuda")
    return HF_EMBEDDING_MODEL_CACHE[cache_key]


def get_embedding_dim(embed_func, embed_model: str = LLAMA_EMBEDDING_MODEL_NAME, **kwargs) -> int:
    dummy_text = ["This is a dummy text for embedding dimension detection."]
    if inspect.iscoroutinefunction(embed_func):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            result_container = []

            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    res = new_loop.run_until_complete(embed_func(dummy_text, embed_model=embed_model, **kwargs))
                    result_container.append(res)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()
            result = result_container[0]
        else:
            result = asyncio.run(embed_func(dummy_text, embed_model=embed_model, **kwargs))
    else:
        result = embed_func(dummy_text, embed_model=embed_model, **kwargs)
    emb_array = result["embeddings"]
    if emb_array.size == 0:
        return 0
    return emb_array.shape[1]


async def ollama_embedding(texts: list[str], embed_model: str = LLAMA_EMBEDDING_MODEL_NAME, **kwargs) -> dict:
    ollama_client = ollama.Client(**kwargs)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    async def _embed(text):
        data = await asyncio.to_thread(ollama_client.embeddings, model=embed_model, prompt=text)
        embedding = np.array(data["embedding"])
        if embedding.size == 0:
            raise ValueError(f"Empty embedding returned for text: {text[:50]}")
        return embedding

    embeddings = await asyncio.gather(*[_embed(text) for text in texts], return_exceptions=True)
    failed_indices = [idx for idx, item in enumerate(embeddings) if isinstance(item, Exception)]
    valid_embeddings = []
    valid_indices = []
    for idx, item in enumerate(embeddings):
        if isinstance(item, Exception):
            continue
        if isinstance(item, np.ndarray) and item.size > 0:
            valid_embeddings.append(item)
            valid_indices.append(idx)
    if not valid_embeddings:
        return {"embeddings": np.array([]), "failed_indices": failed_indices, "valid_indices": []}
    valid_embeddings = [np.array(item).reshape(1, -1) if item.ndim == 1 else item for item in valid_embeddings]
    return {"embeddings": np.vstack(valid_embeddings), "failed_indices": failed_indices, "valid_indices": valid_indices}


async def hf_embedding(texts: list[str], embed_model: str = DEFAULT_HF_EMBEDDING_MODEL, batch_size: int = 32, **kwargs) -> dict:
    if not isinstance(texts, list) or not all(isinstance(item, str) and item.strip() for item in texts):
        raise ValueError("`texts` must be a list of non-empty strings.")
    texts = [item.replace("\n", " ").strip() for item in texts]
    cache_folder = kwargs.get("cache_folder")
    use_fp16 = kwargs.get("use_fp16", True)
    model = (
        get_hf_embedding_model(embed_model=embed_model, use_fp16=use_fp16, cache_folder=cache_folder)
        if isinstance(embed_model, str)
        else embed_model
    )
    all_embeddings = []
    failed_indices = []
    valid_indices = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            emb_batch = await asyncio.to_thread(model.encode, batch, batch_size=batch_size, show_progress_bar=False)
            for offset, emb in enumerate(emb_batch):
                arr = np.array(emb)
                if arr.size == 0:
                    failed_indices.append(start + offset)
                else:
                    all_embeddings.append(arr)
                    valid_indices.append(start + offset)
        except Exception:
            failed_indices.extend(list(range(start, min(start + batch_size, len(texts)))))
    if not all_embeddings:
        return {"embeddings": np.array([]), "failed_indices": failed_indices, "valid_indices": []}
    all_embeddings = [np.array(item).reshape(1, -1) if np.array(item).ndim == 1 else np.array(item) for item in all_embeddings]
    return {"embeddings": np.vstack(all_embeddings).astype(np.float32), "failed_indices": failed_indices, "valid_indices": valid_indices}


def generate_from_prompt_template(
    prompt_input: Union[str, List[str]],
    prompt_lib_file: str,
    model: str = LLM_NAME,
    model_provider: str = LLM_MODEL_MODE,
    max_tokens: int = MAX_TOKEN,
    use_cache: bool = True,
) -> str:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    response = ""
    if model_provider == "azure":
        response = gpt_azure_request(prompt, model=model, max_tokens=max_tokens)
    elif model_provider == "ollama":
        response = gpt_ollama_request(prompt, model=model, max_tokens=max_tokens)
    elif model_provider == "vv":
        response = gpt_api_request(prompt, model=model, max_tokens=max_tokens)
    elif model_provider == "ali":
        response = gpt_ali_api_request(prompt, model=model, max_tokens=max_tokens, use_cache=use_cache)
    gpt_logger.info(generate_gpt_log(prompt_input, prompt, response))
    return response


LLAMA_TOKENIZER = None


def encode_string_by_llama(content: str, model_name: str = LLAMA_TOKENIZER_MODEL_NAME):
    global LLAMA_TOKENIZER
    if LLAMA_TOKENIZER is None:
        LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return LLAMA_TOKENIZER.encode(content)


def decode_tokens_by_llama(tokens: list[int], model_name: str = LLAMA_TOKENIZER_MODEL_NAME):
    global LLAMA_TOKENIZER
    if LLAMA_TOKENIZER is None:
        LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return LLAMA_TOKENIZER.decode(tokens)


def truncate_str_by_max_token_llama(input_str: str, max_token: int, model_name: str = LLAMA_TOKENIZER_MODEL_NAME):
    tokens = encode_string_by_llama(input_str, model_name=model_name)
    token_count = len(tokens)
    if token_count <= max_token:
        return input_str, token_count
    low, high, best_end = 0, len(input_str), 0
    while low <= high:
        mid = (low + high) // 2
        substr = input_str[:mid]
        current_count = len(encode_string_by_llama(substr, model_name=model_name))
        if current_count <= max_token:
            best_end = mid
            low = mid + 1
        else:
            high = mid - 1
    return input_str[:best_end], max_token


def truncate_list_by_token_size(list_data: list, key: Callable, max_token_size: int):
    if max_token_size <= 0:
        return []
    tokens = 0
    for idx, data in enumerate(list_data):
        tokens += len(encode_string_by_llama(key(data)))
        if tokens > max_token_size:
            return list_data[:idx]
    return list_data
