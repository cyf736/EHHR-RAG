import json
import random
import re
import string
from hashlib import md5
from typing import Any

from ehhr_rag.logging_utils import logger


def compute_md5_hash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def generate_alphanumeric_string(length: int) -> str:
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def extract_first_json_dict(input_str: str) -> Any | None:
    try:
        start_index = input_str.index("{")
        count = 1
        end_index = start_index + 1
        while count > 0 and end_index < len(input_str):
            if input_str[end_index] == "{":
                count += 1
            elif input_str[end_index] == "}":
                count -= 1
            end_index += 1
        return json.loads(input_str[start_index:end_index])
    except ValueError:
        logger.error("Failed to extract JSON dict from text")
        return None


def extract_first_json_list_from_str(input_str: str) -> list | None:
    try:
        start_index = input_str.find("[")
        if start_index == -1:
            logger.error("Failed to find JSON list start marker")
            return None
        bracket_count = 1
        end_index = start_index + 1
        while end_index < len(input_str) and bracket_count > 0:
            char = input_str[end_index]
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            end_index += 1
        if bracket_count != 0:
            logger.error("JSON list brackets are not balanced")
            return None
        return json.loads(input_str[start_index:end_index].strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Extract JSON list failed: {exc}") from exc


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def find_first_yes_no(text: str) -> bool:
    match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return False


def parse_hyperedge_name(name: str):
    match = re.search(r"\[\[(.+?)]]$", name)
    if match:
        desc_part = match.group(1)
        sentence_part = name[:match.start()].strip()
        desc_parts = [part.strip() for part in desc_part.split(" | ") if part.strip()]
        title_desc_pairs = []
        for part in desc_parts:
            if "->" in part:
                title, desc = part.split("->", 1)
                title_desc_pairs.append((title.strip(), desc.strip()))
        return sentence_part, title_desc_pairs
    return name, []


def remove_substring_duplicates(strings: list[str]) -> list[str]:
    if not strings:
        return []
    sorted_strings = sorted(strings, key=len, reverse=True)
    result = []
    for item in sorted_strings:
        if not any(item in existing for existing in result):
            result.append(item)
    return result
