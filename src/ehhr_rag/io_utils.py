import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ehhr_rag.logging_utils import logger


def create_folder_if_not_there(curr_path: str | Path) -> bool:
    path = Path(curr_path)
    target_dir = path.parent if path.suffix else path
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        return True
    return False


def create_file_if_not_exists(file_path: str | Path):
    file_path = Path(file_path)
    if file_path.parent and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.write_text("", encoding="utf-8")
        return True
    return False


def write_dict_to_json(data: object, filename: str | Path):
    filename = Path(filename)
    create_folder_if_not_there(filename)
    with filename.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def read_json_to_dict(file_path: str | Path):
    file_path = Path(file_path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"The file at {file_path} was not found.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the file at {file_path}.")
    except Exception as exc:
        logger.error(f"An error occurred: {exc}")


def read_csv_to_dict_list(file_path: str | Path) -> list[dict]:
    with Path(file_path).open("r", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def process_csv_by_name_key_to_dict(file_path: str | Path, key_for_dict: str = "name"):
    result = defaultdict(list)
    with Path(file_path).open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if key_for_dict not in fieldnames:
            raise ValueError(f"CSV file does not contain '{key_for_dict}'")
        for row in reader:
            result[row[key_for_dict]].append({key: value for key, value in row.items() if key != key_for_dict})
    return dict(result)


def write_dict_list_to_csv(
    dict_list: list[dict],
    file_path: str | Path,
    encoding: str = "utf-8",
    mode: str = "a",
    write_header: bool = True,
    delimiter: str = ",",
    fieldnames: list[str] | None = None,
) -> None:
    if not all(isinstance(item, dict) for item in dict_list):
        raise TypeError("dict_list must contain only dict entries")
    file_path = Path(file_path)
    file_exists = file_path.exists()
    file_is_empty = (not file_exists) or file_path.stat().st_size == 0
    if not dict_list:
        if write_header and file_is_empty:
            with file_path.open("w", encoding=encoding, newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames or [], delimiter=delimiter)
                writer.writeheader()
        return
    if fieldnames is None:
        all_keys = set()
        for item in dict_list:
            all_keys.update(item.keys())
        fieldnames = sorted(all_keys)
    with file_path.open(mode, encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        if write_header and file_is_empty:
            writer.writeheader()
        writer.writerows(dict_list)


def list_of_list_to_csv(data: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> list[list[str]]:
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def extract_column_from_csv(csv_text: str, column_name: str):
    df = pd.read_csv(io.StringIO(csv_text.strip()))
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found. Available columns: {list(df.columns)}")
    return df[column_name].tolist()


def process_combine_contexts(high_level: str, low_level: str) -> str:
    header = None
    list_hl = csv_string_to_list(high_level.strip())
    list_ll = csv_string_to_list(low_level.strip())
    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""
    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]
    combined_sources = []
    seen = set()
    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)
    combined_sources_result = [",\t".join(header)]
    for idx, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{idx},\t{item}")
    return "\n".join(combined_sources_result)
