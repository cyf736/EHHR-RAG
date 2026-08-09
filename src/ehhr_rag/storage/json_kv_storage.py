from pathlib import Path

from ehhr_rag.config import dataset_db_dir
from ehhr_rag.io_utils import read_json_to_dict, write_dict_to_json
from ehhr_rag.logging_utils import logger


class JsonKVStorage:
    def __init__(self, namespace: str, base_dir: str | Path | None = None):
        self.namespace = namespace
        resolved_base_dir = Path(base_dir) if base_dir is not None else dataset_db_dir()
        self._file_name = resolved_base_dir / f"kv_{self.namespace}.json"
        self._data = read_json_to_dict(self._file_name) or {}
        logger.info("Load KV %s with %s data", self.namespace, len(self._data))

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        await self.save()

    async def save(self):
        write_dict_to_json(self._data, self._file_name)

    async def get_by_id(self, _id):
        return self._data.get(_id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(_id, None) for _id in ids]
        return [
            ({k: v for k, v in self._data[item_id].items() if k in fields} if self._data.get(item_id, None) else None)
            for item_id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return {value for value in data if value not in self._data}

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}
