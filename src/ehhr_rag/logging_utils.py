import logging
from pathlib import Path

from ehhr_rag.config import CHAIN_LOG_FILEPATH, GPT_LOG_FILEPATH, SYSTEM_LOG_FILEPATH


def setup_logger(name: str, is_need_print: bool = True, log_file_path: Path | None = None) -> logging.Logger:
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    if not _logger.handlers:
        if log_file_path:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            _logger.addHandler(file_handler)
        if is_need_print:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            _logger.addHandler(console_handler)
    return _logger


logger = setup_logger("system", is_need_print=True, log_file_path=SYSTEM_LOG_FILEPATH)
gpt_logger = setup_logger("gpt", is_need_print=True, log_file_path=GPT_LOG_FILEPATH)
chain_logger = setup_logger("chain", is_need_print=True, log_file_path=CHAIN_LOG_FILEPATH)
