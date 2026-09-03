"""P0 日志模块单测：初始化、写入文件、命名空间级别、自动初始化。"""
import logging
from pathlib import Path

from src.utils.logger import get_logger, setup_logging

LOG_FILE = Path("logs/rag_engine.log")


def test_setup_and_file_write():
    setup_logging(force=True)
    log = get_logger("rag_engine.p0.smoke")
    assert log.getEffectiveLevel() == logging.DEBUG  # 从父级 rag_engine 继承
    log.info("hello from p0 smoke test")
    assert LOG_FILE.is_file()
    assert "hello from p0 smoke test" in LOG_FILE.read_text(encoding="utf-8")


def test_get_logger_autoconfig():
    # 不显式 setup 也能拿到可用 logger（内部自动初始化）
    log = get_logger("rag_engine.p0.auto")
    log.warning("auto configured ok")
    assert log.name == "rag_engine.p0.auto"
