import logging
import sys
import asyncio
from fastapi import WebSocket


class WebSocketLogHandler(logging.Handler):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def emit(self, record):
        log_entry = self.format(record)
        # Use asyncio.create_task to send the message without blocking the logger
        asyncio.create_task(self.manager.broadcast(log_entry))


def setup_logging(log_manager=None):
    """
    Configure structured JSON logging.
    """
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handlers = [handler]
    if log_manager:
        ws_handler = WebSocketLogHandler(log_manager)
        ws_handler.setFormatter(formatter)
        handlers.append(ws_handler)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.info("Structured logging configured.")