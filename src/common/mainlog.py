import os
import sys
from loguru import logger
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConsoleFormatter:
    def __init__(self):
        self.fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{message}</cyan>"
        )

    def format(self, record):
        fmt = self.fmt
        for k, v in record["extra"].items():
            fmt = f"{fmt} | {k}={v}"
        return f"{fmt}\n"


class MainLogger:
    def __init__(self, log_path="/tmp/json_logs", extra_fields=None):
        self.log_path = log_path
        logger.remove()
        self.logger = logger
        self.extra_fields = extra_fields or {}
        self.json_log_level = LogLevel.INFO
        self.console_log_level = LogLevel.INFO
        self.json_handler_id = self.setup_json_logger()
        self.console_handler_id = self.setup_console_logger()

    def setup_json_logger(self):
        log_file = os.path.join(self.log_path, "{time:UNIX}.json")
        return self.logger.add(
            log_file,
            rotation="100 MB",
            serialize=True,
            enqueue=True,
            level=self.json_log_level.value,
        )

    def setup_console_logger(self):
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{message}</cyan> | "
            "{extra}"
        )
        formatter = ConsoleFormatter()
        return self.logger.add(
            sys.stderr,
            # format=console_format,
            format=formatter.format,
            colorize=True,
            enqueue=True,
            level=self.console_log_level.value,
            # serialize=False,
            # filter=lambda record: record["extra"].update({"extra": self.format_extra(record)})
        )

    def set_json_log_level(self, level: LogLevel):
        self.json_log_level = level
        self.logger.remove(self.json_handler_id)
        self.json_handler_id = self.setup_json_logger()

    def set_console_log_level(self, level: LogLevel):
        self.console_log_level = level
        self.logger.remove(self.console_handler_id)
        self.console_handler_id = self.setup_console_logger()

    def log(self, level: LogLevel, message, **kwargs):
        self.logger.log(level.value, message, **kwargs)

    def debug(self, message, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message, **kwargs):
        self.log(LogLevel.CRITICAL, message, **kwargs)


# Usage example
if __name__ == "__main__":
    logger = MainLogger()

    logger.info("This log only")

    logger.set_json_log_level(LogLevel.DEBUG)
    logger.debug("This debug message should appear in the JSON log")

    logger.info("Log with extra field", extra_field="extra_value")
