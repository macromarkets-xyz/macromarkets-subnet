import logging
from logging.handlers import RotatingFileHandler
from pylogrus import PyLogrus, TextFormatter, JsonFormatter


class Logmaster:
    def __init__(self, log_file, max_bytes=10 * 1024 * 1024, backup_count=5):
        # Create logger
        self.logger = logging.getLogger("logmaster")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Create console handler with pretty printer
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(TextFormatter(colorize=True))

        # Create file handler with JSON formatter and log rotation
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())

        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Store handlers for dynamic log level changes
        self.handlers = {"console": console_handler, "file": file_handler}

        # Log level lookup
        self.log_methods = {
            "info": self.logger.info,
            "error": self.logger.error,
            "warn": self.logger.warning,
            "debug": self.logger.debug,
            "fatal": self.logger.fatal,
        }

    def log(self, level, msg, **kwargs):
        extra = {"extra": kwargs}
        log_method = self.log_methods.get(level)
        if log_method:
            log_method(msg, **extra)
        else:
            raise ValueError("Invalid log level")

    def info(self, msg, **kwargs):
        self.log("info", msg, **kwargs)

    def error(self, msg, **kwargs):
        self.log("error", msg, **kwargs)

    def warn(self, msg, **kwargs):
        self.log("warn", msg, **kwargs)

    def debug(self, msg, **kwargs):
        self.log("debug", msg, **kwargs)

    def fatal(self, msg, **kwargs):
        self.log("fatal", msg, **kwargs)

    def set_log_level(self, handler_name, level):
        handler = self.handlers.get(handler_name)
        if handler:
            handler.setLevel(level)
        else:
            raise ValueError("Invalid handler name")


# Example usage
logmaster = Logmaster("logfile.json")
logmaster.info("This is an info message", foo="bar", user="Alice")
logmaster.error("This is an error message", error_code=500, path="/api/data")
logmaster.debug("Debugging application", state="initializing")

# Dynamically change log levels
logmaster.set_log_level("console", logging.WARNING)
logmaster.set_log_level("file", logging.ERROR)
