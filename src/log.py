import logging
import sys

# Optional: Initialize colorama for Windows support of ANSI escape codes.
try:
    import colorama
    colorama.init()
except ImportError:
    pass

# Define log filenames
FILENAME = "evaluation.log"

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds red coloring to ERROR and CRITICAL log messages
    for the console output.
    """
    RED = "\033[91m"
    RESET = "\033[0m"
    
    def format(self, record):
        message = super(ColoredFormatter, self).format(record)
        if record.levelno >= logging.ERROR:
            message = f"{self.RED}{message}{self.RESET}"
        return message

class FlushStreamHandler(logging.StreamHandler):
    """
    Custom stream handler that flushes the terminal output after emitting each record.
    """
    def emit(self, record):
        super(FlushStreamHandler, self).emit(record)
        self.flush()

class Log:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Log, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, stdout=True):
        if self._initialized:
            return
        
        self.logger = logging.getLogger("ModalityEval")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        file_handler = logging.FileHandler(FILENAME)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        if stdout:
            console_handler = FlushStreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
        
        self._initialized = True


if __name__ == "__main__":
    Log = Log().logger
    
    Log.debug("This DEBUG message will not be shown (INFO level set).")
    Log.info("This is an INFO message.")
    Log.warning("This is a WARNING message.")
    Log.error("This is an ERROR message.")
    Log.critical("This is a CRITICAL message.")
