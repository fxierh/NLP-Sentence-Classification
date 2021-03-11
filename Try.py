import logging
import time

from ANSI_color_codes import *

# ANSI color codes (available on MacOS)
# RESET = '\033[0m'
# BOLD = '\033[1m'
# DISABLE = '\033[02m'
# UNDERLINE = '\033[4m'
# REVERSE = '\033[07m'
# INVISIBLE = '\033[08m'
# STRIKETHROUGH = '\033[09m'
# BLACK = '\033[30m'
# RED = '\033[31m'
# GREEN = '\033[32m'
# YELLOW = '\033[33m'
# BLUE = '\033[34m'
# MAGENTA = '\033[35m'
# CYAN = '\033[36m'
# WHITE = '\033[37m'
# FAIL = '\033[91m'
# OKGREEN = '\033[92m'
# WARNING = '\033[93m'
# OKBLUE = '\033[94m'
# HEADER = '\033[95m'
# OKCYAN = '\033[96m'

# Create and configure logger
# noinspection PyArgumentList
# logging.basicConfig(
#     level=logging.DEBUG,  # Also possible: logger.setLevel(logging.DEBUG)
#     format=CYAN + BOLD + '%(asctime)s - %(name)s - %(filename)s - %(lineno)s - %(levelname)s - %(message)s' + RESET,
#     handlers=[logging.StreamHandler(),
#               logging.FileHandler(time.strftime("%Y%m%d_%H%M%S") + ".log")
#               ]  # Also possible: logger.addHandler(logging.StreamHandler())
# )

logging.basicConfig(
    filename="./Logging/" + time.strftime("%Y%m%d_%H%M%S") + ".log",
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(filename)s - line %(lineno)s - %(levelname)s - %(message)s',
    datefmt="%Z %Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(OKBLUE + '%(asctime)s - %(message)s' + RESET))
logger.addHandler(console_handler)

# Test messages
logger.debug("Harmless debug Message")
logger.info("Just an information")
logger.warning("Its a Warning")
logger.error("Did you try to divide by zero")
logger.critical("Internet is down")
