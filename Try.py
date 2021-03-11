import logging
import time

from ANSI_color_codes import *

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
    datefmt="%Z %Y-%m-%d %H:%M:%S"  # Time zone yyyy-MM-dd HH:mm:ss.SSS
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(OKBLUE + '%(asctime)s - %(message)s' + RESET, "%H:%M:%S"))
logger.addHandler(console_handler)

# Test messages
logger.debug("Harmless debug Message")
logger.info("Just an information")
logger.warning("Its a Warning")
logger.error("Did you try to divide by zero")
logger.critical("Internet is down")
logger.debug("Harmless debug Message")
