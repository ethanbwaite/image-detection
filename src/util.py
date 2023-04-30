import logging
import sys

def get_logger():
    # Create a logger
    logger = logging.getLogger(__name__)

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create a StreamHandler with stdout as the stream
    stream_handler = logging.StreamHandler(sys.stdout)

    # Set the logging level for the handler
    stream_handler.setLevel(logging.DEBUG)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the handler
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)

    return logger


def add_history(current_history, new_record, max_history=10):
    """
    Adds an object to a history queue. Pops a value from the front if the max history
    size is reached
    """
    current_history.append(new_record)
    if len(current_history) > max_history:
        current_history.pop(0)
    return current_history