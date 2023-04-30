import logging
import sys
import csv
import time

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


def write_dict_to_csv_with_timestamp(dict_to_write, filename):
    """
    Writes a dictionary to CSV with a timestamp column
    """
    with open(filename, mode='a', newline='') as csv_file:
        fieldnames = list(dict_to_write.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header row if file is empty
        csv_file.seek(0)
        first_char = csv_file.read(1)
        if not first_char:
            writer.writeheader()

        # Write data row with timestamp
        dict_to_write['timestamp'] = time.time()
        writer.writerow(dict_to_write)


def zero_out_dict_values(d):
    """Set all values of a dictionary to zero."""
    for key in d:
        d[key] = 0