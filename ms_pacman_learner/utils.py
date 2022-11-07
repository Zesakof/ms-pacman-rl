import os
from datetime import datetime as dt

def setup_dirs(logging_base_path: str, include_subfolders: bool):

    # Create folder to store trajectories
    timestamp = dt.strftime(dt.now(), '%Y%m%d_%H%M%S')
    logging_path = os.path.join(logging_base_path, timestamp)
    os.mkdir(logging_path)

    if include_subfolders:
        os.mkdir(os.path.join(logging_path, 'images'))

    return logging_path
