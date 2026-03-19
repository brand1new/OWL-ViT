from datetime import datetime
import os

def set_log_level(is_verbose: bool=False):
    os.environ['LOG_LEVEL_VERBOSE'] = str(is_verbose)

def get_readable_time():
    now = datetime.now()
    readable_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return readable_time

def log_info(str):
    print(f"[{get_readable_time()}] {str}")

def log_verbose(str):
    if os.getenv("LOG_LEVEL_VERBOSE") == "True":
        print(f"[{get_readable_time()}] {str}")