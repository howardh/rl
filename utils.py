import os
import time

START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")
_results_dir = None

def get_results_directory():
    global _results_dir
    if _results_dir is not None:
        return _results_dir
    host_name = os.uname()[1]
    if host_name == "agent-server-1" or host_name == "agent-server-2":
        return os.path.join("/NOBACKUP/hhuang63/results3",START_TIME)
    raise NotImplementedError()

def set_results_directory(d):
    global _results_dir
    _results_dir = d

def find_next_free_file(prefix, suffix, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    i = 0
    while True:
        path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
        if not os.path.isfile(path):
            break
        i += 1
    return path

