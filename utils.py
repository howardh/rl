import os
import time

START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")

def get_results_directory():
    host_name = os.uname()[1]
    if host_name == "agent-server-1":
        return os.path.join("/NOBACKUP/hhuang63/results3",START_TIME)
    raise NotImplementedError()

def find_next_free_file(prefix, suffix, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    i = 0
    while True:
        path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
        if not os.path.isfile(path):
            break
        i += 1
    return path

