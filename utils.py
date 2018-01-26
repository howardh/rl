import os
import time
import threading
import re
from tqdm import tqdm 
import scipy.sparse
import scipy.sparse.linalg
import timeit
import csv
import numpy as np
import torch

START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")
_results_dir = None

lock = threading.Lock()

def get_results_directory():
    global _results_dir
    if _results_dir is not None:
        return _results_dir
    host_name = os.uname()[1]
    if host_name == "agent-server-1" or host_name == "agent-server-2":
        return os.path.join("/NOBACKUP/hhuang63/results3",START_TIME)
    raise NotImplementedError("No default path defined for %s" % host_name)

def set_results_directory(d):
    global _results_dir
    _results_dir = d

def find_next_free_file(prefix, suffix, directory):
    global lock
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with lock:
        i = 0
        while True:
            while True:
                path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
                if not os.path.isfile(path):
                    break
                i += 1
            # Create the file to avoid a race condition.
            # Will give an error if the file already exists.
            try:
                f = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(f)
            except FileExistsError as e:
                # Trying to create a file that already exists.
                # Try a new file name
                continue
            break
    return path, i

def load_data(directory, pattern=re.compile(r'^.+\.csv$')):
    import csv
    import numpy as np
    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]

    # A place to store our results
    data = []
    params = dict()

    # Load results from all csv files
    for file_name in tqdm(files, desc="Parsing File Contents"):
        regex_result = pattern.match(file_name)
        if regex_result is None:
            continue # Skip files that don't match the pattern
        p = parse_file_name(file_name)
        for k,v in p.items():
            if k not in params.keys():
                params[k] = set()
            params[k].add(v)
        with open(os.path.join(directory,file_name), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            results = [np.sum(eval(r[1])) for r in reader]
            data += [results]

    # Remove all incomplete series
    max_len = max([len(x) for x in data])
    data = [x for x in data if len(x) == max_len]

    # Compute stuff
    m = np.mean(data, axis=0)
    v = np.var(data, axis=0)
    s = np.std(data, axis=0)

    # Build output
    output = dict()
    output['mean'] = m
    output['std'] = s
    output['var'] = v
    output['params'] = params

    return output

def parse_file_name(file_name):
    pattern = re.compile(r'((?:(?:^|-)(?:[a-zA-Z]+)(?:\.|\d)+)+)-\d+\.csv')
    regex_result = pattern.match(file_name)
    if regex_result is None:
        return None # Should maybe return a descriptive error? Or throw exception?

    # Parse param string
    param_string = regex_result.group(1)
    tokens = param_string.split('-')
    token_pattern = re.compile(r'([a-zA_Z]+)((?:0|[1-9]\d*)(?:\.\d+)?)')

    results = dict()
    for t in tokens:
        regex_results = token_pattern.match(t)
        if regex_results is None:
            return None # Invalid file name
        param_name = regex_results.group(1)
        param_value = regex_results.group(2)
        results[param_name] = param_value

    return results

def collect_file_params(file_names):
    results = dict()
    for file_name in file_names:
        file_params = parse_file_name(file_name)
        if file_params is None:
            continue
        for k,v in file_params.items():
            if k not in results:
                results[k] = set()
            results[k].add(v)
    return results

def parse_file(file_name, threshold=None):
    """
    Process the contents of a file with data from a trial, and return the sum
    of rewards.
    """
    threshold_met = None
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        results = [(int(r[0]), np.mean(eval(r[1]))) for r in reader]
        # Check if we have data to return
        if len(results) == 0:
            return None
        # Compute time to learn
        if threshold is not None:
            for r in results:
                if threshold_met is None and r[1] >= threshold:
                    threshold_met = r[0]
            if threshold_met is None:
                threshold_met = results[-1][0]
        # Compute mean reward over time
        means = [r[1] for r in results]

        return np.mean(means), threshold_met

def svd_inv(a):
    #u,s,vt = scipy.sparse.linalg.svds(a,k=int(min(a.shape)/2))
    u,s,vt = scipy.sparse.linalg.svds(a,k=2000)
    sinv = np.matrix(np.diag(1/s))
    u = np.matrix(u)
    vt = np.matrix(vt)

    ainv = vt.H*sinv*u.H

    return ainv

def torch_svd_inv(a):
    u, s, v = torch.svd(a)
    #x = torch.mm(torch.mm(u, torch.diag(s)), v.t())
    sinv = torch.FloatTensor([1/x if x != 0 else 0 for x in s])
    if s.is_cuda:
        sinv = sinv.cuda()
    y = torch.mm(torch.mm(v, torch.diag(sinv)), u.t())
    return y

_solve_start_time = timeit.default_timer()
def solve(a,b):
    global _solve_start_time
    print("Solve time: %s" % (timeit.default_timer()-_solve_start_time))
    _solve_start_time = timeit.default_timer()
    result = scipy.sparse.linalg.lsqr(a, np.array(b.todense()).reshape((b.shape[0],)))
    return result[0].reshape((b.shape[0],1))

def solve_approx(a,b):
    print("Starting solve")
    solve_start_time = timeit.default_timer()
    result = svd_inv(a)*b
    print("Solve time: %s" % (timeit.default_timer()-solve_start_time))
    return result.reshape((b.shape[0],1))

def cc2(fn, params, proc=10, keyworded=False):
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    plist = list(params)
    futures = []
    with ProcessPoolExecutor(max_workers=proc) as executor:
        for p in tqdm(plist):
            if keyworded:
                futures.append(executor.submit(fn, **p))
            else:
                futures.append(executor.submit(fn, *p))
            if len(futures) >= proc:
                wait(futures,return_when=FIRST_COMPLETED)
            futures = [f for f in futures if not f.done()]
        wait(futures)

def cc(fn, params, proc=10, keyworded=False):
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import as_completed
    if proc == 1:
        if keyworded:
            futures = [fn(**p) for p in tqdm(list(params), desc="Executing jobs")]
        else:
            futures = [fn(*p) for p in tqdm(list(params), desc="Executing jobs")]
        return
    try:
        with ProcessPoolExecutor(max_workers=proc) as executor:
            if keyworded:
                futures = [executor.submit(fn, **p) for p in tqdm(params, desc="Adding jobs")]
            else:
                futures = [executor.submit(fn, *p) for p in tqdm(params, desc="Adding jobs")]
            pbar = tqdm(total=len(futures), desc="Job completion")
            while len(futures) > 0:
                count = [f.done() for f in futures].count(True)
                pbar.update(count)
                futures = [f for f in futures if not f.done()]
                #wait(futures,return_when=FIRST_COMPLETED)
                time.sleep(1)
        #for f in tqdm(as_completed(futures), desc="Job completion",
        #        total=len(futures), unit='it', unit_scale=True, leave=True):
        #    pass
    except Exception as e:
        print("Something broke")

def cc3(fn, params, proc=10, keyworded=False):
    futures = []
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=proc) as executor:
            for i in tqdm(params, desc="Adding jobs"):
                if keyworded:
                    future = [executor.submit(fn, **i)]
                else:
                    future = [executor.submit(fn, *i)]
                futures += future
            pbar = tqdm(total=len(futures), desc="Job completion")
            while len(futures) > 0:
                count = [f.done() for f in futures].count(True)
                pbar.update(count)
                futures = [f for f in futures if not f.done()]
                time.sleep(1)
    except Exception as e:
        print("Something broke")

