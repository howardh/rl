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
import dill
import pandas
import operator

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
    if host_name == "garden-path" or host_name == "ppl-3":
        return os.path.join("/home/ml/hhuang63/results",START_TIME)
    if host_name.find('gra') == 0:
        return os.path.join("/home/hhuang63/scratch/results",START_TIME)
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

def collect_file_params_pkl(file_names):
    results = dict()
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            try:
                x = dill.load(f)
            except Exception as e:
                tqdm.write("Skipping %s" % file_name)
                continue
        file_params = x[0]
        if file_params is None:
            continue
        for k,v in file_params.items():
            if k not in results:
                results[k] = set()
            results[k].add(v)
    return results

def parse_file(file_name, threshold=None, delete_invalid=False):
    """
    Process the contents of a file with data from a trial, and return
    the rewards as time series data, the sum of rewards, and when the threshold
    was met.
    """
    threshold_met = None
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        series = [(int(r[0]), np.mean(eval(r[1]))) for r in reader]
        # Check if we have data to return
        if len(series) == 0:
            if delete_invalid:
                os.remove(file_name)
            return None
        # Compute time to learn
        if threshold is not None:
            for r in series:
                if threshold_met is None and r[1] >= threshold:
                    threshold_met = r[0]
            if threshold_met is None:
                threshold_met = series[-1][0]
        # Compute mean reward over time
        means = [r[1] for r in series]
        times = [r[0] for r in series]

        return (times,means), np.mean(means), threshold_met

def parse_results(directory, learned_threshold=None):
    """
    Given a directory containing the results of experiments as CSV files,
    compute statistics on the data, and return it as a Pandas dataframe.

    CSV format:
        Two columns:
        * Time
            An integer value, which could represent episode count, step count,
            or clock time
        * Rewards
            A list of floating point values, where each value represents the
            total reward obtained from each test run. This list may be of any
            length, and must be wrapped in double quotes.
    e.g.
        0,"[0,0,0]"
        10,"[1,0,0,0,0]"
        20,"[0,1,0,1,0]"
        30,"[0,1,1,0,1,0,1]"

    Pandas Dataframe format:
        Columns:
        * MRS - Mean Reward Sum
            Given the graph of the mean testing reward over time, the MRS is
            the average of these testing rewards over time.
        * TTLS - Time to Learn Sum
            The first time step at which the testing reward matches/surpasses the given
            threshold. Units may be in episodes, steps, or clock time,
            depending on the units used in the CSV data.
        * Count
            Number of trials that were run with the given parameters.
        Indices:
            Obtained from the parameters in the file names.
    """
    # Check if pickle files are there
    results_file_name = os.path.join(directory, "results.pkl") 
    sorted_results_file_name = os.path.join(directory, "sorted_results.pkl") 
    if os.path.isfile(results_file_name) and os.path.isfile(sorted_results_file_name):
        print("Data already computed. Loading pickle files.")
        with open(results_file_name, 'rb') as f:
            data = dill.load(f)
        with open(sorted_results_file_name, 'rb') as f:
            sorted_data = dill.load(f)
        return data, sorted_data

    # Parse set of parameters
    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]
    params = collect_file_params(files)
    param_names = list(params.keys())
    param_vals = [params[k] for k in param_names]

    # A place to store our results
    indices = pandas.MultiIndex.from_product(param_vals, names=param_names)
    data = pandas.DataFrame(0, index=indices, columns=["MRS", "TTLS", "Count"])
    data['MRS'] = data.MRS.astype(float)
    data.sortlevel(inplace=True) # Why am I doing this again? Can I replace it with sort_index?
    #data.sort_index(inplace=True)

    # Load results from all csv files
    for file_name in tqdm(files, desc="Parsing File Contents"):
        file_params = parse_file_name(file_name)
        if file_params is None:
            tqdm.write("Invalid file (%s). Skipping." % file_name)
            continue
        output = parse_file(os.path.join(directory,file_name),
                learned_threshold)
        if output is None:
            tqdm.write("Skipping empty file: %s" % file_name)
            continue
        _,mr,ttl = output
        key = tuple([file_params[k] for k in param_names])
        data.loc[key,'MRS'] += mr
        if ttl is None:
            data.loc[key,'TTLS'] = None
        else:
            data.loc[key,'TTLS'] += ttl
        data.loc[key,'Count'] += 1

    # Return results
    return data

def parse_graphing_results(directory):
    # TODO: Remove incomplete series
    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]
    params = collect_file_params(files)
    if 's' in set(params.keys()):
        sigmas = params['s']
        data = dict()
        times = dict()
        for s in sigmas:
            data[s] = []
            times[s] = None
    else:
        sigmas = None
        data = []
        times = None
    for file_name in tqdm(files, desc="Parsing File Contents"):
        try:
            full_path = os.path.join(directory,file_name)
            series,_,_ = parse_file(full_path)
            if series is not None:
                if sigmas is not None:
                    s = parse_file_name(file_name)['s']
                    data[s].append(series[1])
                    if times[s] is None or len(times[s]) < len(series[0]):
                        times[s] = series[0]
                else:
                    data.append(series[1])
                    if times is None or len(times) < len(series[0]):
                        times = series[0]
        except SyntaxError as e:
            tqdm.write("Broken file: %s" % file_name)
        except Exception as e:
            tqdm.write("Broken file: %s" % file_name)

    if sigmas is None:
        max_len = 0
        for row in data:
            max_len = max(max_len, len(row))
        data = [d for d in data if len(d) == max_len]
    else:
        results = dict()
        for s in data.keys():
            max_len = 0
            for row in data[s]:
                max_len = max(max_len, len(row))
            data[s] = [d for d in data[s] if len(d) == max_len]

    if sigmas is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return times, mean, std
    else:
        results = dict()
        for s in data.keys():
            mean = np.mean(data[s], axis=0)
            std = np.std(data[s], axis=0)
            t = times[s]
            results[s] = (t,mean,std)
        return results

def parse_graphing_results_pkl(directory):
    # Load data
    file_names = [os.path.join(directory,f) for f in tqdm(os.listdir(directory)) if os.path.isfile(os.path.join(directory,f))]
    params = collect_file_params_pkl(file_names)
    print(params)
    sigmas = params['sigma']
    data = dict()
    times = dict()
    for s in sigmas:
        data[s] = []
        times[s] = None
    for file_name in tqdm(file_names, desc="Parsing File Contents"):
        with open(os.path.join(directory,file_name), 'rb') as f:
            try:
                x = dill.load(f)
            except Exception:
                tqdm.write(file_name)
                continue
        diverged = None in x[1]
        if diverged:
            tqdm.write("Diverged %s" % file_name)
            continue
        else:
            s = x[0]['sigma']
            data[s].append(x[1])
            if times[s] is None or len(times[s]) < len(x[1]):
                times[s] = np.arange(len(x[1]))*x[0]['epoch']
    for s in data.keys():
        max_len = 0
        for row in data[s]:
            max_len = max(max_len, len(row))
        data[s] = [d for d in data[s] if len(d) == max_len]
    results = dict()
    for s in data.keys():
        x = np.mean(data[s], axis=2)
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        t = times[s]
        results[s] = (t,mean,std)
    return results

def sort_data(data):
    """
    Return the data sorted by performance using two measures:
    - Mean reward
    - Time to learn
    The best results (i.e. highest mean reward, or lowest time to learn) is
    found at index 0, and the worst at index -1.
    """
    print("Sorting by MR")
    sorted_by_mr = [(i,data.loc[i,'MRS']/data.loc[i,'Count']) for i in data.index]
    sorted_by_mr = sorted(sorted_by_mr, key=operator.itemgetter(1))
    sorted_by_mr.reverse()
    print("Sorting by TTL")
    sorted_by_ttl = [(i,data.loc[i,'TTLS']/data.loc[i,'Count']) for i in data.index]
    sorted_by_ttl = sorted(sorted_by_ttl, key=operator.itemgetter(1))
    return sorted_by_mr, sorted_by_ttl

def combine_params_with_names(data, params):
    """
    Return a dictionary of keyworded parameters.

    data: Pandas dataframe
    params: An iterable of unlabelled parameters in the same order as the
    dataframe indices
    """
    names = data.index.names
    if len(names) != len(params):
        raise Exception("Dataframe indices (%s) and length of parameters (%s) do not match." % (names, params))
    return dict(zip(names, params))

def get_best_params_by_sigma(directory, learned_threshold):
    data = parse_results(directory, learned_threshold=learned_threshold)
    sigmas = set(data.index.get_level_values('s'))
    results = dict()
    for s in sigmas:
        df = data.xs(s,level='s')
        mr, ttl = sort_data(df)
        params = [eval(x) for x in ttl[0][0]]
        params = combine_params_with_names(df,params)
        params['s'] = eval(s)
        results[s] = params
    return results

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
    sinv = torch.FloatTensor([1/x if abs(x) > 0.000000001 else 0 for x in s])
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

