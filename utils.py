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
import traceback

START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")
_results_dir = None
_skip_new_files = False

lock = threading.Lock()

# File IO

def get_results_directory():
    global _results_dir
    if _results_dir is not None:
        return _results_dir
    host_name = os.uname()[1]
    if host_name == "agent-server-1" or host_name == "agent-server-2":
        return os.path.join("/NOBACKUP/hhuang63/results3",START_TIME)
    if host_name == "garden-path" or host_name == "ppl-3":
        return os.path.join("/home/ml/hhuang63/results",START_TIME)
    if host_name.find('gra') == 0 or host_name.find('cdr') == 0:
        return os.path.join("/home/hhuang63/scratch/results",START_TIME)
    if host_name.find('howard-pc') == 0:
        return os.path.join("/home/howard/tmp/results",START_TIME)
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

def skip_new_files(skip=None):
    global _skip_new_files
    if skip is not None:
        _skip_new_files = skip
    return _skip_new_files

# Data processing

def collect_file_params(file_names):
    results = dict()
    for file_name in tqdm(file_names, desc='Collecting file parameters'):
        try:
            with open(file_name, 'rb') as f:
                file_params,_,_ = dill.load(f)
        except Exception as e:
            tqdm.write("Skipping %s" % file_name)
            continue
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
    if file_name.endswith('.csv'):
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

    if file_name.endswith('.pkl'):
        with open(os.path.join(directory,file_name), 'rb') as f:                                                 
            try:
                x = dill.load(f)                                                                                 
            except Exception as e:
                tqdm.write("Skipping %s" % file_name)
                return None
        series = x[1]                                                                                            
        params = x[0]
        epoch = 50 # default
        if 'epoch' in params:                                                                                    
            epoch = params['epoch']                                                                              
        threshold_met = None
        if threshold is not None:
            for r in series:
                if threshold_met is None and r[1] >= threshold:                                                  
                    threshold_met = r[0]                                                                         
            if threshold_met is None:                                                                            
                threshold_met = series[-1][0]
        means = np.mean(series, axis=1)
        times = [epoch*i for i in range(len(means))]
        return (times,series), means, threshold_met

    raise Exception('Invalid file name. Expected a csv or pkl file.')

def parse_results(directory, learned_threshold=None,
        dataframe_filename = "dataframe.pkl",
        parsedfiles_filename = "parsedfiles.pkl"):
    dataframe_fullpath = os.path.join(directory, dataframe_filename)
    parsedfiles_fullpath = os.path.join(directory, parsedfiles_filename)

    # Get a list of all files that have already been parsed
    if os.path.isfile(parsedfiles_fullpath):
        with open(parsedfiles_fullpath, 'rb') as f:
            parsed_files = dill.load(f)
    else:
        parsed_files = [dataframe_fullpath, parsedfiles_fullpath]

    # Get a list of all files in existence, and remove those that have already
    # been parsed
    files = []
    if is_first_task() and not skip_new_files():
        for d,_,file_names in tqdm(os.walk(directory)):
            files += [os.path.join(d,f) for f in file_names if os.path.isfile(os.path.join(d,f))]
        if len(files) != len(parsed_files):
            files = [f for f in tqdm(files, desc='Removed parsed files') if f not in parsed_files]
        else:
            files = []
    elif skip_new_files():
        print("Skipping file parsing. New files not being parsed.")
    else:
        print("Skipping file parsing. Task ID does not match first task.")

    # A place to store our results
    if os.path.isfile(dataframe_fullpath):
        # A dataframe already exists, so load that instead of making a new one
        print("File exists. Loading...")
        data = pandas.read_pickle(dataframe_fullpath)
        keys = data.index.names
        all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])
    else:
        # Create new dataframe
        # Parse set of parameters
        if len(files) == 0:
            raise Exception('No files found. Are you sure you ran the experiment?')
        all_params = collect_file_params(files)
        del all_params['directory']
        kv_pairs = list(all_params.items())
        vals = [v for k,v in kv_pairs]
        keys = [k for k,v in kv_pairs]

        indices = pandas.MultiIndex.from_product(vals, names=keys)

        data = pandas.DataFrame(0, index=indices, columns=["MRS", "TTLS",
            "MaxS", "Count"])
        data.sort_index(inplace=True)

    # Load results from all pickle files
    types = {'sigma': float,
            'lam': float,
            'eps_b': float,
            'eps_t': float,
            'alpha': float,
            'trace_factor': float,
            'behaviour_eps': float,
            'target_eps': float,
            'learning_rate': float,
            'decay': float}
    def cast_params(param_dict):
        for k in param_dict.keys():
            if k in types:
                param_dict[k] = types[k](param_dict[k])
        return param_dict
    try:
        print("%d files to process." % len(files))
        for file_name in tqdm(files, desc="Parsing File Contents"):
            try:
                with open(os.path.join(directory,file_name), 'rb') as f:
                    params, series, time_to_learn = dill.load(f)
            except Exception as e:
                tqdm.write("Skipping %s" % file_name)
                parsed_files.append(file_name)
                continue
            params = cast_params(params)
            param_vals = tuple([params[k] for k in keys])
            series = np.mean(series, axis=1)

            if time_to_learn is None:
                data.loc[param_vals,'TTLS'] = float('inf')
            else:
                data.loc[param_vals,'TTLS'] += time_to_learn
            data.loc[param_vals, 'MRS'] += np.mean(series)
            data.loc[param_vals, 'MaxS'] += series[-1]
            data.loc[param_vals, 'Count'] += 1
            parsed_files.append(file_name)
    except KeyboardInterrupt as e:
        pass
    finally:
        if len(files) > 0:
            print("Saving file to %s" % dataframe_fullpath)
            data.to_pickle(dataframe_fullpath)
            print("Done")
            print("Saving file to %s" % parsedfiles_fullpath)
            with open(parsedfiles_fullpath, 'wb') as f:
                dill.dump(parsed_files, f)
            print("Done")

    return data

def get_all_series(directory,
        series_filename="series.pkl",
        parsedfiles_filename="sparsedfiles.pkl"):
    series_fullpath = os.path.join(directory, series_filename)
    parsedfiles_fullpath = os.path.join(directory, parsedfiles_filename) 
    # Get a list of all files that have already been parsed
    if os.path.isfile(parsedfiles_fullpath):
        with open(parsedfiles_fullpath, 'rb') as f:
            parsed_files = dill.load(f)
    else:
        parsed_files = set([series_fullpath, parsedfiles_fullpath])

    # Get a list of all files in existence, and remove those that have already
    # been parsed
    files = []
    if is_first_task() and not skip_new_files():
        for d,_,file_names in tqdm(os.walk(directory)):
            files += [os.path.join(d,f) for f in file_names if os.path.isfile(os.path.join(d,f))]
        if len(files) != len(parsed_files):
            files = [f for f in tqdm(files, desc='Removed parsed files') if f not in parsed_files]
        else:
            files = []
    elif skip_new_files():
        print("Skipping file parsing. New files not being parsed.")
    else:
        print("Skipping file parsing. Task ID does not match first task.")

    # A place to store our results
    if os.path.isfile(series_fullpath):
        # A dataframe already exists, so load that instead of making a new one
        print("File exists. Loading %s..." % series_fullpath)
        while True:
            try:
                data = pandas.read_pickle(series_fullpath)
                break
            except:
                print('Failed to load file: %s. Retrying in 30s.' % series_fullpath)
                time.sleep(30)
        keys = data.index.names
        all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])
    else:
        # Create new Series
        # Parse set of parameters
        all_params = collect_file_params(files)
        del all_params['directory']
        kv_pairs = list(all_params.items())
        vals = [v for k,v in kv_pairs]
        keys = [k for k,v in kv_pairs]

        indices = pandas.MultiIndex.from_product(vals, names=keys)

        data = pandas.Series(data=None, index=indices, dtype=object)
        data.sort_index(inplace=True)
        #for i in tqdm(data.index, desc="Setting default value"):
        #    data.loc[i] = []

    # Load results from all pickle files
    types = {'sigma': float,
            'lam': float,
            'eps_b': float,
            'eps_t': float,
            'alpha': float,
            'trace_factor': float,
            'behaviour_eps': float,
            'target_eps': float,
            'learning_rate': float,
            'decay': float}
    def cast_params(param_dict):
        for k in param_dict.keys():
            if k in types:
                param_dict[k] = types[k](param_dict[k])
        return param_dict
    def params_equal(params1, params2):
        for k,v in params1.items():
            if k in params2:
                if params1[k] != params2[k]:
                    return False
            else:
                return False
        return True
    try:
        count = 0
        for file_name in tqdm(files, desc="Parsing File Contents"):
            with open(os.path.join(directory,file_name), 'rb') as f:
                try:
                    file_params, series, time_to_learn = dill.load(f)
                    if type(file_params) is not dict:
                        continue
                except Exception as e:
                    tqdm.write("Skipping %s" % file_name)
                    parsed_files.add(os.path.join(directory,file_name))
                    continue
            file_params = cast_params(file_params)
            #file_params['eps_t'] = file_params['eps_b']
            #with open(os.path.join(directory,file_name), 'wb') as f:
            #    #tqdm.write('Rewriting %s.' % file_name)
            #    dill.dump((file_params, series, time_to_learn), f)
            #    #tqdm.write('Done')
            #    continue
            #if len(series) < file_params['max_iters']/file_params['epoch']+1:
            #    while len(series) < file_params['max_iters']/file_params['epoch']+1:
            #        series.append([-200]*file_params['test_iters'])
            #    with open(os.path.join(directory,file_name), 'wb') as f:
            #        tqdm.write('Rewriting %s.' % file_name)
            #        dill.dump((file_params, series, time_to_learn), f)
            #        tqdm.write('Done')
            index = tuple([file_params[k] for k in data.index.names])
            series = np.mean(series, axis=1).tolist()
            if type(data.loc[index]) is not list:
                data.loc[index] = []
            data.loc[index].append(series)
            count += 1
            parsed_files.add(os.path.join(directory,file_name))
    except KeyboardInterrupt as e:
        print("Keyboard Interrupt")
    except Exception as e:
        print("Exception occured")
        traceback.print_exc()
        print(e)
        print(file_params)
        print(index)
        print(series)
        raise e
    finally:
        print('%d files parsed' % count)

        if len(files) > 0:
            with open(series_fullpath, 'wb') as f:
                dill.dump(data, f)
                print("Saved ", series_fullpath)
            with open(parsedfiles_fullpath, 'wb') as f:
                dill.dump(parsed_files, f)
                print("Saved ", parsedfiles_fullpath)

        return data

def get_series_with_params(directory, params):
    data = get_all_series(directory)
    keys = [k for k in params.keys()]
    vals = [params[k] for k in keys]
    if len(params) == 0:
        dataxs = data
    else:
        dataxs = data.xs(vals, level=keys)
    all_series = []
    for i in dataxs.index:
        all_series += dataxs.loc[i]
    return all_series

# Math

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

def solve(a,b):
    result = scipy.sparse.linalg.lsqr(a, np.array(b.todense()).reshape((b.shape[0],)))
    return result[0].reshape((b.shape[0],1))

def solve_approx(a,b):
    print("Starting solve")
    solve_start_time = timeit.default_timer()
    result = svd_inv(a)*b
    print("Solve time: %s" % (timeit.default_timer()-solve_start_time))
    return result.reshape((b.shape[0],1))

# Multiprocessing

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

# Compute Canada

def split_params(params):
    """ Split params by array task ID
    See https://slurm.schedmd.com/job_array.html for details
    """
    try:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        task_min = int(os.environ['SLURM_ARRAY_TASK_MIN'])
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
    except KeyError:
        return params
    per_task = int(np.ceil(len(params)/(task_max-task_min+1)))
    start_index = int((task_id-task_min)*per_task)
    end_index = int(np.min([(task_id-task_min+1)*per_task, len(params)]))
    return params[start_index:end_index]

def is_first_task():
    """ Check if the task ID is the first one.
    Some tasks should only be done by one task in an array, or else it could break things due to concurrency problems.
    """
    try:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        task_min = int(os.environ['SLURM_ARRAY_TASK_MIN'])
        return task_id == task_min
    except KeyError:
        # If we're not on compute canada, then everything is fine
        return True
