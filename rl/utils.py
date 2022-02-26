import os
import time
import threading
import itertools
from typing import Mapping
import random

from tqdm import tqdm 
import numpy as np
import dill
import gym
import gym.envs.atari.environment
import gym.wrappers.frame_stack
import gym.wrappers.time_limit

START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")
_results_dir = None
_skip_new_files = False

lock = threading.Lock()

# File IO

def get_results_root_directory(temp=False):
    host_name = os.uname()[1]
    # Mila (Check available resources with `sinfo`)
    mila_hostnames = ['rtx', 'leto', 'eos', 'bart', 'mila', 'kepler', 'power', 'apollor', 'apollov', 'cn-', 'login-1', 'login-2', 'login-3', 'login-4']
    if host_name.endswith('server.mila.quebec') or any((host_name.startswith(n) for n in mila_hostnames)):
        if temp:
            return "/miniscratch/huanghow"
        else:
            return "/network/projects/h/huanghow"
    # RL Lab
    if host_name == "agent-server-1" or host_name == "agent-server-2":
        return "/home/ml/users/hhuang63/results"
        #return "/NOBACKUP/hhuang63/results"
    if host_name == "garden-path" or host_name == "ppl-3":
        return "/home/ml/hhuang63/results"
    # Compute Canada
    if host_name.find('gra') == 0 or host_name.find('cdr') == 0:
        return "/home/hhuang63/scratch/results"
    # Local
    if host_name.find('howard-pc') == 0:
        return "/home/howard/tmp/results"
    # Travis
    if host_name.startswith('travis-'):
        return './tmp'
    raise NotImplementedError("No default path defined for %s" % host_name)

def get_results_directory():
    global _results_dir
    if _results_dir is not None:
        return _results_dir
    return os.path.join(get_results_root_directory(),START_TIME)

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
            except FileExistsError:
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

# Data Storage

def save_results(results, directory=None, file_path=None,
        file_name_prefix='results'):
    while True:
        try:
            if file_path is None:
                if directory is None:
                    raise Exception('No directory or file path provided. One of the other is needed.')
                file_path, _ = find_next_free_file(file_name_prefix, "pkl", directory)
            with open(file_path, "wb") as f:
                dill.dump(results, f)
            return file_path
        except KeyboardInterrupt:
            print('Writing data to disk. Ignoring keyboard interrupt.')

def get_all_result_paths(directory):
    """ Generator for the contents of all files in the given directory. """
    for d,_,file_names in os.walk(directory):
        for fn in file_names:
            yield os.path.join(d,fn)

def get_all_results(directory):
    """ Generator for the contents of all files in the given directory. """
    for fn in get_all_result_paths(directory):
        with open(fn, 'rb') as f:
            yield dill.load(f)

def modify_all_results(directory):
    """ A tool to modify all results. For each result, return the file's entire content, along with a function that takes the updated value as an argument. """
    for d,_,file_names in tqdm(os.walk(directory)):
        for fn in file_names:
            with open(os.path.join(d,fn), 'rb') as f:
                try:
                    val = dill.load(f)
                except EOFError:
                    val = None
            def save(val):
                if val is None:
                    os.remove(os.path.join(d,fn))
                else:
                    with open(os.path.join(d,fn), 'wb') as f:
                        dill.dump(val,f)
            yield val,save

def get_results(params, directory, match_exactly=False):
    """ Return a generator containing all results whose parameters match the provided parameters
    """
    for p,r in get_all_results(directory):
        if match_exactly:
            cond = len(params) == len(p) and all((k in p and p[k]==v for k,v in params.items()))
        else:
            cond = all((k in p and p[k]==v for k,v in params.items()))
        if cond:
            yield r

def get_results_reduce(params, directory, func, initial):
    total = initial
    for r in get_results(params, directory):
        total = func(r, total)
    return total

def get_all_results_reduce(directory, func, initial=lambda: []):
    results = {}
    for p,r in get_all_results(directory):
        key = recursive_frozenset(p)
        if key in results:
            results[key] = func(r, results[key])
        else:
            results[key] = func(r, initial())
    return results

def get_all_results_map_reduce(directory, map_func, reduce_func, initial=lambda: []):
    results = {}
    for p,r in get_all_results(directory):
        key = map_func(p)
        if key in results:
            results[key] = reduce_func(r, results[key])
        else:
            results[key] = reduce_func(r, initial())
    return results

def sort_parameters(directory, func, initial):
    get_all_results_reduce(directory, func, initial)

def recursive_frozenset(d):
    def to_hashable(x):
        if type(x) is dict:
            return recursive_frozenset(x)
        if type(x) is list:
            return tuple(x)
        return x
    return frozenset([(k,to_hashable(v)) for k,v in d.items()])

# State persistence

def default_state_dict(obj, keys):
    def foo(o):
        if hasattr(o, 'state_dict'):
            return o.state_dict()
        if isinstance(o, Mapping):
            return {k:foo(v) for k,v in o.items()}
        if isinstance(o,gym.Env):
            return get_env_state(o)
        return o
    return {k:foo(getattr(obj,k)) for k in keys}

def default_load_state_dict(obj, state):
    for k,v in state.items():
        if isinstance(obj,Mapping):
            attr = obj[k]
        else:
            attr = getattr(obj,k)

        if hasattr(attr,'load_state_dict'):
            attr.load_state_dict(v)
        elif isinstance(attr,dict):
            default_load_state_dict(attr,v)
        elif isinstance(attr,gym.Env):
            set_env_state(env=attr,state=v)
        else:
            if isinstance(v,dict):
                raise Exception(f'Unable to load a dictionary ({v}) into a non-dictionary object ({attr}).')
            if isinstance(obj,dict):
                obj[k] = v
            else:
                if hasattr(obj,k):
                    setattr(obj,k,v)
                else:
                    raise Exception(f'Object {obj} does not have attribute {k}')

def get_env_state(env):
    from rl.experiments.training._utils import AtariPreprocessing
    import gym.wrappers.frame_stack
    import gym.wrappers.time_limit
    import gym.envs.atari.environment
    if hasattr(env, 'state_dict'):
        return env.state_dict()
    if isinstance(env, gym.wrappers.frame_stack.FrameStack):
        # See https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py
        return {
                'frames': env.frames,
                'env': get_env_state(env.env),
        }
    if isinstance(env, gym.wrappers.time_limit.TimeLimit):
        return {
                '_elapsed_steps': env._elapsed_steps,
                'env': get_env_state(env.env),
        }
    if isinstance(env, AtariPreprocessing):
        return {
                'game_over': env.game_over,
                'lives': env.lives,
                'env': get_env_state(env.env),
        }
    if isinstance(env,gym.envs.atari.environment.AtariEnv):
        # https://github.com/openai/gym/issues/402#issuecomment-260744758
        return env.clone_state(include_rng=True)
    if type(env).__name__ == 'AtariGymEnvPool':
        return None # TODO: How do I save an envpool state? I can't pickle the entire environment.
    if type(env).__name__ == 'AsyncVectorEnv':
        return None # Data is stored in a separate process, so I don't think we can access it.
    try:
        # This needs to be in a try block because mujoco might not be installed on the machine.
        # There's no reason to require mujoco to be installed if it's not being used, so we just ignore this.
        import gym.envs.mujoco
        if isinstance(env,gym.envs.mujoco.MujocoEnv):
            return {
                    'qpos': env.sim.data.qpos,
                    'qvel': env.sim.data.qvel,
            }
    except:
        pass
    env_type = type(env.unwrapped)
    raise NotImplementedError(f'Unable to handle environment of type {env_type}')

def set_env_state(env, state):
    from rl.experiments.training._utils import AtariPreprocessing
    import gym.wrappers.frame_stack
    import gym.wrappers.time_limit
    import gym.envs.atari.environment
    if hasattr(env,'load_state_dict'):
        env.load_state_dict(state)
        return
    if isinstance(env, gym.wrappers.frame_stack.FrameStack):
        # See https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py
        env.frames = state['frames']
        set_env_state(env.env, state['env'])
        return
    if isinstance(env, gym.wrappers.time_limit.TimeLimit):
        env._elapsed_steps = state['_elapsed_steps']
        set_env_state(env.env, state['env'])
        return
    if isinstance(env, AtariPreprocessing):
        env.game_over = state['game_over']
        env.lives = state['lives']
        set_env_state(env.env, state['env'])
        return
    if isinstance(env,gym.envs.atari.environment.AtariEnv):
        # https://github.com/openai/gym/issues/402#issuecomment-260744758
        env.restore_state(state)
        return
    if type(env).__name__ == 'AtariGymEnvPool':
        return # TODO: How do I save an envpool state? I can't pickle the entire environment.
    if type(env).__name__ == 'AsyncVectorEnv':
        return
    try:
        # This needs to be in a try block because mujoco might not be installed on the machine.
        # There's no reason to require mujoco to be installed if it's not being used, so we just ignore this.
        import gym.envs.mujoco
        if isinstance(env,gym.envs.mujoco.MujocoEnv):
            env.set_state(qpos=state['qpos'],qvel=state['qvel'])
            return
    except:
        pass
    env_type = type(env)
    raise NotImplementedError(f'Unable to handle environment of type {env_type}')

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

def cc1(fn, params, proc=10, keyworded=False):
    from concurrent.futures import ProcessPoolExecutor
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
    except Exception:
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

def cc(funcs, proc=1):
    if proc == 1:
        for f in tqdm(list(funcs), desc="Executing jobs"):
            f()
    else:
        from pathos.multiprocessing import ProcessPool
        pp = ProcessPool(nodes=proc)
        results = []
        for f in tqdm(list(funcs), desc="Creating jobs"):
            results.append(pp.apipe(f))
        pbar = tqdm(total=len(results), desc="Job completion")
        while len(results) > 0:
            count = [r.ready() for r in results].count(True)
            pbar.update(count)
            results = [r for r in results if not r.ready()]
            time.sleep(1)

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

# Experiment Execution

def gridsearch(parameters, func, repetitions=1, shuffle=False):
    """ Output a generator containing a function call for each 
    """
    keys = list(parameters.keys())
    values = [parameters[k] for k in keys]

    params = itertools.product(*values)
    params = (zip(keys,p) for p in params)
    params = itertools.repeat(params, repetitions)
    params = itertools.chain(*list(params))
    params = list(params)
    params = split_params(params)
    if shuffle:
        random.shuffle(params)
    params = [dict(p) for p in params]
    return [lambda p=p: func(**p) for p in params]

# Git

def get_git_patch():
    import git
    repo = git.Repo('.')
    untracked_files = repo.untracked_files

    untracked_file_content = {}
    for filename in untracked_files:
        with open(filename,'rb') as f:
            untracked_file_content[filename] = f.read()

    breakpoint()

    return {
        'commit_id': repo.head.commit.hexsha,
        'diff': repo.git.diff(),
        'untracked_files': untracked_file_content
    }

# Debugging

def compute_grad_mean(optim):
    #critic_optimizer.param_groups[0]['params'][0].grad
    total = 0
    for pg in optim.param_groups:
        for p in pg['params']:
            total += p.grad.mean()
    return total
