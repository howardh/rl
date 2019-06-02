import numpy as np

def optimal_policy(state):
    """
    ?Optimal? policy for frozen lake

    SFFF
    FHFH
    FFFH
    HFFG
    """

    if not isinstance(state,int):
        state = state.tolist().index([1])

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    NONE = 0
    policy = [
            LEFT, UP, LEFT, UP,
            LEFT, NONE, RIGHT, NONE,
            UP, DOWN, LEFT, NONE,
            NONE, RIGHT, DOWN, NONE
    ]
    result = [0,0,0,0]
    result[policy[state]] = 1
    return result

def print_policy(agent, f=lambda x: x):
    dirs = "<v>^"
    holes = [5,7,11,12]
    x = ''
    for i in range(4*4):
        if i%4 == 0:
            x += '\n'
        if i in holes:
            x = x+' '
        else:
            x = x+dirs[np.argmax(agent.learner.get_target_policy(f(i)))]
    print(x)
    return x

def print_values(agent, f=lambda x: x):
    print("Values")
    vals = [agent.learner.get_state_value(f(s)) for s in range(16)]
    vals = np.reshape(vals, (4,4))
    np.set_printoptions(precision=5, suppress=True)
    print(vals)
    return vals

def print_values2(agent, f=lambda x: x):
    print("Values")
    for a,d in zip(range(4),"<v>^"):
        print(d)
        vals = [agent.learner.get_state_action_value(f(s),a) for s in range(16)]
        vals = np.reshape(vals, (4,4))
        np.set_printoptions(precision=5, suppress=True)
        print(vals)
    return vals

def print_traces(agent, f=lambda x: x):
    print("Traces")
    vals = np.reshape(agent.learner.traces.numpy(), (4,4,4))
    np.set_printoptions(precision=5, suppress=True)
    for a,d in zip(range(4),"<v>^"):
        print(d)
        print(vals[:,:,a])
    return vals
