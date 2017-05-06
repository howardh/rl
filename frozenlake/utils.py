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

def print_values(agent, f=lambda x: x):
    vals = [agent.learner.get_state_value(f(s)) for s in range(16)]
    vals = np.reshape(vals, (4,4))
    np.set_printoptions(precision=5, suppress=True)
    print(vals)
