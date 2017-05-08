import numpy as np

def optimal_policy(state):
    """
    Optimal policy for frozen lake learned using LSTD

    ^^>>>>>>
    ^^^^^>>>
    ^^< >^>>
    ^^^^< >>
    ^^< >v^>
    <  v^< >
    < >^ > >
    <v> vvvv
    """

    if not isinstance(state,int):
        state = state.tolist().index([1])

    policy_str = "^^>>>>>>^^^^^>>>^^< >^>>^^^^< >>^^< >v^><  v^< >< >^ > ><v> vvvv"


    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    NONE = 0
    dir_dict = {'<': LEFT, 'v': DOWN, '>': RIGHT, '^': UP, ' ': NONE}

    policy = [dir_dict[x] for x in policy_str]
    result = [0,0,0,0]
    result[policy[state]] = 1
    return result

def print_policy(agent, f=lambda x: x):
    dirs = "<v>^"
    holes = [19,29,35,41,42,46,49,52,54,59]
    x = ''
    for i in range(8*8):
        if i%8 == 0:
            x += '\n'
        if i in holes:
            x = x+' '
        else:
            x = x+dirs[np.argmax(agent.learner.get_target_policy(f(i)))]
    print(x)

def print_values(agent, f=lambda x: x):
    vals = [agent.learner.get_state_value(f(s)) for s in range(8*8)]
    vals = np.reshape(vals, (8,8))
    np.set_printoptions(precision=5, suppress=True)
    print(vals)
