import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from learner.rbf_learner import RBFLearner

def optimal_policy(state):
    if state[1] < 0:
        return np.array([1,0,0])
    else:
        return np.array([0,0,1])

def less_optimal_policy(state, p=0.75):
    q = (1-p)/2
    if state[1] < 0:
        return np.array([p,q,q])
    else:
        return np.array([q,q,p])

def get_one_hot_optimal_policy(num_pos, num_vel, prob):
    p = prob
    q = (1-p)/2
    n = 10
    i=([True]*int(num_vel/2)+[False]*(num_vel-int(n/2)))*num_pos
    j=([False]*(num_vel-int(num_vel/2))+[True]*int(num_vel/2))*num_pos
    def policy(state):
        if any(state[i]):
            return np.array([p,q,q])
        elif any(state[j]):
            return np.array([q,q,p])
        else:
            return np.array([q,p,q])
    return policy

def print_policy(agent, f=lambda x: x):
    pass

def print_values(agent, f=lambda x: x):
    pass

def output_rbf_policy(agent, output=None):
    if not isinstance(agent.learner, RBFLearner):
        raise TypeError("Agent does not use an RBF learner")
    w = agent.learner.weights
    c = agent.learner.centres
    s = agent.learner.spread
    x = y = np.arange(0, 1.0, 0.01)
    X, Y = np.meshgrid(x, y)
    def diff(x,y):
        return agent.learner.get_state_action_value(np.array([x,y]),0)-agent.learner.get_state_action_value(np.array([x,y]),2)
    def av(x,y,a):
        return agent.learner.get_state_action_value(np.array([x,y]),a)
    zs0 = np.array([av(x,y,0) for x,y in zip(np.ravel(X), np.ravel(Y))])
    zs1 = np.array([av(x,y,1) for x,y in zip(np.ravel(X), np.ravel(Y))])
    zs2 = np.array([av(x,y,2) for x,y in zip(np.ravel(X), np.ravel(Y))])
    zsd = np.array([diff(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])

    fig = plt.figure()

    ax = fig.add_subplot(221, projection='3d')
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('State-Action Value')

    ax = fig.add_subplot(222, projection='3d')
    Z = zs2.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('State-Action Value')

    ax = fig.add_subplot(223, projection='3d')
    Z = zs1.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('State-Action Value')

    #colours = [(1, 0, 0), (0, 0, 1)]
    #cm = LinearSegmentedColormap.from_list("Boop", colours, N=2)

    ax = fig.add_subplot(224, projection='3d')
    Z = zsd.reshape(X.shape)
    #ax.plot_surface(X, Y, Z, cmap=cm)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('State-Action Value Difference')

    if output is None:
        plt.show()
    else:
        fig.savefig(output)
    fig.clf()
    plt.close()
