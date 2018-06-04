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

def optimal_rbf_policy(state):
    weights = np.matrix([[-1.1834485530853271,-41.304813385009766,-25.199222564697266,-26.727540969848633,-20.209440231323242,-4.263505935668945,-0.09154077619314194,0.16917160153388977,-1.4641937017440796,-15.969250679016113,-25.300539016723633,-41.7911491394043,-49.41340255737305,-29.14012908935547,-5.564312934875488,1.692258358001709,-42.61417770385742,-30.51908302307129,-47.34035873413086,-105.9865951538086,-76.91807556152344,-54.05254364013672,-24.427690505981445,10.342952728271484,9.282782554626465,-21.64356231689453,-46.58332824707031,-96.14895629882812,-117.00223541259766,-53.98837661743164,-38.92033767700195,21.96170997619629,52.24226760864258,-46.27212142944336,-14.588804244995117,-50.42322540283203,-33.9958381652832,-50.304134368896484,-23.401798248291016,31.416906356811523,1.3725413084030151,-17.536590576171875,-53.04994201660156,-19.18398666381836,-63.32190704345703,-37.389522552490234,9.290604591369629,5.02604866027832,0.26066961884498596,1.054516077041626,-24.229839324951172,-63.90379333496094,-22.981945037841797,14.703646659851074,2.448122024536133,-0.19959352910518646,0.9840611219406128,0.4148904085159302,0.7276606559753418,-17.772714614868164,29.74003791809082,-25.66726303100586,-18.553619384765625,0.28902074694633484],[-1.0436227321624756,-34.624027252197266,-23.44317054748535,-26.889591217041016,-19.93766212463379,-3.710942029953003,0.8793333768844604,0.2676287889480591,-7.264525413513184,-16.50070571899414,-27.067842483520508,-42.632659912109375,-49.31440353393555,-27.63265609741211,-5.377845764160156,1.5648831129074097,-36.42418670654297,-29.42346954345703,-51.419158935546875,-106.09632873535156,-74.6250228881836,-53.983642578125,-21.995901107788086,17.150861740112305,6.847253322601318,-23.01405143737793,-46.69566345214844,-98.60111999511719,-115.14900207519531,-51.697120666503906,-39.842803955078125,31.469999313354492,14.448491096496582,-55.22377014160156,-12.274627685546875,-55.01224136352539,-28.444067001342773,-52.92018127441406,-18.851177215576172,24.10666275024414,0.2715245187282562,-17.773386001586914,-57.53664016723633,-17.150259017944336,-70.76119995117188,-33.27447509765625,8.303849220275879,4.522276401519775,0.8719921112060547,0.31964313983917236,-24.31632423400879,-60.7477912902832,-18.62325096130371,17.595338821411133,1.3234589099884033,0.7585973143577576,0.8760332465171814,0.6220904588699341,0.1214103251695633,-10.112794876098633,28.146289825439453,-30.499448776245117,-11.097813606262207,-0.022227367386221886],[-0.2028425931930542,-29.385892868041992,-24.26200294494629,-26.35126304626465,-20.122676849365234,-3.8199806213378906,4.137567520141602,0.7461850643157959,-5.44918966293335,-16.67724609375,-26.632658004760742,-43.03236389160156,-48.19023513793945,-27.565568923950195,-2.2291524410247803,0.5073729753494263,-35.97770309448242,-27.337671279907227,-57.355079650878906,-103.98107147216797,-73.90950012207031,-52.45332717895508,-22.13858985900879,35.278743743896484,-0.0396508052945137,-24.580175399780273,-48.19310760498047,-102.50992584228516,-114.00931549072266,-48.27218246459961,-39.055763244628906,42.04291915893555,10.345991134643555,-52.0847053527832,-13.456257820129395,-55.938232421875,-26.966676712036133,-55.24554443359375,-12.3472900390625,18.738285064697266,0.22260044515132904,-21.25895881652832,-57.0584602355957,-17.04106330871582,-73.40092468261719,-25.856473922729492,7.37724494934082,12.06048583984375,0.991234540939331,0.08275414258241653,-23.602008819580078,-65.98269653320312,-13.241683959960938,19.252342224121094,1.0836620330810547,0.45319458842277527,0.18265695869922638,0.42976897954940796,-0.32108479738235474,-3.8674633502960205,25.85572052001953,-36.65726089477539,-11.925726890563965,0.7929561734199524]])
    state = np.matrix(state)
    state = state.reshape(64,1)
    vals = weights*state
    a = np.argmax(vals)
    output = np.array([0,0,0])
    output[a] = 1
    return output

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

def output_rbf_policy_2d(agent, output=None):
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
