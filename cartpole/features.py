import numpy as np

IDENTITY_NUM_FEATURES = 5

def identity(x):
    return np.append(x,1).reshape([IDENTITY_NUM_FEATURES,1])

def identity2(x):
    state = [x[0], np.tanh(x[1]/10), x[2], np.tanh(x[3]/10)]
    return np.append(x,1).reshape([IDENTITY_NUM_FEATURES,1])

ONE_HOT_NUM_FEATURES = 4*4*4*4

CART_POSITION_MIN = -2.4
CART_POSITION_MAX = 2.4
POLE_ANGLE_MAX = 41.8
POLE_ANGLE_MIN = -41.8

def one_hot(x):
    """Return a feature vector representing the given state and action"""
    cart_pos,cart_vel,pole_ang,pole_vel = x

    # Cart position
    discrete_cart_pos = int((cart_pos - CART_POSITION_MIN)/(CART_POSITION_MAX-CART_POSITION_MIN)*4)

    # Pole angle
    discrete_pol_ang = int((cart_pos - POLE_ANGLE_MIN)/(POLE_ANGLE_MAX-POLE_ANGLE_MIN)*4)

    # Cart velocity
    cart_vel_discretisations = [-1,0,1]
    discrete_cart_vel= 3
    for i,v in enumerate(cart_vel_discretisations):
        if cart_vel < v:
            discrete_cart_vel = i
            break

    # Pole tip velocity
    pole_vel_discretisations = [-1,0,1]
    discrete_pole_vel= 3
    for i,v in enumerate(pole_vel_discretisations):
        if pole_vel < v:
            discrete_pole_vel = i
            break

    # Convert to one-hot encoding
    x = discrete_cart_pos + discrete_cart_vel*4 + discrete_pol_ang*8 + discrete_cart_vel*12
    output = [0] * ONE_HOT_NUM_FEATURES
    output[x] = 1
    return np.array([output]).transpose()

def radial_basis(x):
    # Normalize
    # generate centres
    pass
