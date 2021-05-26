import abc
from typing import Optional, Union

import numpy as np

class Agent(abc.ABC):
    """ Base class for RL agents. """
    @abc.abstractclassmethod
    def observe(self,
            obs : np.ndarray,
            reward : Optional[float] = None,
            terminal : bool = False,
            testing: bool = False,
            time : int = 1,
            discount : Optional[float] = None,
            env_key : Union[int,str] = 0):
        """ The agent observes the state of the environment and keeps the relevant information.

        Args:
            obs (np.ndarray): Observation obtained from the environment
            reward (Optional[float]): Reward obtained from the environment. Set to `None` if it is the first observation of the episode.
            terminal (bool): Set to `True` if the given observation is for a terminal state.
            testing (bool): If `True`, then the observation is received in testing mode. This means that this data point is not used for training.
            time (int): The number of timesteps between the last action taken using `act` and this observation. This is only used with semi-Markov decision processes. In all other cases, defaults to `1`.
            discount (Optional[float]): Discount factor.
            env_key (Union[int,str]): A unique key identifying the environment from which the observation was obtained. This is used when the agent is concurrently acting on multiple environments. For example, one environment for training, and another for testing, or on-policy deep RL which require multiple independent environments to train on.
        """
        pass

    @abc.abstractclassmethod
    def act(self, testing : bool = False, env_key : Union[int,str] = 0) -> Union[int,np.ndarray]:
        """[TODO:summary]

        [TODO:description]

        Args:
            self: [TODO:description]
            testing: [TODO:description]
            env_key: [TODO:description]

        Returns:
            Union[int,np.ndarray]: [TODO:description]
        """
        pass
