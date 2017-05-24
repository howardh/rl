import numpy as np
import re
import itertools
import gym

class Agent(object):
    """
    Attributes
    ----------
    learner
    """

    def parse_policy(self, policy):
        if callable(policy):
            return policy
        pattern = re.compile(r"^((0|[1-9]\d*)?(\.\d+)?(?<=\d))-epsilon$")
        match_obj = pattern.match(policy)
        if match_obj is not None:
            eps = float(match_obj.group(1))
            print("e-greedy with epsilon %f" % eps)
            return self.learner.get_epsilon_greedy(eps)
        pattern = re.compile(r"^((0|[1-9]\d*)?(\.\d+)?(?<=\d))-softmax$")
        match_obj = pattern.match(policy)
        if match_obj is not None:
            temp = float(match_obj.group(1))
            print("Softmax with temperature %f" % temp)
            return self.learner.get_softmax(temp)
        raise ValueError("Invalid policy provided.")

    def set_target_policy(self, policy):
        self.learner.set_target_policy(self.parse_policy(policy))

    def set_behaviour_policy(self, policy):
        self.learner.set_behaviour_policy(self.parse_policy(policy))
        
    def get_weight_change(self):
        return self.learner.get_weight_change()
        
    def reset_weight_change(self):
        return self.learner.reset_weight_change()

    def act(self, observation, testing=False):
        """Return a random action according to the current behaviour policy"""
        if testing:
            dist = self.learner.get_target_policy(observation)
        else:
            dist = self.learner.get_behaviour_policy(observation)
        return np.random.choice(len(dist), 1, p=dist)[0]

    def test_once(self, env, render=False):
        """
        Run an episode on the environment by following the target behaviour policy (Probably using a greedy deterministic policy).

        env
            Environment on which to run the test
        """
        reward_sum = 0
        obs = env.reset()
        done = False
        while not done:
            action = self.act(obs, testing=True)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if render:
                env.render()
        return reward_sum
        
    def test(self, env, iterations, render=False, record=True, processors=1):
        """
        Run multiple episodes on the given environment, following the target policy.

        env
            Environment on which to run the tests
        iterations
            Number of episodes to run
        """
        if processors==1:
            rewards = []
            for i in range(iterations):
                rewards.append(self.test_once(env, render))
            return rewards
        else:
            from pathos.multiprocessing import ProcessPool
            import logging
            pool = ProcessPool(processes=processors)
            env_name = env.spec.id
            def test(proc_id):
                output = self.test_once(gym.make(env_name))
                return output
            rewards = pool.map(test, range(iterations))
            return rewards
