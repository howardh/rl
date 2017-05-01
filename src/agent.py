import numpy as np

class Agent(object):
    """
    Attributes
    ----------
    learner
    """

    def set_target_policy(self, policy):
        self.target_policy = policy

    def set_behaviour_policy(self, policy):
        self.behaviour_policy = policy
        
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

    def test_once(self, env):
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
        return reward_sum
        
    def test(self, env, iterations):
        """
        Run multiple episodes on the given environment, following the target policy.

        env
            Environment on which to run the tests
        iterations
            Number of episodes to run
        """
        rewards = []
        for i in range(iterations):
            rewards.append(self.test_once(env))
        return rewards
