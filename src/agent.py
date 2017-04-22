class Agent(object):

    def act(self, observation):
        """Return a random action according to the current behaviour policy"""
        raise NotImplementedError

    def test_once(self, env):
        """
        Run an episode on the environment by following the target behaviour policy (Probably using a greedy deterministic policy)

        env : ???
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
        rewards = []
        for i in range(iterations):
            rewards.append(self.test_once(env))
        return rewards
