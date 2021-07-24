import gym

from rl.debug_tools.frozenlake import optimal_policy

def test_optimal_policy():
    env = gym.make('FrozenLake-v0')

    def test():
        total = 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            obs,reward,done,_ = env.step(optimal_policy[obs])
            total += reward
        return total

    mean_reward = sum([test() for _ in range(100)])/100
    assert mean_reward >= 0.7, 'Failed to achieve adequate performance. There is a small chance of failure, so if `mean_reward` is high enough, this is probably not a problem.'
