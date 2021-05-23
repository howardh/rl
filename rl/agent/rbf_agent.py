from rl.agent.agent import Agent
from rl.learner.rbf_learner import RBFLearner
from rl.learner.rbf_learner import RBFTracesLearner

class RBFAgent(Agent):

    def __init__(self, action_space, observation_space, discount_factor, learning_rate,
            initial_value=0, optimizer=None, features=lambda x: x):
        self.learner = RBFLearner(
                action_space=action_space,
                observation_space=observation_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate,
                optimizer=optimizer,
                initial_value=initial_value
        )
        self.features = features

    def run_episode(self, env):
        obs = env.reset()
        obs = self.features(obs)
        action = self.act(obs)

        obs2 = None
        done = False
        step_count = 0
        reward_sum = 0
        # Run an episode
        while not done:
            step_count += 1

            obs2, reward, done, _ = env.step(action)
            obs2 = self.features(obs2)
            action2 = self.act(obs2)
            reward_sum += reward

            self.learner.observe_step(obs, action, reward, obs2, terminal=done)

            # Next time step
            obs = obs2
            action = action2
        return reward_sum, step_count

class RBFTracesAgent(Agent):

    def __init__(self, action_space, observation_space, discount_factor, learning_rate, trace_factor,
            initial_value=0, optimizer=None, features=lambda x: x):
        self.learner = RBFTracesLearner(
                action_space=action_space,
                observation_space=observation_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate,
                optimizer=optimizer,
                initial_value=initial_value,
                trace_factor=trace_factor
        )
        self.features = features

    def run_episode(self, env):
        obs = env.reset()
        obs = self.features(obs)
        action = self.act(obs)

        obs2 = None
        done = False
        step_count = 0
        reward_sum = 0
        # Run an episode
        while not done:
            step_count += 1

            obs2, reward, done, _ = env.step(action)
            obs2 = self.features(obs2)
            action2 = self.act(obs2)
            reward_sum += reward

            self.learner.observe_step(obs, action, reward, obs2, terminal=done)

            # Next time step
            obs = obs2
            action = action2
        return reward_sum, step_count
