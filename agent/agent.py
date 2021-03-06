import os
os.environ['GYM_NO_LOGGER_SETUP'] = "1"
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
            #print("e-greedy with epsilon %f" % eps)
            return self.learner.get_epsilon_greedy(eps)
        pattern = re.compile(r"^((0|[1-9]\d*)?(\.\d+)?(?<=\d))-softmax$")
        match_obj = pattern.match(policy)
        if match_obj is not None:
            temp = float(match_obj.group(1))
            #print("Softmax with temperature %f" % temp)
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

    def test_once(self, env, max_steps=np.inf, render=False):
        """
        Run an episode on the environment by following the target behaviour policy (Probably using a greedy deterministic policy).

        env
            Environment on which to run the test
        """
        reward_sum = 0
        obs = env.reset()
        for steps in itertools.count():
            if steps > max_steps:
                break
            action = self.act(obs, testing=True)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if render:
                env.render()
            if done:
                break
        return reward_sum
        
    def test(self, env, iterations, max_steps=np.inf, render=False, record=True, processors=1):
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
                rewards.append(self.test_once(env, render=render, max_steps=max_steps))
            return rewards
        else:
            # TODO: Outdated code. Redo with utils functions.
            raise NotImplementedError('Outdated code removed.')

    def run_episode(self, env):
        obs = env.reset()
        action = self.act(obs)

        obs2 = None
        done = False
        step_count = 0
        reward_sum = 0
        # Run an episode
        while not done:
            step_count += 1

            obs2, reward2, done, _ = env.step(action)
            action2 = self.act(obs2)
            reward_sum += reward2

            self.learner.observe_step(obs, action, reward2, obs2, terminal=done)

            # Next time step
            obs = obs2
            action = action2
        return reward_sum, step_count

    def run_steps(self, env, num_steps):
        obs = self.features(obs)
        action = self.act(obs)

        reward_sum = 0
        # Run steps
        for step_count in range(1,num_steps+1):
            obs2, reward, done, _ = env.step(action)
            obs2 = self.features(obs2)
            action2 = self.act(obs2)
            reward_sum += reward

            self.learner.observe_step(obs, action, reward, obs2, terminal=done)

            # Next time step
            if done:
                obs = env.reset()
                obs = self.features(obs)
                action = self.act(obs)
                obs2 = None
                done = False
            else:
                obs = obs2
                action = action2

        return reward_sum, num_steps
