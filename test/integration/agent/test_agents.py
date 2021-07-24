import pytest
import gym
import gym.spaces
import numpy as np
import torch
import torch.nn
import itertools

from rl.experiments.hrl.model import QFunction, PolicyFunction, PolicyFunctionAugmentatedState
from rl.agent.augmented_obs_stack import AugmentedObservationStack, create_transform_one_hot_action
from rl.agent.hdqn_agent import HDQNAgentWithDelayAC, HDQNAgentWithDelayAC_v2, HDQNAgentWithDelayAC_v3
from rl.agent.dqn_agent import DQNAgent

def create_agent(agent_name, env, seed=None, params={}):
    num_actions = env.action_space.n
    state_size = env.observation_space.low.shape[0]
    if agent_name == 'HDQNAgentWithDelayAC':
        num_options = 3
        default_params = {
                'action_space': env.action_space,
                'observation_space': env.observation_space,
                'controller_learning_rate': 0.01,
                'subpolicy_learning_rate': 0.01,
                'q_net_learning_rate': 0.01,
                'discount_factor': 0.9,
                'polyak_rate': 0.001,
                'behaviour_epsilon': 0.5,
                'replay_buffer_size': 10,
                'controller_net': PolicyFunction(
                    layer_sizes=[],input_size=state_size,output_size=num_options),
                'subpolicy_nets': [PolicyFunction(layer_sizes=[],input_size=state_size,output_size=num_actions) 
                    for _ in range(num_options)],
                'q_net': QFunction(layer_sizes=[],input_size=state_size,output_size=num_actions),
                'seed': seed,
                **params
        }
        return HDQNAgentWithDelayAC(
                **default_params
        )
    elif agent_name == 'HDQNAgentWithDelayAC_v2':
        num_options = 3
        default_params = {
                'action_space': env.action_space,
                'observation_space': env.observation_space,
                'controller_learning_rate': 0.01,
                'subpolicy_learning_rate': 0.01,
                'subpolicy_q_net_learning_rate': 0.01,
                'q_net_learning_rate': 0.01,
                'discount_factor': 0.9,
                'polyak_rate': 0.001,
                'behaviour_epsilon': 0.5,
                'replay_buffer_size': 10,
                'controller_net': PolicyFunction(
                    layer_sizes=[],input_size=state_size,output_size=num_options),
                'subpolicy_nets': [PolicyFunction(layer_sizes=[],input_size=state_size,output_size=num_actions) 
                    for _ in range(num_options)],
                'q_net': QFunction(layer_sizes=[],input_size=state_size,output_size=num_actions),
                'seed': seed,
                **params
        }
        return HDQNAgentWithDelayAC_v2(
                **default_params
        )
    elif agent_name == 'HDQNAgentWithDelayAC_v3':
        num_options = 3
        default_params = {
                'action_space': env.action_space,
                'observation_space': env.observation_space,
                'controller_learning_rate': 0.01,
                'subpolicy_learning_rate': 0.01,
                'subpolicy_q_net_learning_rate': 0.01,
                'q_net_learning_rate': 0.01,
                'discount_factor': 0.9,
                'polyak_rate': 0.001,
                'behaviour_epsilon': 0.5,
                'replay_buffer_size': 10,
                'controller_net': PolicyFunctionAugmentatedState(
                    layer_sizes=[],state_size=state_size,
                    num_actions=num_actions,output_size=num_options),
                'subpolicy_nets': [PolicyFunction(layer_sizes=[],input_size=state_size,output_size=num_actions) 
                    for _ in range(num_options)],
                'q_net': QFunction(layer_sizes=[],input_size=state_size,output_size=num_actions),
                'seed': seed,
                **params
        }
        return HDQNAgentWithDelayAC_v3(
                **default_params
        )
    elif agent_name == 'DQNAgent':
        return DQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                learning_rate=0.01,
                discount_factor=0.9,
                polyak_rate=0.001,
                q_net=QFunction(layer_sizes=[],input_size=state_size,output_size=num_actions),
        )

class DummyEnv(gym.Env):
    """ Create a simple Markov chain environment"""
    def __init__(self, actions, observations, length=3):
        self.length = length
        action_spaces = {
            'discrete': gym.spaces.Discrete(2),
            'box': gym.spaces.Box(high=np.array([1]),low=np.array([0]))
        }
        observation_spaces = {
            'discrete': gym.spaces.Discrete(self.length),
            'box': gym.spaces.Box(high=np.array([1]*self.length),low=np.array([0]*self.length))
        }
        self.action_space = action_spaces[actions]
        self.observation_space = observation_spaces[observations]

        self.state = 0

    def step(self, action):
        if self.state is None:
            raise Exception('Must reset env')
        done = False
        reward = 0
        if type(self.action_space) is gym.spaces.Box:
            action = 0 if np.random.rand() < action else 1
        if action == 0:
            self.state -= 1
            if self.state < 0:
                self.state = 0
        else:
            self.state += 1
            if self.state >= self.length-1:
                done = True
                reward = 1

        obs = self.state
        if type(self.observation_space) is gym.spaces.Box:
            obs = np.zeros([self.length])
            obs[self.state] = 1

        info = {}
        return obs, reward, done, info

    def reset(self):
        self.state = 0
        if type(self.observation_space) is gym.spaces.Box:
            obs = np.zeros([self.length])
            obs[self.state] = 1
            return obs
        else:
            return self.state

    def render(self):
        pass

    def close(self):
        self.state = None

@pytest.mark.parametrize('agent_name', ['DQNAgent','HDQNAgentWithDelayAC'])
def test_agents_train_without_error(agent_name):
    env = DummyEnv(actions='discrete',observations='box')
    agent = create_agent(agent_name,env)

    obs = env.reset()
    agent.observe_change(obs)

    for _ in range(5):
        obs, reward, done, _ = env.step(agent.act())
        agent.observe_change(obs, reward, terminal=done)
        agent.train()
        if done:
            break

def test_HDQNAC_v3():
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(
            high=np.array([1,1,1]),low=np.array([0,0,0]))

    class DummyPolicy(torch.nn.Module):
        def __init__(self,ret_val):
            super().__init__()
            self.fc = torch.nn.Linear(1,1)
            self.call_stack = []
            self.ret_val = ret_val
        def forward(self,*params):
            self.call_stack.append(params)
            return self.ret_val.float()

    agent = HDQNAgentWithDelayAC_v3(
                action_space=action_space,
                observation_space=observation_space,
                discount_factor=1,
                behaviour_epsilon=0,
                replay_buffer_size=10,
                controller_net=DummyPolicy(torch.tensor([[0.5,0.5]])),
                subpolicy_nets=[DummyPolicy(torch.tensor([[0.5,0.5]])),DummyPolicy(torch.tensor([[0.5,0.5]]))],
                q_net=DummyPolicy(torch.tensor([[0,0]]))
            )
    
    for testing in [False,True]:
        agent.observe_change(np.array([0,0,0]),testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs1 == torch.tensor([[0,0,0]]).float()).all()
        assert (mask == torch.tensor([[0,1]]).float()).all()

        a = agent.act(testing=testing)
        agent.observe_change(np.array([0,0,0]),0,False,testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs0 == torch.tensor([[0,0,0]]).float()).all()
        assert action0 == a
        assert (obs1 == torch.tensor([[0,0,0]]).float()).all()
        assert (mask == torch.tensor([[1,1]]).float()).all()

        a = agent.act(testing=testing)
        agent.observe_change(np.array([0,0,1]),0,False,testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs0 == torch.tensor([[0,0,0]]).float()).all()
        assert action0 == a
        assert (obs1 == torch.tensor([[0,0,1]]).float()).all()
        assert (mask == torch.tensor([[1,1]]).float()).all()

        a = agent.act(testing=testing)
        agent.observe_change(np.array([0,1,0]),0,False,testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs0 == torch.tensor([[0,0,1]]).float()).all()
        assert action0 == a
        assert (obs1 == torch.tensor([[0,1,0]]).float()).all()
        assert (mask == torch.tensor([[1,1]]).float()).all()

    s0,a0,s1,a1,r2,s2,t,m1 = agent.replay_buffer.buffer[0]
    assert (s0 == torch.tensor([[0,0,0]]).float()).all()
    assert a0 is not None
    assert (s1 == torch.tensor([[0,0,0]]).float()).all()
    assert (s2 == torch.tensor([[0,0,1]]).float()).all()

    s0,a0,s1,a1,r2,s2,t,m1 = agent.replay_buffer.buffer[1]
    assert (s0 == torch.tensor([[0,0,0]]).float()).all()
    assert a0 is not None
    assert (s1 == torch.tensor([[0,0,1]]).float()).all()
    assert (s2 == torch.tensor([[0,1,0]]).float()).all()

@pytest.mark.parametrize('agent_name,seed', list(itertools.product(['HDQNAgentWithDelayAC','HDQNAgentWithDelayAC_v2','HDQNAgentWithDelayAC_v3'],[0])))
def test_HDQNAC_state_dict_all(agent_name,seed,num_steps=100):
    env = DummyEnv(actions='discrete',observations='box')

    agent1 = create_agent(agent_name,env,seed=seed)
    agent2 = create_agent(agent_name,env)
    agent3 = create_agent(agent_name,env)

    state = agent1.state_dict()
    agent2.load_state_dict(state)

    assert agent1.obs_stack == agent2.obs_stack

    # verify equality
    done = True
    for _ in range(num_steps):
        if done:
            obs = env.reset()
            agent1.observe_change(obs)
            agent2.observe_change(obs)
        a1 = agent1.act()
        a2 = agent2.act()

        assert a1 == a2

        obs, reward, done, _ = env.step(a1)
        agent1.observe_change(obs, reward, terminal=done)
        agent2.observe_change(obs, reward, terminal=done)

        agent1.train()
        agent2.train()

    state = agent1.state_dict()
    agent3.load_state_dict(state)

    for _ in range(num_steps):
        if done:
            obs = env.reset()
            agent1.observe_change(obs)
            agent3.observe_change(obs)
        a1 = agent1.act()
        a2 = agent3.act()

        assert a1 == a2

        obs, reward, done, _ = env.step(a1)
        agent1.observe_change(obs, reward, terminal=done)
        agent3.observe_change(obs, reward, terminal=done)

        #agent1.train()
        #agent3.train()

@pytest.mark.parametrize('agent_name', ['HDQNAgentWithDelayAC','HDQNAgentWithDelayAC_v3'])
def test_HDQNAC_state_dict_same_action_choices(agent_name,seed=0):
    env = DummyEnv(actions='discrete',observations='box')

    agent1 = create_agent(agent_name,env,seed=seed)
    agent2 = create_agent(agent_name,env)

    state = agent1.state_dict()

    obs = env.reset()
    agent1.observe_change(obs)
    a1 = [agent1.act() for _ in range(100)]
    agent2.observe_change(obs)
    a2 = [agent2.act() for _ in range(100)]

    assert a1 != a2

    agent2.load_state_dict(state)
    agent2.observe_change(obs)
    a2 = [agent2.act() for _ in range(100)]

    assert a1 == a2

@pytest.mark.parametrize('agent_name', ['HDQNAgentWithDelayAC','HDQNAgentWithDelayAC_v3'])
def test_HDQNAC_state_dict_np_rng(agent_name,seed=0):
    env = DummyEnv(actions='discrete',observations='box')

    agent1 = create_agent(agent_name,env,seed=seed)
    agent2 = create_agent(agent_name,env)

    state = agent1.state_dict()

    r1 = [agent1.rand.rand() for _ in range(100)]
    r2 = [agent2.rand.rand() for _ in range(100)]

    assert r1 != r2

    agent2.load_state_dict(state)
    r2 = [agent2.rand.rand() for _ in range(100)]

    assert r1 == r2

@pytest.mark.parametrize('agent_name', ['HDQNAgentWithDelayAC','HDQNAgentWithDelayAC_v3'])
def test_HDQNAC_state_dict_torch_rng(agent_name,seed=0):
    env = DummyEnv(actions='discrete',observations='box')

    agent1 = create_agent(agent_name,env,seed=seed)
    agent2 = create_agent(agent_name,env)

    state = agent1.state_dict()

    r1 = [torch.bernoulli(torch.tensor(0.5),generator=agent1.generator) for _ in range(100)]
    r2 = [torch.bernoulli(torch.tensor(0.5),generator=agent2.generator) for _ in range(100)]

    assert r1 != r2

    agent2.load_state_dict(state)
    r2 = [torch.bernoulli(torch.tensor(0.5),generator=agent2.generator) for _ in range(100)]

    assert r1 == r2

def test_HDQNAC_state_dict_dataloader(agent_name='HDQNAgentWithDelayAC',seed=0,num_steps=100):
    env = DummyEnv(actions='discrete',observations='box')

    agent1 = create_agent(agent_name,env,seed=seed)
    agent2 = create_agent(agent_name,env)

    assert (agent1.q_net.seq[0].weight != agent2.q_net.seq[0].weight).all()
    assert (agent1.controller_net.seq[0].weight != agent2.controller_net.seq[0].weight).all()
    #assert agent1.policy_net.state_dict() != agent2.policy_net.state_dict()

    state = agent1.state_dict()
    agent2.load_state_dict(state)

    assert (agent1.q_net.seq[0].weight == agent2.q_net.seq[0].weight).all()
    assert (agent1.controller_net.seq[0].weight == agent2.controller_net.seq[0].weight).all()
    #assert agent1.policy_net.state_dict() == agent2.policy_net.state_dict()
    assert agent1.obs_stack == agent2.obs_stack

    # verify equality
    done = True
    for _ in range(num_steps):
        if done:
            obs = env.reset()
            agent1.observe_change(obs)
            agent2.observe_change(obs)
        a1 = agent1.act()
        a2 = agent2.act()

        obs, reward, done, _ = env.step(a1)
        agent1.observe_change(obs, reward, terminal=done)
        agent2.observe_change(obs, reward, terminal=done)

    for _ in range(10):
        for x1,x2 in zip(agent1.get_dataloader(5),agent2.get_dataloader(5)):
            assert str(x1) == str(x2)

@pytest.mark.parametrize('delay', [0,1,2])
def test_HDQNAC_observation(delay):
    """ Observations used by the agent to make its decisions are correct.
    """
    agent_name = 'HDQNAgentWithDelayAC'
    seed = 0

    env = DummyEnv(actions='discrete',observations='box', length=5)

    agent = create_agent(agent_name,env,seed=seed,params={'delay_steps': delay})
    agent.controller_dropout = 0

    obs_stack = []
    agent_obs_stack = []

    done = True
    for _ in range(10):
        if done:
            obs = env.reset()
            obs_stack.append(obs)
            agent.observe_change(obs)
            done = False
        else:
            obs,r,done,_ = env.step(agent.act())
            obs_stack.append(obs)
            agent.observe_change(obs,r)
        print(agent.get_current_obs())
        agent_obs = agent.get_current_obs()
        agent_obs_stack.append(agent_obs)

    print(obs_stack)

    for o1,(o2,_,m) in zip(obs_stack, agent_obs_stack[delay:]):
        if m[0,0] == 1:
            print('Comparing',o1,o2)
            assert (o1 == o2.flatten().numpy()).all()
        else:
            print('Skipping',o1,o2)

@pytest.mark.skip(reason="Unimportant for now. Fix later.")
def test_HDQNAC_v3_observation():
    """ Observations used by the agent to make its decisions are correct.
    """
    agent_name = 'HDQNAgentWithDelayAC_v3'
    seed = 0
    delay = 0

    env = DummyEnv(actions='discrete',observations='box', length=5)

    agent = create_agent(agent_name,env,seed=seed,params={'delay_steps': delay})
    agent.controller_dropout = 0

    obs_stack = []
    agent_obs_stack = []

    done = True
    for _ in range(10):
        if done:
            obs = env.reset()
            obs_stack.append(obs)
            agent.observe_change(obs)
            done = False
        else:
            obs,r,done,_ = env.step(agent.act())
            obs_stack.append(obs)
            agent.observe_change(obs,r)
        print(agent.get_current_obs())
        agent_obs = agent.get_current_obs()
        agent_obs_stack.append(agent_obs)

    print(obs_stack)

    for o1,(o2,a,_,m) in zip(obs_stack, agent_obs_stack[delay:]):
        if m[0,0] == 1:
            print('Comparing',o1,o2)
            assert (o1 == o2.flatten().numpy()).all()
        else:
            print('Skipping',o1,o2)

    assert False

def test_AugmentedObservationStack():
    stack = AugmentedObservationStack(stack_len=3)

    # t=1
    stack.append_obs(np.array([0,1,2]))
    assert stack.get(0,0).tolist() == [0,1,2]
    stack.append_action(-1)
    assert stack.get(0,0).tolist() == [0,1,2]
    assert stack.get(0,1) is None

    # t=2
    stack.append_obs(np.array([0,1,2])+1)
    assert stack.get(0,0).tolist() == [1,2,3]
    assert stack.get(1,0).tolist() == [0,1,2]
    assert stack.get(0,1).tolist() == [0,1,2,-1]
    stack.append_action(-2)
    assert stack.get(0,0).tolist() == [1,2,3]
    assert stack.get(1,0).tolist() == [0,1,2]
    assert stack.get(0,1).tolist() == [0,1,2,-1]

    # t=3
    stack.append_obs(np.array([0,1,2])+2)
    assert stack.get(0,0).tolist() == [2,3,4]
    assert stack.get(1,0).tolist() == [1,2,3]
    assert stack.get(0,1).tolist() == [1,2,3,-2]
    assert stack.get(2,0).tolist() == [0,1,2]
    assert stack.get(1,1).tolist() == [0,1,2,-1]
    assert stack.get(0,2).tolist() == [0,1,2,-1,-2]
    stack.append_action(-3)
    assert stack.get(0,0).tolist() == [2,3,4]
    assert stack.get(1,0).tolist() == [1,2,3]
    assert stack.get(0,1).tolist() == [1,2,3,-2]
    assert stack.get(2,0).tolist() == [0,1,2]
    assert stack.get(1,1).tolist() == [0,1,2,-1]
    assert stack.get(0,2).tolist() == [0,1,2,-1,-2]

    # t=4
    stack.append_obs(np.array([0,1,2])+3)
    assert stack.get(0,0).tolist() == [3,4,5]
    assert stack.get(1,0).tolist() == [2,3,4]
    assert stack.get(0,1).tolist() == [2,3,4,-3]
    assert stack.get(2,0).tolist() == [1,2,3]
    assert stack.get(1,1).tolist() == [1,2,3,-2]
    assert stack.get(0,2).tolist() == [1,2,3,-2,-3]
    stack.append_action(-4)
    assert stack.get(0,0).tolist() == [3,4,5]
    assert stack.get(1,0).tolist() == [2,3,4]
    assert stack.get(0,1).tolist() == [2,3,4,-3]
    assert stack.get(2,0).tolist() == [1,2,3]
    assert stack.get(1,1).tolist() == [1,2,3,-2]
    assert stack.get(0,2).tolist() == [1,2,3,-2,-3]

def test_AugmentedObservationStack_get_action():
    stack = AugmentedObservationStack(stack_len=2)

    # t=1
    stack.append_obs(np.array([0,0,0]))
    assert stack.get_action(0) is None
    stack.append_action(1)
    assert stack.get_action(0) == 1

    # t=2
    stack.append_obs(np.array([0,0,0]))
    assert stack.get_action(0) is None
    stack.append_action(2)
    assert stack.get_action(0) == 2
    assert stack.get_action(1) == 1
    assert stack.get_action(2) is None

    # t=3
    stack.append_obs(np.array([0,0,0]))
    assert stack.get_action(0) is None
    stack.append_action(3)
    assert stack.get_action(0) == 3
    assert stack.get_action(1) == 2
    assert stack.get_action(2) is None
