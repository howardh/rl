from rl.agent.smdp.dqn import ObservationStack

def test_last_obs():
    stack = ObservationStack()

    stack.append_obs('obs1',reward=None,terminal=False)
    assert stack.get_obs() == 'obs1'
    stack.append_action('action1')
    assert stack.get_obs() == 'obs1'
    stack.append_obs('obs2',reward=None,terminal=False)
    assert stack.get_obs() == 'obs2'
    stack.append_action('action2')
    assert stack.get_obs() == 'obs2'

def test_transition():
    stack = ObservationStack()

    stack.append_obs('obs1',reward=None,terminal=False)
    assert stack.get_transition() is None
    stack.append_action('action1')
    assert stack.get_transition() is None
    stack.append_obs('obs2',reward=0,terminal=False)

    assert stack.get_transition() == ('obs1','action1',0,'obs2',False)
    stack.append_action('action2')
    assert stack.get_transition() == ('obs1','action1',0,'obs2',False)
    stack.append_obs('obs3',reward=0,terminal=False)

    assert stack.get_transition() == ('obs2','action2',0,'obs3',False)
    stack.append_action('action3')
    stack.append_obs('obs4',reward=0,terminal=True)
    assert stack.get_transition() == ('obs3','action3',0,'obs4',True)

def test_transition_reset():
    stack = ObservationStack()

    stack.append_obs('obs1',reward=None,terminal=False)
    stack.append_action('action1')
    stack.append_obs('obs2',reward=0,terminal=False)
    stack.append_action('action2')
    stack.append_obs('obs3',reward=0,terminal=False)

    # Copied from `test_transition()`
    stack.append_obs('obs1',reward=None,terminal=False)
    assert stack.get_transition() is None
    stack.append_action('action1')
    assert stack.get_transition() is None
    stack.append_obs('obs2',reward=0,terminal=False)

    assert stack.get_transition() == ('obs1','action1',0,'obs2',False)
    stack.append_action('action2')
    assert stack.get_transition() == ('obs1','action1',0,'obs2',False)
    stack.append_obs('obs3',reward=0,terminal=False)

    assert stack.get_transition() == ('obs2','action2',0,'obs3',False)
