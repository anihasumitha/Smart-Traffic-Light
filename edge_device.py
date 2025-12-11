# edge_device.py

def edge_loop(state, agent):
    """
    Simulated edge device inference:
    - it receives a state (tuple) and uses the agent to choose an action.
    - returns an integer action (0 or 1).
    """
    return agent.choose_action(state)
