# train.py
from env import TrafficEnv
from agent import QAgent
from edge_device import edge_loop

def train(episodes=200, steps_per_episode=20):
    env = TrafficEnv()
    agent = QAgent()
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(steps_per_episode):
            action = edge_loop(state, agent)
            next_state, reward = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} - total reward: {total_reward}")
    print("Training complete.")
    return agent

if __name__ == "__main__":
    trained_agent = train()
    # save Q-table to numpy file for later reuse
    import numpy as np
    np.save("q_table.npy", trained_agent.q)
    print("Saved q_table.npy")
