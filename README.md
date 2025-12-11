# AI-Driven Smart Traffic Signal Digital Twin

A simulation project combining reinforcement learning, AI agents, digital twins, and embedded-style control logic, all without using any physical hardware.
The system simulates a single traffic signal at a pedestrian crossing.
A reinforcement learning agent learns when to switch the signal between GREEN and RED based on virtual car flow.
A digital twin dashboard displays the traffic level, signal status, and the agent’s decisions in real time.

Project showing:
- a traffic intersection simulator (env.py)
- a simple Q-learning agent (agent.py)
- a simulated edge device loop (edge_device.py)
- a Streamlit digital twin dashboard (dashboard.py)


## Quick start

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python train.py          # trains and saves q_table.npy
streamlit run dashboard.py
