# dashboard.py
import streamlit as st
import pandas as pd
import time
import numpy as np
from env import TrafficEnv
from agent import QAgent
from edge_device import edge_loop

st.set_page_config(page_title="Smart Traffic Light — Digital Twin", layout="centered")
st.title("Smart Traffic Light — Digital Twin ")

# show controls
episodes = st.sidebar.number_input("Training episodes (quick demo)", min_value=10, max_value=2000, value=200, step=10)
train_button = st.sidebar.button("Train Agent (quick)")
step_button = st.sidebar.button("Step Simulation")
load_button = st.sidebar.button("Load Q-table (q_table.npy)")

# state trackers
if "env" not in st.session_state:
    st.session_state.env = TrafficEnv()
if "agent" not in st.session_state:
    st.session_state.agent = QAgent()
if "history" not in st.session_state:
    st.session_state.history = []

def run_train():
    agent = QAgent()
    env = TrafficEnv()
    for ep in range(episodes):
        state = env.reset()
        for _ in range(20):
            action = edge_loop(state, agent)
            next_state, reward = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
    st.session_state.agent = agent
    st.success("Training done (in-page)")

if train_button:
    run_train()

if load_button:
    try:
        q = np.load("q_table.npy")
        a = QAgent()
        a.q = q
        st.session_state.agent = a
        st.success("Loaded q_table.npy")
    except Exception as e:
        st.error(f"Could not load q_table.npy: {e}")

# step simulation once
if step_button:
    env = st.session_state.env
    agent = st.session_state.agent
    state = env._get_state()
    action = edge_loop(state, agent)
    next_state, reward = env.step(action)
    st.session_state.history.append({"state": state, "action": action, "reward": reward})
    st.session_state.env = env

# display latest state
if st.session_state.history:
    last = st.session_state.history[-1]
    st.write("Last step:")
    cn, cs, light = last["state"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Cars North", cn)
    col2.metric("Cars South", cs)
    col3.metric("Light", "GREEN" if light == 0 else "RED")
    st.write(f"Action taken: {'SWITCH' if last['action']==1 else 'KEEP'}")
    st.write(f"Reward: {last['reward']}")

# show history table
if st.session_state.history:
    df = pd.DataFrame([{
        "cars_north": h["state"][0],
        "cars_south": h["state"][1],
        "light": "GREEN" if h["state"][2]==0 else "RED",
        "action": "SWITCH" if h["action"]==1 else "KEEP",
        "reward": h["reward"]
    } for h in st.session_state.history])
    st.dataframe(df.tail(10))
else:
    st.write("No steps yet. Press 'Step Simulation' or train the agent.")
