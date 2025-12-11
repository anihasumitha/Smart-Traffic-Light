# env.py
import random

class TrafficEnv:
    """
    Tiny traffic intersection simulator.
    State: (cars_north, cars_south, light)
      - cars_north: 0..5 (integer)
      - cars_south: 0..5
      - light: 0 for GREEN (north-south moving), 1 for RED (north-south stopped)
    Actions:
      - 0: keep current light
      - 1: switch light
    Reward:
      - negative of total waiting cars (we want to minimize waiting)
    """

    def __init__(self):
        # initialize state variables
        self.light = 0  # 0 = GREEN, 1 = RED
        self.cars_north = 0
        self.cars_south = 0

    def reset(self):
        # start episode with random counts to make the agent robust
        self.cars_north = random.randint(0, 5)
        self.cars_south = random.randint(0, 5)
        self.light = 0
        return self._get_state()

    def step(self, action):
        # apply action
        if action == 1:
            # flip the light: if green -> red, if red -> green
            self.light = 1 - self.light

        # new cars arrive each step (stochastic)
        self.cars_north = random.randint(0, 5)
        self.cars_south = random.randint(0, 5)

        # reward is negative waiting cars (lower is better)
        reward = - (self.cars_north + self.cars_south)

        return self._get_state(), reward

    def _get_state(self):
        # return a tuple representing current state
        return (self.cars_north, self.cars_south, self.light)
