from environment import Simulator
from agent import PolicyGradientAgent, CriticsAgent
import datetime as dt
import numpy as np

def main():
    actions = ["buy", "sell", "hold"]
    env_train = Simulator(['scg', 'wec'], dt.datetime(2002, 01, 04), dt.datetime(2016, 12, 30))

    #agent = PolicyGradientAgent(lookback=env_train.init_state())
    #critic_agent = CriticsAgent(lookback=env.init_state())
    while env_train.has_more():
        action = np.random.randint(3)
        action = actions[action] # map action from id to name
        print "Runner: Taking action", env_train.date, action
        reward, state = env_train.step(action)
        #action = agent.query(state, reward)

if __name__ == '__main__':
    main()
