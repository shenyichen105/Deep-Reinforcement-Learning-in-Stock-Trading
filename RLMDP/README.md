Deep Reinforcement Learning for Pairs Trading using Actor-critic
===============================================================

This project is about using a deep reinforcement learning advantage actor-critic algorithm with a critic network and a target network to train a trading agent. All related source code and data is available in the `RLMDP` folder.

## Installation
This project is implemented in python. In order to run below dependencies need to install first:

- theano
- lasagne
- numpy
- statsmodels
- pandas

All these python modules can be easily installed through Anaconda, a python package manage tool free available for [download](https://www.continuum.io/downloads).

```
conda install theano lasagne numpy statsmodels pandas
```

It's highly recommended to run the code on a GPU server as it's very time-consuming to run on CPU machine.

## Running
The project can be easily invoked by run the `runner.py` in `RLMDP` folder.
```
cd RLMDP
python2 runner
```
