

Uses pytorch-geometric to learn on a graph with all possible actions against the DOM.
User provides a set of instructions without knowledge of the DOM.
The agent matches instructions to projected actions and executes.

Usage
=====

Example login with arsenic::

  from fuzzdom.action_chains import FuzzyActionChains

  await driver.get("http://localhost/login")
  actions = FuzzyActionChains(
      driver, "Enter username and password and login"
  )
  actions.input_field("username", "BillyJane")
  actions.input_field("password", "pas$word")
  actions.submit()
  await actions.async_perform()


Features
========

* Graph network representation to reduce action space to a single discrete selection
* Does not implement/imports RL Agent
* OpenAI gym compatible environment and wrappers (adds graph action space support)
* asyncio interface with arsenic

TODO: GAIL Dataset


Setup
=====

Getting Started::

  docker-compose build


Run Unit tests::

  docker-compose run app pytest /code/fuzzdom


Training
========


Train DOM Autoencoder::

  docker-compose run app python -m fuzzdom.datasets
  docker-compose run app python -m fuzzdom.train.autoencoder


Train agent::

  docker-compose run app python -m fuzzdom.train.graph --num-processes=20 --num-steps=30 --log-interval=1 --algo=ppo --env-name=levels


Inspirations
============

Reinforcement Learning on Web Interfaces using Workflow
https://arxiv.org/pdf/1802.08802

https://github.com/Sheng-J/DOM-Q-NET


Software Used
=============

* https://github.com/stanfordnlp/miniwob-plusplus
* https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
* https://github.com/rusty1s/pytorch_geometric
