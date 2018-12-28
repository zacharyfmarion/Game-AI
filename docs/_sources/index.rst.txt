.. Game AI documentation master file, created by
   sphinx-quickstart on Fri Dec 28 12:22:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Game AI's documentation!
===================================

GameAI contains a series of well-defined abstractions that are common in AI, such as a games, agents, and trainers which optimize the behavior of agents. As long as a class inherits from the base implemetation of a primitive (e.g. :code:`Agent`) and implements the required methods, it can be used in place of the standard implementations given.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   getting-started.rst
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api/core.rst
   api/agents.rst
   api/games.rst
   api/trainers.rst