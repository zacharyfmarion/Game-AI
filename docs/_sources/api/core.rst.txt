Core
===================================

Core contains the primitives for playing a game between two players.

.. autoclass:: core.Player
    :members:

.. autoclass:: core.Arena
    :members:

.. autoclass:: core.Game
    :members:

Note: There are two types of agents, agents that are trainable and agents that are not. If an agent is trainable then it inherits from the :code:`TrainableAgent` class and must implement all of the members defined below. For example, :code:`MCTSAgent` is a trainable agent, while :code:`MinimaxAgent` is not.

.. autoclass:: core.Agent
    :members:
    
.. autoclass:: core.TrainableAgent
    :members:
    
.. autoclass:: core.Algorithm
    :members: