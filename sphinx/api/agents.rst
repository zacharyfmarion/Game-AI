Agents
===================================

Agents are essentially actors that takes actions based on some heuristic, either randomly in the case of `RandomAgent` or based on some trained information, in the case of `MCTSAgent`.

Interfaces
----------

There are two types of agents, agents that are trainable and agents that are not. If an agent is trainable then it inherits from the :code:`TrainableAgent` class and must implement all of the members defined below. For example, :code:`MCTSAgent` is a trainable agent, while :code:`MinimaxAgent` is not.

.. autoclass:: agents.Agent
    :members:
    
.. autoclass:: agents.TrainableAgent
    :members:
    
Classes
----------
    
.. autoclass:: agents.RandomAgent
    
.. autoclass:: agents.HumanAgent
    
.. autoclass:: agents.MinimaxAgent
    
.. autoclass:: agents.LimitedDepthMinimaxAgent
    
.. autoclass:: agents.MCTSAgent