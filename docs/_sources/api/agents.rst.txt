Agents
===================================

Agents are essentially actors that takes actions based on some heuristic, either randomly in the case of `RandomAgent` or based on some trained information, in the case of `MCTSAgent`.

Interfaces
----------

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