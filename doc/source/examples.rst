Quick Start Examples
====================

We have provided some demo scripts in `demo/` folder.
Once EasyRL has been installed, users are able to directly execute these scripts, e.g.,

.. code-block:: python
    :linenos:

    cd demo

train a dqn agent to solve problem of cartpole

.. code-block:: python
    :linenos:

    python run_dqn_on_cartpole.py
    python run_ddpg_on_pendulum.py
    
or run the distributed structure Ape-X with multi-processes

.. code-block:: python
    :linenos:

    demo/run_apex_agent_on_cartpole.sh
