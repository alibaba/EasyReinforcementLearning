How to Use EasyRL on PAI
===========================

The recommended way to use EasyRL is to package your code with it and submit task on PAI by `a3gentnew` command. we have provided `tutourial <http://gitlab.alibaba-inc.com/A3gent_Customer/tutourial_for_a3gentnew>`_ to show how to use EasyRL on PAI.

submit a non-interaction training task on PAI.
---------------------------------------------------------

first clone `easy_rl <https://github.com/alibaba/EasyRL>`_ to the directory of tutourial and prepare your odps table, we have provided sample data in the `sample_data` folder

then modify the customized configuration

.. code-block:: python
    :linenos:

    cd non-interaction
  
specify the path of odpscmd in `run_single.sh`

.. code-block:: python
    :linenos:

    ODPSCMD="/Users/ohso/odps_dir/bin/odpscmd"
    CONFIG="/Users/ohso/ser.ini"

specify the table name and buckets configuration

.. code-block:: python
    :linenos:

    table='odps://sre_mpi_algo_dev/tables/a3gent_offline_data_tmp'
    buckets='oss://hxpaitfoss/?role_arn=acs:ram::1418xxxxxxxxxxxx06:role/paitfsredev&host=cn-hangzhou.oss-internal.aliyun-inc.com'

finally, submit the task

.. code-block:: python
    :linenos:

    ./run_single.sh run_dqn_on_pai.py

you can change the dqn task to ddpg task by passing the different main script

.. code-block:: python
    :linenos:

    ./run_single.sh run_ddpg_on_pai.py

submit the interactive training task on PAI with simulator.
-----------------------------------------------------------

We also have migrated the `tutourials <https://yuque.antfin-inc.com/pai/a3gent/sosnwc>`_ to the easy_rl including `RankingPolicy with ES/DDPG` and `News deliver with PPO`.

you can find them in the `simulator-interaction` folder.

run the ES agent to optimize the ranking policy

.. code-block:: python
    :linenos:

    ./run_sync_dist.sh ranking_policy_env.py run_es_with_simulator_on_pai.py

in `run_sync_dist.sh`, set `actorNum` to different value to change the scalability. `workerNum` and `memoryNum` shoule be equal to 1.

run ddpg agent to optimize the same ranking policy

.. code-block:: python
    :linenos:

    ./run_async_dist.sh run_apex_ddpg_with_simulator_on_pai.py

in `run_async_dist.sh`, increase `actorNum` to speed up the interaction progress and increase `workerNum` to speed up the training progress. Note that `memoryNum` should not be less than `workerNum`.

