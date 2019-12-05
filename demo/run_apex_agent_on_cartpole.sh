#!/bin/bash

rm -f ps_0.stdout
rm -f memory_0.stdout
rm -f actor_0.stdout
rm -f actor_1.stdout
rm -f learner_0.stdout
rm -f learner_1.stdout

ps_hosts='localhost:6670'
memory_hosts='localhost:6667'
actor_hosts='localhost:6668'
learner_hosts='localhost:6669,localhost:6672'

CUDA_VISIBLE_DEVICES="" python run_apex_agent_on_cartpole.py --config "configs/apex_on_cartpole.json" --job_name "ps" --task_index 0 --ps_hosts $ps_hosts --memory_hosts $memory_hosts --actor_hosts $actor_hosts --learner_hosts $learner_hosts >> ps_0.stdout &

CUDA_VISIBLE_DEVICES="" python run_apex_agent_on_cartpole.py --config "configs/apex_on_cartpole.json" --job_name "memory" --task_index 0 --ps_hosts $ps_hosts --memory_hosts $memory_hosts --actor_hosts $actor_hosts --learner_hosts $learner_hosts >> memory_0.stdout &

CUDA_VISIBLE_DEVICES="" python run_apex_agent_on_cartpole.py --config "configs/apex_on_cartpole.json" --job_name "actor" --task_index 0 --ps_hosts $ps_hosts --memory_hosts $memory_hosts --actor_hosts $actor_hosts --learner_hosts $learner_hosts >> actor_0.stdout &

#CUDA_VISIBLE_DEVICES="" python run_apex_agent_on_cartpole.py --config "configs/apex_on_cartpole.json" --job_name "actor" --task_index 1 --ps_hosts $ps_hosts --memory_hosts $memory_hosts --actor_hosts $actor_hosts --learner_hosts $learner_hosts >> actor_1.stdout &

CUDA_VISIBLE_DEVICES="" python run_apex_agent_on_cartpole.py --config "configs/apex_on_cartpole.json" --job_name "learner" --task_index 0 --ps_hosts $ps_hosts --memory_hosts $memory_hosts --actor_hosts $actor_hosts --learner_hosts $learner_hosts >> learner_0.stdout &

CUDA_VISIBLE_DEVICES="" python run_apex_agent_on_cartpole.py --config "configs/apex_on_cartpole.json" --job_name "learner" --task_index 1 --ps_hosts $ps_hosts --memory_hosts $memory_hosts --actor_hosts $actor_hosts --learner_hosts $learner_hosts >> learner_1.stdout &
