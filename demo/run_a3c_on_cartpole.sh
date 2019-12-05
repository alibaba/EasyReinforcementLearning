#!/bin/bash

rm -f ps_0.stdout
rm -f worker_0.stdout
rm -f worker_1.stdout


CUDA_VISIBLE_DEVICES="" python run_a3c_on_cartpole.py --config "configs/a3c_on_cartpole.json" --job_name "ps" --task_index 0 --ps_hosts "localhost:6670" --worker_hosts "localhost:6668,localhost:6669" >> ps_0.stdout &

CUDA_VISIBLE_DEVICES="" python run_a3c_on_cartpole.py --config "configs/a3c_on_cartpole.json" --job_name "worker" --task_index 0 --ps_hosts "localhost:6670" --worker_hosts "localhost:6668,localhost:6669" >> worker_0.stdout &

CUDA_VISIBLE_DEVICES="" python run_a3c_on_cartpole.py --config "configs/a3c_on_cartpole.json" --job_name "worker" --task_index 1 --ps_hosts "localhost:6670" --worker_hosts "localhost:6668,localhost:6669" >> worker_1.stdout &
