#!/usr/bin/env bash

env_name=$1
checkpoint_name=$2
gpu=$3

SECONDS=0
export CUDA_VISIBLE_DEVICES=$gpu
# do some work

python3 scripts/train_agent.py "./runs/${checkpoint_name}_checkpoints" SB3_OFF ${env_name} cuda '{"ALGO": "DQN"}' --save_freq=10000
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > log.txt

# python3 scripts/generate_eval_jobs.py --batch-grad --num-steps=1024 --num-episodes=2000 "./runs/${checkpoint_name}_checkpoints" "./runs/eval_grad/${checkpoint_name}/"
# python3 scripts/run_jobs_multiproc.py --num-cpus=64 "./runs/eval_grad/${checkpoint_name}/jobs.sh"
# duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt

# python3 scripts/generate_plane_jobs.py --grid-size=31 --magnitude=1.0 --num-steps=20000 "./runs/${checkpoint_name}_checkpoints/best/" "./runs/${checkpoint_name}_surface/"
# duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt

# python3 scripts/run_jobs_multiproc.py --num-cpus=64 "./runs/${checkpoint_name}_surface/jobs.sh"
# duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt

# python3 scripts/job_results_to_csv.py "./runs/${checkpoint_name}_surface/"
# duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt

# python3 scripts/plot_plane.py "./runs/${checkpoint_name}_surface/results.csv" --outname="./runs/${env_name}" --env_name="${env_name}" --type="mesh"
# duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt