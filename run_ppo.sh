#!/usr/bin/env bash

env_name=$1
checkpoint_name=$2
gpu=$3
algo=$4
policy=$5


export CUDA_VISIBLE_DEVICES=$gpu
# do some work

# python3 scripts/train_agent.py "./runs/${checkpoint_name}_checkpoints" ${policy} ${env_name} cuda "{\"ALGO\": \"${algo}\"}" --save_freq=10000


##############
#eval####
#######
# python3 scripts/generate_eval_jobs.py --batch-grad --num-steps=1024 --num-episodes=2000 "./runs/${checkpoint_name}_checkpoints" "./runs/eval_grad/${checkpoint_name}/"
# python3 scripts/run_jobs_multiproc.py --num-cpus=64 "./runs/eval_grad/${checkpoint_name}/jobs.sh"

### generate training curve plot
python scripts/generate_eval_jobs.py --num-episodes=200 "./runs/${checkpoint_name}_checkpoints" "./runs/eval_grad/${checkpoint_name}/"
python scripts/run_jobs_multiproc.py --num-cpus=1 "./runs/eval_grad/${checkpoint_name}/jobs.sh"
python scripts/job_results_to_csv.py "./runs/eval_grad/${checkpoint_name}"
python scripts/plot_traj.py  "./runs/eval_grad/${checkpoint_name}/results.csv"

####################
#######Graphing######
####################
# python3 scripts/generate_plane_jobs.py --grid-size=31 --magnitude=1.0 --num-steps=20000 "./runs/${checkpoint_name}_checkpoints/best/" "./runs/${checkpoint_name}_surface/"
# python3 scripts/run_jobs_multiproc.py --num-cpus=64 "./runs/${checkpoint_name}_surface/jobs.sh"
# python3 scripts/job_results_to_csv.py "./runs/${checkpoint_name}_surface/"
python3 scripts/plot_plane.py "./runs/${checkpoint_name}_surface/results.csv" --outname="./runs/${checkpoint_name}_${env_name}_${algo}" --env_name="${env_name}"

duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt
