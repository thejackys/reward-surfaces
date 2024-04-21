#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name
# python3 scripts/train_agent.py "./runs/cartpole_checkpoints" SB3_ON CartPole-v1 cuda '{"ALGO": "PPO"}' --save_freq=10000
export PYTHONPATH=/home/grads/yfs5313/data_yfs5313/RL/How_Sharp_Is_Your_Policy/reward-surfaces
python3 scripts/train_agent.py "./${checkpoint_name}_checkpoints" SB3_ON "$env_name" cuda '{"ALGO": "PPO", "num_envs": 1, "n_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "learning_rate": 0.0003, "batch_size": 64}' --save_freq=10000
python3 scripts/generate_eval_jobs.py --calc-hesh --num-steps=1000000 "./${checkpoint_name}_checkpoints/" "./${checkpoint_name}_eig_vecs/"
python3 scripts/run_jobs_multiproc.py "./${checkpoint_name}_eig_vecs/jobs.sh"
python3 scripts/generate_plane_jobs.py --dir1="./${checkpoint_name}_eig_vecs/results/0040000/maxeigvec.npz" --dir2="./${checkpoint_name}_eig_vecs/results/0040000/mineigvec.npz" --grid-size=31 --magnitude=1.0 --num-steps=200000 "./${checkpoint_name}_checkpoints/0040000" "./${checkpoint_name}_eig_vecs_plane/"
python3 scripts/run_jobs_multiproc.py "./${checkpoint_name}_eig_vecs_plane/jobs.sh"
python3 scripts/job_results_to_csv.py "./${checkpoint_name}_eig_vecs_plane/"
python3 scripts/plot_plane.py "./${checkpoint_name}_eig_vecs_plane/results.csv" --outname="${checkpoint_name}_curvature_plot.png"
