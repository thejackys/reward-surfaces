SECONDS=0
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt

echo "DDPG" >> log.txt
bash run_ppo.sh Pendulum-v1 DDPG_Pendulum 0,1 DDPG SB3_OFF
bash run_ppo.sh MountainCarContinuous-v0 DDPG_MountainCarContinuous 0,1 DDPG SB3_OFF
bash run_ppo.sh LunarLanderContinuous-v2 DDPG_LunarLanderContinuous 0,1 DDPG SB3_OFF
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt


echo "ddpg_sam" >> log.txt
bash run_ppo.sh Pendulum-v1 ddpg_sam_Pendulum 0,1 ddpg_sam SB3_OFF
bash run_ppo.sh MountainCarContinuous-v0 ddpg_sam_MountainCarContinuous 0,1 ddpg_sam SB3_OFF
bash run_ppo.sh LunarLanderContinuous-v2 ddpg_sam_LunarLanderContinuous 0,1 ddpg_sam SB3_OFF
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt

echo "ppo" >> log.txt
bash run_ppo.sh Pendulum-v1 ppo_Pendulum 0,1 PPO SB3_ON
bash run_ppo.sh CartPole-v1 ppo_cartpole 0,1 PPO SB3_ON
bash run_ppo.sh MountainCarContinuous-v0 ppo_MountainCarContinuous 0,1 PPO SB3_ON
bash run_ppo.sh LunarLanderContinuous-v2 ppo_LunarLanderContinuous 0,1 PPO SB3_ON

duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt


echo "ppo_sam" >> log.txt
bash run_ppo.sh Pendulum-v1 sam_ppo_Pendulum 0,1 ppo_sam SB3_ON
bash run_ppo.sh CartPole-v1 sam_ppo_cartpole 0,1 ppo_sam SB3_ON
bash run_ppo.sh MountainCarContinuous-v0 sam_ppo_MountainCarContinuous 0,1 ppo_sam SB3_ON
bash run_ppo.sh LunarLanderContinuous-v2 sam_ppo_LunarLanderContinuous 0,1 ppo_sam SB3_ON
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." >> log.txt



