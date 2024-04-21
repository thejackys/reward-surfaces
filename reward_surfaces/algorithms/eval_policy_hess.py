import time
from .evaluate import mean, generate_data
from .evaluate_est_hesh import npvec_to_tensorlist, npvec_to_nplist
import numpy as np
import torch
import torch as th
from torch.nn import functional as F
from gym import spaces
from scipy.sparse.linalg import LinearOperator, eigsh


def gen_advantage_est_episode(rews, vals, decay, gae_lambda=1.):
    last_value = (1/decay)*(vals[-1]-rews[-1])# estiamte of next value

    advantages = [0]*len(rews)
    last_gae_lam = 0
    buf_size = len(rews)
    for step in reversed(range(buf_size)):
        if step == buf_size - 1:
            next_non_terminal = 0.
            next_values = last_value
        else:
            next_non_terminal = 1.
            next_values = vals[step + 1]
        delta = rews[step] + decay * next_values * next_non_terminal - vals[step]
        last_gae_lam = delta + decay * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    return advantages


def gen_advantage_est(rewards, values, decay, gae_lambda=1.):
    return [gen_advantage_est_episode(rew, val, decay, gae_lambda) for rew, val in zip(rewards, values)]


# def split_data(datas):
#     episode_datas = []
#     ep_data = []
#     for rew,done,value in datas:
#         ep_data.append((rew,value))
#         if done:
#             episode_datas.append(ep_data)
#             ep_data = []
#     return episode_datas


def mean_baseline_est(rewards):
    # Calculate average episode reward
    baseline = mean([sum(rew) for rew in rewards])
    # Create np array, with same length as episode, for each episode, where each element has the value of episode reward minus baseline
    baseline = mean([sum(rew) for rew in rewards])
    return [np.ones_like(rew) * (sum(rew)-baseline) for rew in rewards]


def decayed_baselined_values(rewards, decay):
    values = []
    for rews in rewards:
        vals = [0]*len(rews)
        vals[-1] = rews[-1]
        for i in reversed(range(len(vals)-1)):
            vals[i] = rews[i] + vals[i+1]*decay
        values.append(vals)

    baseline_val = mean([mean(vals) for vals in values])

    return [[val-baseline_val for val in vals] for vals in values]


def gather_policy_hess_data(evaluator, num_episodes, num_steps, gamma, returns_method='baselined_vals', gae_lambda=1.0):
    print("Gathering data")
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_value_ests = []

    ep_rews = []
    ep_values = []
    ep_states = []
    ep_actions = []
    tot_steps = 0
    start_t = time.time()
    done = False
    while not done or (len(episode_rewards) < num_episodes and tot_steps < num_steps):
        _, original_rew, done, value, state, act, info = evaluator._next_state_act() #, deterministic=True)
        ep_states.append(state)
        ep_actions.append(act)
        ep_rews.append(original_rew)
        ep_values.append(value)
        tot_steps += 1
        if done:
            episode_states.append(ep_states)
            episode_actions.append(ep_actions)
            episode_rewards.append(ep_rews)
            #episode_value_ests.append((ep_values))

            ep_rews = []
            ep_values = []
            ep_states = []
            ep_actions = []
            end_t = time.time()
            #print("done!", (end_t - start_t)/len(episode_rewards))

    returns = mean_baseline_est(episode_rewards)
    #print(returns)
    # if returns_method == 'baselined_vals':
    #returns = decayed_baselined_values(episode_rewards, gamma)
    # elif returns_method == 'gen_advantage':
    # returns = gen_advantage_est(episode_rewards, episode_value_ests, gamma, gae_lambda)
    # else:
    #     raise ValueError("bad value for `returns_method`")


    single_dim_grad = None
    #
    # all_states = sum(episode_states,[])
    # all_returns = sum(returns,[])
    # all_actions = sum(episode_actions,[])
    # print(len(all_returns))
    # print(len(sum(episode_rewards,[])))
    # print(len(sum(episode_states,[])))
    # exit(0)

    return episode_states, returns, episode_actions


def get_used_params(evaluator, states, actions):
    params = evaluator.parameters()
    out = torch.sum(evaluator.eval_log_prob(states, actions))
    grads = torch.autograd.grad(out, inputs=params, create_graph=True, allow_unused=True)
    new_params = [p for g, p in zip(grads, params) if g is not None]
    # grads = torch.autograd.grad(out, inputs=new_params, create_graph=True, allow_unused=True)
    # print(grads)
    return new_params


def zero_unused_params(params, policy_params, nplist):
    assert len(nplist) == len(policy_params)
    res_list = []
    policy_idx = 0
    for p in params:
        if policy_idx < len(policy_params) and policy_params[policy_idx] is p:
            npval = nplist[policy_idx]
            policy_idx += 1
        else:
            npval = np.zeros(p.shape, dtype=np.float32)
        res_list.append(npval)
    return res_list


def accumulate(accumulator, data):
    assert len(accumulator) == len(data)
    for a, d in zip(accumulator, data):
        a.data += d


def compute_grad_mags(evaluator, params, all_states, all_returns, all_actions):
    print("computing grad mag")
    device = params[0].device
    batch_size = 8
    num_grad_steps = 0
    mag_accum = [p.detach()*0 for p in params]
    grad_accum = [p.detach()*0 for p in params]

    # Iterate through every epsiode in dataset
    for eps in range(len(all_states)):
        # Access episode data
        eps_states = all_states[eps]
        eps_returns = all_returns[eps]
        eps_act = all_actions[eps]
        assert len(eps_act) == len(eps_states)
        assert len(eps_act) == len(eps_returns)

        # Iterate through each episode in batches:
        for idx in range(0, len(eps_act), batch_size):
            # Make sure batch doesn't extend past end of episode
            eps_batch_size = min(batch_size, len(eps_act) - idx)
            batch_states = torch.squeeze(torch.tensor(eps_states[idx:idx + eps_batch_size], device=device), dim=1)
            batch_actions = torch.tensor(eps_act[idx:idx + eps_batch_size], device=device).reshape(eps_batch_size, -1)
            batch_returns = torch.tensor(eps_returns[idx:idx + eps_batch_size], device=device).float()
            # Calculate log probabilities of each state-action pair
            logprob = evaluator.eval_log_prob(batch_states, batch_actions)
            # Calculate expected return over batch?
            logprob = torch.dot(logprob, batch_returns)

            # Add batch grad mags to accumulator
            grad = torch.autograd.grad(outputs=logprob, inputs=tuple(params))
            for g, ma, ga in zip(grad, mag_accum, grad_accum):
                ma.data += torch.square(g)
                ga.data += g

            num_grad_steps += 1

    # Scale final grad mags by number of gradients steps
    mag_accum = [m/num_grad_steps for m in mag_accum]
    grad_accum = [m/num_grad_steps for m in grad_accum]
    return mag_accum, grad_accum


def compute_grad_mags_batch(evaluator, params, action_evaluator, num_episodes, num_steps):
    print("computing batch grad mag")
    device = params[0].device
    batch_size = 8
    num_grad_steps = 0
    mag_accum = [p.detach()*0 for p in params]
    grad_accum = [p.detach()*0 for p in params]
    step_count = 0
    episode_count = 0
    all_episode_rewards = []
    while episode_count < num_episodes and step_count < num_steps:
        # Collect episode data
        episode_rewards = []
        episode_values = []
        episode_states = []
        episode_actions = []
        done = False
        while not done:
            _, original_reward, done, value, state, act, info = evaluator._next_state_act() #, deterministic=True)
            episode_states.append(state)
            episode_actions.append(act)
            episode_rewards.append(original_reward)
            episode_values.append(value)
            step_count += 1
            if done:
                all_episode_rewards.append(episode_rewards)
        baseline = mean([sum(rewards) for rewards in all_episode_rewards])
        episode_returns = np.ones_like(episode_rewards) * (sum(episode_rewards) - baseline)
        episode_count += 1

        # Compute grads from episode data
        assert len(episode_actions) == len(episode_states)
        assert len(episode_actions) == len(episode_returns)
        for idx in range(0, len(episode_actions), batch_size):
            clipped_batch_size = min(batch_size, len(episode_actions) - idx)
            # Convert the list of NumPy arrays to a single NumPy array
            batch_states_array = np.array(episode_states[idx:idx + clipped_batch_size])
            batch_actions_array = np.array(episode_actions[idx:idx + clipped_batch_size])
            batch_returns_array = np.array(episode_returns[idx:idx + clipped_batch_size])
            
            # Create tensors from the single NumPy arrays
            batch_states = torch.from_numpy(batch_states_array).to(device)
            batch_actions = torch.from_numpy(batch_actions_array).to(device)
            batch_returns = torch.from_numpy(batch_returns_array).to(device)
            
            # Fix batch dimensions
            batch_states = torch.squeeze(batch_states, dim=1)
            batch_actions = batch_actions.reshape(clipped_batch_size, -1)
            batch_returns = batch_returns.float()
            logprob = action_evaluator.eval_log_prob(batch_states, batch_actions)
            logprob = torch.dot(logprob, batch_returns)

            grad = torch.autograd.grad(outputs=logprob, inputs=tuple(params))
            for g, ma, ga in zip(grad, mag_accum, grad_accum):
                ma.data += torch.square(g)
                ga.data += g

            num_grad_steps += 1

    mag_accum = [m/num_grad_steps for m in mag_accum]
    grad_accum = [m/num_grad_steps for m in grad_accum]
    return mag_accum, grad_accum


def compute_policy_gradient(evaluator, all_states, all_returns, all_actions, device):
    device = evaluator.parameters()[0].device
    # torch.squeeze is used to fix atari observation shape
    params = get_used_params(evaluator, torch.squeeze(torch.tensor(all_states[0][0:2], device=device), dim=1),
                             torch.tensor(all_actions[0][0:2], device=device))

    grad_mag, grad_dir = compute_grad_mags(evaluator, params, all_states, all_returns, all_actions)

    grad_dir = zero_unused_params(evaluator.parameters(), params, grad_dir)
    grad_mag = zero_unused_params(evaluator.parameters(), params, grad_mag)

    return grad_dir, grad_mag


def compute_policy_gradient_batch(evaluator, action_evaluator, num_episodes, num_steps):
    device = action_evaluator.parameters()[0].device
    # torch.squeeze is used to fix atari observation shape
    test_states, test_returns, test_actions = gather_policy_hess_data(evaluator,
                                                                      2,
                                                                      num_steps,
                                                                      action_evaluator.gamma,
                                                                      "UNUSED",
                                                                      gae_lambda=1.0)
    test_states_array = np.array(test_states[0][0:2])
    test_actions_array = np.array(test_actions[0][0:2])

    params = get_used_params(action_evaluator, 
                             torch.squeeze(torch.from_numpy(test_states_array).to(device), dim=1), 
                             torch.from_numpy(test_actions_array).to(action_evaluator.device))

    # params = get_used_params(action_evaluator,
    #                          torch.squeeze(torch.tensor(test_states[0][0:2], device=device), dim=1),
    #                          torch.tensor(test_actions[0][0:2], device=action_evaluator.device))

    grad_mag, grad_dir = compute_grad_mags_batch(evaluator, params, action_evaluator, num_episodes, num_steps)

    grad_dir = zero_unused_params(action_evaluator.parameters(), params, grad_dir)
    grad_mag = zero_unused_params(action_evaluator.parameters(), params, grad_mag)

    return grad_dir, grad_mag


def compute_vec_hesh_prod(evaluator, params, all_states, all_returns, all_actions, vec, batch_size=512):
    device = params[0].device
    accum = [p*0 for p in params]
    assert len(all_states) == len(all_actions)
    assert len(all_states) == len(all_returns)
    for eps in range(len(all_states)):
        grad_accum = [p*0 for p in params]
        grad_m_mr_dot_v_accum = torch.zeros(1, device=device)
        hesh_prod_accum = [p*0 for p in params]
        eps_states = all_states[eps]
        eps_returns = all_returns[eps]
        eps_act = all_actions[eps]
        assert len(eps_act) == len(eps_states)
        assert len(eps_act) == len(eps_returns)
        for idx in range(0, len(eps_act), batch_size):
            eps_batch_size = min(batch_size, len(eps_act) - idx)
            batch_states = torch.squeeze(torch.tensor(eps_states[idx:idx + eps_batch_size], device=device), dim=1)
            batch_actions = torch.squeeze(torch.tensor(eps_act[idx:idx + eps_batch_size],
                                                       device=device).reshape(eps_batch_size, -1))
            batch_returns = torch.tensor(eps_returns[idx:idx + eps_batch_size], device=device).float()

            logprob = torch.sum(evaluator.eval_log_prob(batch_states, batch_actions))
            grads = torch.autograd.grad(outputs=logprob, inputs=tuple(params), create_graph=True)

            logprob_mul_return = torch.dot(evaluator.eval_log_prob(batch_states, batch_actions), batch_returns)
            grad_mul_ret = torch.autograd.grad(outputs=logprob_mul_return, inputs=tuple(params), create_graph=True)
            assert len(vec) == len(grads)
            g_mr_dot_v = sum([torch.dot(g_mr.view(-1), v.view(-1)) for g_mr, v in zip(grad_mul_ret, vec)], torch.zeros(1, device=device))

            hesh_prods = torch.autograd.grad(g_mr_dot_v, inputs=params, create_graph=True)
            assert len(hesh_prods) == len(vec)
            grad_m_mr_dot_v_accum.data += g_mr_dot_v
            #accumulate(grad_mul_ret_accum,grad_mul_ret)
            accumulate(grad_accum, grads)
            accumulate(hesh_prod_accum, hesh_prods)

        # grad_vec_prod = sum([torch.dot(g_acc.view(-1),v.view(-1)) for g_acc,v in zip(grad_accum, vec)], torch.zeros(1,device=device))
        t1s = [g_mr_acc * grad_m_mr_dot_v_accum for g_mr_acc in grad_accum]
        t2s = hesh_prod_accum
        assert len(accum) == len(t1s) == len(t2s)
        for acc, t1, t2 in zip(accum, t1s, t2s):
            acc.data += (t1 + t2)

    return accum


def gradtensor_to_npvec(params, include_bn=True):
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.data.cpu().numpy().ravel() for p in params if filter(p)])


def calculate_true_hesh_eigenvalues(evaluator, all_states, all_returns, all_actions, tol, device):
    evaluator.dot_prod_calcs = 0
    device = evaluator.parameters()[0].device

    params = get_used_params(evaluator, torch.squeeze(torch.tensor(all_states[0][0:2], device=device)),
                             torch.squeeze(torch.tensor(all_actions[0][0:2], device=device)))
    #grad_mags = compute_grad_mags(evaluator, params, all_states, all_returns, all_actions)

    def hess_vec_prod(vec):
        evaluator.dot_prod_calcs += 1
        vec = npvec_to_tensorlist(vec, params, device)
        accum = compute_vec_hesh_prod(evaluator, params, all_states, all_returns, all_actions, vec)
        return gradtensor_to_npvec(accum)

    N = sum(np.prod(param.shape) for param in params)
    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=tol, which="LA")
    maxeigval = eigvals[0]
    maxeigvec = npvec_to_nplist(eigvecs.reshape(N), params)
    maxeigvec = zero_unused_params(evaluator.parameters(), params, maxeigvec)
    print(f"max eignvalue = {maxeigval}")

    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=tol, which="SA")
    mineigval = eigvals[0]
    mineigvec = npvec_to_nplist(eigvecs.reshape(N), params)
    mineigvec = zero_unused_params(evaluator.parameters(), params, mineigvec)
    print(f"min eignvalue = {mineigval}")

    assert maxeigval > 0, "something weird is going on"

    print("number of evaluations required: ", evaluator.dot_prod_calcs)

    return float(maxeigval), float(mineigval), maxeigvec, mineigvec


#
