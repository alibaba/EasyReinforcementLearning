import numpy as np
from easy_rl.utils.utils import prod

def ips_eval_old(batch_weights, batch_rewards, gamma):
    assert len(batch_weights) == len(batch_rewards), "length of weights must be same with length of rewards"
    all_weights = []
    all_weights_stepwise = []
    batch_ips_scores = []
    batch_ips_scores_stepwise = []
    for weights, rewards in zip(batch_weights, batch_rewards):
        w_init = prod(weights)
        all_weights.append(w_init)
        w_init_stepwise = 1.0

        w_steps = np.ones_like(weights)
        w_cum_tmp = 1.0
        w_steps_stepwise = []
        for w in weights:
            w_cum_tmp *= w
            w_steps_stepwise.append(w_cum_tmp)
        all_weights_stepwise.append(w_cum_tmp)

        ips_score = [w_init * w_step * (gamma ** idx) * r for idx, (w_step, r) in enumerate(zip(w_steps, rewards))]
        ips_score_stepwise = [w_init_stepwise * w_step * (gamma ** idx) * r for idx, (w_step, r) in enumerate(zip(w_steps_stepwise, rewards))]

        batch_ips_scores.append(ips_score)
        batch_ips_scores_stepwise.append(ips_score_stepwise)

    avg_weights = np.mean(all_weights)
    avg_weights_stepwise = np.mean(all_weights_stepwise)

    output_per_traj = [np.sum(ips_score) for ips_score in batch_ips_scores]
    output_per_traj_stepwise = [np.sum(ips_score) for ips_score in batch_ips_scores_stepwise]
    output_norm_per_traj = [np.sum(ips_score) / avg_weights for ips_score in batch_ips_scores]
    output_norm_per_traj_stepwise = [np.sum(ips_score) / avg_weights_stepwise for ips_score in batch_ips_scores_stepwise]

    ips = np.mean(output_per_traj)
    ips_stepwise = np.mean(output_per_traj_stepwise)
    wips = np.mean(output_norm_per_traj)
    wips_stepwise = np.mean(output_norm_per_traj_stepwise)

    return ips, ips_stepwise, wips, wips_stepwise


def ips_eval(batch_weights, batch_rewards, gamma, max_len=200):
    assert len(batch_weights) == len(batch_rewards), "length of weights must be same with length of rewards"

    batch_size = len(batch_weights)
    MAX_LEN = max_len
    all_weights_stepwise_arr = np.empty((batch_size, MAX_LEN))
    all_weights_stepwise_arr[:] = np.nan

    all_weights = []
    batch_ips_scores = []
    batch_ips_scores_stepwise = []

    for i_traj, (weights, rewards) in enumerate(zip(batch_weights, batch_rewards)):

        w_init = prod(weights)
        all_weights.append(w_init)
        w_init_stepwise = 1.0

        w_steps = np.ones_like(weights)
        w_cum_tmp = 1.0
        w_steps_stepwise = []

        for i_step, w in enumerate(weights):
            w_cum_tmp *= w
            all_weights_stepwise_arr[i_traj, i_step] = w_cum_tmp
            w_steps_stepwise.append(w_cum_tmp)

        ips_score = [w_init * w_step * (gamma ** idx) * r for idx, (w_step, r) in enumerate(zip(w_steps, rewards))]
        ips_score_stepwise = [w_init_stepwise * w_step * (gamma ** idx) * r for idx, (w_step, r) in
                              enumerate(zip(w_steps_stepwise, rewards))]

        batch_ips_scores.append(ips_score)
        batch_ips_scores_stepwise.append(ips_score_stepwise)

    avg_weights = np.mean(all_weights)

    avg_weights_stepwise_mean = np.nanmean(all_weights_stepwise_arr, 0)
    avg_weights_stepwise_mean = avg_weights_stepwise_mean[~np.isnan(avg_weights_stepwise_mean)]

    output_per_traj = [np.sum(ips_score) for ips_score in batch_ips_scores]
    output_per_traj_stepwise = [np.sum(ips_score) for ips_score in batch_ips_scores_stepwise]
    output_norm_per_traj = [np.sum(ips_score) / avg_weights for ips_score in batch_ips_scores]

    output_norm_per_traj_stepwise = []
    output_norm_per_traj_stepwise_mean = []
    for ips_score in batch_ips_scores_stepwise:
        batch_ips = []
        for ips, step_weight in zip(ips_score, avg_weights_stepwise_mean):
            avg_weights_stepwise = ips / step_weight
            batch_ips.append(avg_weights_stepwise)
        output_norm_per_traj_stepwise.append(np.sum(batch_ips))
        output_norm_per_traj_stepwise_mean.append(np.sum(batch_ips) / len(batch_ips))

    ips = np.mean(output_per_traj)
    ips_stepwise = np.mean(output_per_traj_stepwise)
    wips = np.mean(output_norm_per_traj)
    wips_stepwise = np.mean(output_norm_per_traj_stepwise)
    wips_stepwise_mean = np.mean(output_norm_per_traj_stepwise_mean)

    return ips, ips_stepwise, wips, wips_stepwise, wips_stepwise_mean
