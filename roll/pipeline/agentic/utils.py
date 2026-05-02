import copy
import multiprocessing
import os
import os.path
import shutil
import subprocess
from datetime import datetime
from multiprocessing import Pool
from typing import List, Callable, Dict, Optional, Tuple

import imageio
import numpy as np
import torch
from codetiming import Timer
from torch import Tensor

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import AgenticConfig, RewardNormalizationConfig
from roll.pipeline.rlvr.utils import DUMPING_FUNC
from roll.utils.logging import get_logger
from roll.utils.functionals import (
    masked_whiten,
    compute_gae_advantage_return,
    compute_clip_fraction,
    compute_reinforce_return,
    compute_approx_kl,
)

logger = get_logger()


def dump_rollout_render(save_dir, step, frames: List[List], env_ids: List, tags: List, episode_scores: List):
    with Timer(name="dump", logger=None) as timer:
        try:
            local_save_dir = f'/tmp/rollout_render/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            os.makedirs(local_save_dir, exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            args_list = [
                (os.path.join(local_save_dir, f"{step}", f"{env_id}_{tag}_{episode_score:.1f}.gif"), frame_list)
                for frame_list, env_id, tag, episode_score in zip(frames, env_ids, tags, episode_scores)
                if len(frame_list) > 0
            ]
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with Pool(processes=16) as pool:
                pool.starmap(dump_frames_as_gif, args_list)

            rar_file_path = os.path.join(
                "/tmp", f'rollout_render_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{step}.zip'
            )
            command = ["zip", "-rq", rar_file_path, local_save_dir]
            subprocess.run(command, check=True)
            shutil.move(rar_file_path, save_dir)
            shutil.rmtree(local_save_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"dump rollout render failed: {e}")
    logger.info(f"dump_rollout_render_cost: {timer.last}")


@torch.no_grad()
def compute_discounted_returns(batch: DataProto, adv_estimator, gamma=1.0) -> DataProto:
    """
    Compute discounted returns for each trajectory in the batch.

    Args:
        batch (DataProto): A `DataProto` instance containing trajectories.
        adv_estimator (str): Advantage estimator type; only `"gigpo"` triggers computation here.
        gamma (float, optional): Discount factor applied to future rewards. Defaults to 1.0.

    Returns:
        DataProto: Updated batch where each trajectory contains an extra tensor key
                   `"step_rewards"` holding the computed discounted returns.
    """
    if adv_estimator in ["gigpo", "step_reinforce"]:
        batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
        batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
        for traj_id, traj_batch in batch_group_by_traj.items():

            indices: Tensor = torch.argsort(torch.from_numpy(traj_batch.non_tensor_batch["step"].astype(np.int64)))
            traj_batch.reorder(indices)
            step_scores = traj_batch.non_tensor_batch["step_scores"].astype(np.float32)
            rewards = torch.as_tensor(step_scores).float()
            discounts = torch.empty_like(rewards)
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                discounts[t] = running_return
            traj_batch.batch["step_rewards"] = discounts

        merged = DataProto.concat(list(batch_group_by_traj.values()))
        merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
        merged.pop("sample_order_placeholder")
        return merged
    else:
        return batch


# TODO: 这里的功能性和rlvr比较接近，但因为后续agentic会有潜在的修改需求，所以就先拎出来
@torch.no_grad()
def _blended_baseline_norm(
    scores: torch.Tensor,
    opponent_ids: np.ndarray,
    c: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Empirical Bayesian Shrinkage normalization for multi-opponent GRPO.

    Blends per-opponent statistics with global batch statistics. λ_k = n_k / (n_k + c).
    High n_k → local baseline; low n_k → global baseline. Handles n_k == 1 without NaN.
    """
    eps = 1e-5
    device = scores.device
    s = scores.float()

    mu_g = s.mean()
    var_g = ((s - mu_g) ** 2).mean()

    advantages = torch.zeros_like(s)
    unique_opps = np.unique(opponent_ids)
    lambdas: List[float] = []
    n_ks: List[int] = []

    for opp_id in unique_opps:
        opp_mask = torch.from_numpy(opponent_ids == opp_id).to(device)
        opp_scores = s[opp_mask]
        n_k = opp_scores.numel()
        n_ks.append(n_k)

        mu_k = opp_scores.mean()
        # n_k == 1: local variance is 0 by definition (single sample, no spread)
        var_k = ((opp_scores - mu_k) ** 2).mean() if n_k > 1 else s.new_zeros(1).squeeze()

        lambda_k = n_k / (n_k + c)
        lambdas.append(lambda_k)

        mu_hat = lambda_k * mu_k + (1 - lambda_k) * mu_g
        var_hat = lambda_k * var_k + (1 - lambda_k) * var_g
        sigma_hat = torch.sqrt(var_hat.clamp(min=0.0))

        advantages[opp_mask] = (opp_scores - mu_hat) / (sigma_hat + eps)

    metrics: Dict[str, float] = {
        "critic/blended/unique_opponents": float(len(unique_opps)),
        "critic/blended/mean_n_k": float(np.mean(n_ks)),
        "critic/blended/mean_lambda": float(np.mean(lambdas)),
    }
    return advantages.to(dtype=scores.dtype), metrics


@torch.no_grad()
def agentic_reward_norm(
    batch: "DataProto",
    reward_normalization: RewardNormalizationConfig,
    filter_zero_variance_groups: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalize rewards per group and return (normalized_rewards, metrics).

    Args:
        filter_zero_variance_groups: If True, zero out rewards for groups where all
            members have the same score (std < 1e-6). This prevents GRPO from
            producing near-zero advantages that provide no learning signal.
    """
    assert not (filter_zero_variance_groups and reward_normalization.grouping == "traj_group_id"), (
        "filter_zero_variance_groups and grouping='traj_group_id' are mutually exclusive: "
        "small traj_group_id groups will frequently be zero-variance by chance, "
        "silently discarding most of the batch."
    )
    if reward_normalization.blended_baseline_c is not None:
        if filter_zero_variance_groups:
            raise ValueError(
                "blended_baseline_c and filter_zero_variance_groups cannot both be set: "
                "shrinkage already handles the zero-variance case via the global variance floor."
            )
        if "opponent_id" not in batch.non_tensor_batch:
            raise ValueError(
                "blended_baseline_c requires 'opponent_id' in non_tensor_batch. "
                "Ensure TwoPlayerTrajEnvManager.formulate_rollouts() stores it."
            )
        scores = batch.batch["scores"]
        opponent_ids = batch.non_tensor_batch["opponent_id"]
        return _blended_baseline_norm(scores, opponent_ids, c=reward_normalization.blended_baseline_c)
    batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
    grouping = reward_normalization.grouping
    norm_mean_type = reward_normalization.norm_mean_type
    norm_std_type = reward_normalization.norm_std_type

    all_scores = batch.batch["scores"].float()
    batch_mean = None
    batch_std = None
    if norm_mean_type == "batch":
        batch_mean = all_scores.mean()
    if norm_std_type == "batch":
        batch_std = all_scores.std()

    batch_list = []
    group_stds = []
    group_means = []
    num_zero_variance_groups = 0
    num_total_groups = 0
    batch_grouped: Dict[str, DataProto] = {"default": batch}
    if grouping != "batch":
        if "init_state_id" in batch.non_tensor_batch:
            tgids = batch.non_tensor_batch["traj_group_id"]
            sids = batch.non_tensor_batch["init_state_id"]
            for tgid in np.unique(tgids):
                unique_sids = np.unique(sids[tgids == tgid])
                assert len(unique_sids) == 1, \
                    f"traj_group_id {tgid} has multiple init_state_ids: {unique_sids}"
            grouping = "init_state_id"
        batch_grouped = batch.group_by(keys=grouping)
    for group_name, group_batch in batch_grouped.items():
        scores = group_batch.batch["scores"]
        original_dtype = scores.dtype
        scores_float = scores.float()
        num_total_groups += 1

        # Track per-group stats
        group_std_val = scores_float.std().item() if scores_float.numel() > 1 else 0.0
        group_stds.append(group_std_val)
        group_means.append(scores_float.mean().item())

        if norm_mean_type == "batch":
            reward_mean = batch_mean
        elif norm_mean_type == "group":
            reward_mean = scores_float.mean()
        else:
            reward_mean = 0.0

        if norm_std_type == "batch":
            reward_std = batch_std
        elif norm_std_type == "group":
            reward_std = scores_float.std()
        else:
            reward_std = None

        is_zero_variance = scores_float.numel() <= 1 or (reward_std is not None and reward_std.abs() < 1e-6)
        if is_zero_variance:
            # Also check raw score std for groups without std normalization
            if reward_std is None and scores_float.numel() > 1 and scores_float.std().abs() < 1e-6:
                is_zero_variance = True
            elif reward_std is None:
                is_zero_variance = False

        if is_zero_variance:
            num_zero_variance_groups += 1

        if reward_std is not None:
            # 处理单个元素或标准差为0的情况，避免除以0
            if scores_float.numel() > 1 and reward_std.abs() > 1e-6:
                normalized_scores = (scores_float - reward_mean) / (reward_std + 1e-6)
            else:
                normalized_scores = torch.zeros_like(scores_float)
        else:
            normalized_scores = scores_float - reward_mean

        # Option B: filter zero-variance groups by zeroing their rewards
        if filter_zero_variance_groups and is_zero_variance:
            normalized_scores = torch.zeros_like(scores_float)

        normalized_scores = normalized_scores.to(dtype=original_dtype)
        group_batch.batch["grouped_rewards"] = normalized_scores
        batch_list.append(group_batch)

    # Build metrics
    norm_metrics: Dict[str, float] = {}
    if group_stds:
        norm_metrics["critic/group_reward_std/mean"] = float(np.mean(group_stds))
        norm_metrics["critic/group_reward_std/min"] = float(np.min(group_stds))
        norm_metrics["critic/group_reward_std/max"] = float(np.max(group_stds))
        norm_metrics["critic/group_reward_mean/mean"] = float(np.mean(group_means))
        norm_metrics["critic/zero_variance_groups"] = float(num_zero_variance_groups)
        norm_metrics["critic/zero_variance_group_frac"] = float(num_zero_variance_groups) / max(num_total_groups, 1)

    batch = DataProto.concat(batch_list)
    batch.reorder(indices=torch.argsort(batch.batch["sample_order_placeholder"]))
    batch.pop("sample_order_placeholder")
    return batch.batch.pop("grouped_rewards"), norm_metrics


def build_state_group(batch: "DataProto") -> "DataProto":
    batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
    batch_group_by_traj_group: Dict[str, DataProto] = batch.group_by(keys="traj_group_id")
    merged = []
    for traj_group_id, traj_group_batch in batch_group_by_traj_group.items():
        batch_group_by_state: Dict[str, DataProto] = traj_group_batch.group_by(keys="state_hash")
        for state, state_batch in batch_group_by_state.items():
            state_batch.non_tensor_batch["state_group_id"] = np.array(
                [state] * state_batch.batch.batch_size[0], dtype=object
            )
            merged.append(state_batch)
    state_batch_size = [len(m) for m in merged]
    merged = DataProto.concat(merged)
    merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
    merged.pop("sample_order_placeholder")
    metrics = merged.meta_info.pop("metrics", {})
    metrics["system/state_batch_size/max"] = np.max(state_batch_size)
    metrics["system/state_batch_size/mean"] = np.mean(state_batch_size)
    metrics["system/state_batch_size/min"] = np.min(state_batch_size)
    merged.meta_info["metrics"] = metrics
    return merged


@torch.no_grad()
def compute_response_level_rewards(batch: "DataProto", pipeline_config: AgenticConfig) -> "DataProto":
    reward_metrics = {}
    filter_zv = getattr(pipeline_config, "filter_zero_variance_groups", False)
    if pipeline_config.adv_estimator == "gigpo":
        # ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/gigpo/core_gigpo.py#L109
        # step 1
        episode_scores = torch.from_numpy(batch.non_tensor_batch["episode_scores"].astype(np.float32))
        scores_to_group = DataProto.from_dict({"scores": episode_scores})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        episode_rewards, ep_norm_metrics = agentic_reward_norm(
            scores_to_group, reward_normalization=pipeline_config.reward_normalization,
            filter_zero_variance_groups=filter_zv,
        )
        reward_metrics.update(ep_norm_metrics)

        # step 2
        batch = build_state_group(batch=batch)

        # step 3
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        step_rewards, step_norm_metrics = agentic_reward_norm(
            batch=scores_to_group,
            reward_normalization=RewardNormalizationConfig(
                grouping="state_group_id", method=pipeline_config.reward_normalization.method
            ),
            filter_zero_variance_groups=filter_zv,
        )
        reward_metrics.update({f"step_{k}": v for k, v in step_norm_metrics.items()})

        batch.batch["response_level_rewards"] = (
            pipeline_config.episode_reward_weight * episode_rewards + pipeline_config.step_reward_weight * step_rewards
        )
        batch.batch["episode_rewards_norm"] = episode_rewards
        batch.batch["step_rewards_norm"] = step_rewards
    elif pipeline_config.adv_estimator == "step_reinforce":
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        rewards, norm_metrics = agentic_reward_norm(
            scores_to_group, reward_normalization=pipeline_config.reward_normalization,
            filter_zero_variance_groups=filter_zv,
        )
        batch.batch["response_level_rewards"] = rewards
        reward_metrics.update(norm_metrics)
    else:
        scores_to_group = DataProto.from_dict({"scores": batch.batch["scores"].clone().sum(dim=-1)})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        rewards, norm_metrics = agentic_reward_norm(
            scores_to_group, reward_normalization=pipeline_config.reward_normalization,
            filter_zero_variance_groups=filter_zv,
        )
        batch.batch["response_level_rewards"] = rewards
        reward_metrics.update(norm_metrics)

    # 加上clip
    if pipeline_config.reward_clip:
        reward_metrics["critic/reward_clip_frac"] = compute_clip_fraction(
            values=batch.batch["response_level_rewards"],
            clip_min=-pipeline_config.reward_clip,
            clip_max=pipeline_config.reward_clip,
        )
        batch.batch["response_level_rewards"] = torch.clamp(
            batch.batch["response_level_rewards"], min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

    return batch, reward_metrics


@torch.no_grad()
def get_agentic_response_level_mask(data: "DataProto", pipeline_config: AgenticConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    # mask相关策略
    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"][:, 1:].clone()

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    if getattr(pipeline_config, "max_len_mask", False):
        # TODO 当前是混合多个的action/state，需要去判别，或者用别的方式过滤
        final_sample_mask = final_sample_mask
        mask_metrics["actor/max_len_mask_ratio"] = 1.0
    else:
        mask_metrics["actor/max_len_mask_ratio"] = 1.0

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["final_response_mask"] = final_response_mask
    return data, mask_metrics


print_only_once = False


def dump_frames_as_gif(filename, frames, duration=0.2):
    global print_only_once
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with imageio.get_writer(filename, mode="v", duration=duration) as writer:
            for frame in frames:
                writer.append_data(frame.astype(np.uint8))

    except Exception as e:
        if not print_only_once:
            print(f"Error saving gif: {e}")
        print_only_once = True
        pass


def remove_nan_items(data: Dict[str, np.ndarray]):
    if not data:
        return {}

    # 所有数组都假设 dtype=object，只有 None 需要过滤
    arr = np.vstack([np.asarray(v, dtype=object) for v in data.values()])  # (num_keys, N)
    mask = arr != None  # noqa: E711
    valid_row_mask = mask.all(axis=0)
    return {
        k: np.asarray(v, dtype=object)[valid_row_mask]
        for k, v in data.items()
    }


def dump_rollout_trajectories(path, global_step, data: DataProto):
    """
    Dumps rollout trajectories to persistent storage.

    The data is written using a column-based configuration defined in COLUMMNS_CONFIG.
    Each column is specified as a list [column_name, data_type], where:
    - column_name: string identifier for the column
    - data_type: data type specification ('bigint', 'string', 'double', etc.)

    Example configuration:
    colummns_config = [
        ['global_step', 'bigint'],
        ['id', 'string'],
        ['source', 'string'],
        # ... additional columns
    ]
    """
    if not path:
        return

    columns_config: Optional[List] = data.meta_info.get("COLUMMNS_CONFIG", None)
    if columns_config is None:
        return

    write_data = {item[0]: data.non_tensor_batch.pop(item[0]) for item in columns_config if item[0] in data.non_tensor_batch}

    write_data = remove_nan_items(copy.deepcopy(write_data))
    data_cnt = len(write_data[columns_config[0][0]])

    write_data["global_step"] = [global_step] * data_cnt
    columns_config.append(["global_step", "bigint"])

    for checker, func in DUMPING_FUNC:
        if checker(path):
            p = multiprocessing.Process(target=func, args=(path, write_data, columns_config), daemon=False)
            p.start()


def compute_segment_masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    对每段连续的1分别计算 masked_mean，不连续的段不相乘。

    Args:
        tensor: [batch_size, seq_len] 要计算的值
        mask: [batch_size, seq_len] mask，1表示有效位置，0表示无效位置

    Returns:
        [batch_size, seq_len] 结果，每段连续的1位置填充该段的 masked_mean
    """
    batch_size, seq_len = mask.shape
    device = mask.device
    result = torch.zeros_like(tensor)

    # 对每个样本分别处理
    for b in range(batch_size):
        sample_mask = mask[b]  # [seq_len]
        sample_tensor = tensor[b]  # [seq_len]

        # 找到所有连续的1的段
        # 使用 diff 找到边界：1->0 和 0->1 的位置
        diff = torch.diff(sample_mask, prepend=torch.tensor([0], device=device))
        # 找到段的开始位置（0->1）
        segment_starts = torch.where(diff == 1)[0]
        # 找到段的结束位置（1->0），diff[i]==-1 表示 mask[i-1]==1 且 mask[i]==0，所以段的结束位置是 i（不包括i）
        segment_ends = torch.where(diff == -1)[0]

        # 如果最后一个位置是1，需要添加结束位置
        if sample_mask[-1] == 1:
            segment_ends = torch.cat([segment_ends, torch.tensor([seq_len], device=device)])

        # 确保 segment_starts 和 segment_ends 长度匹配
        if len(segment_starts) != len(segment_ends):
            # 如果长度不匹配，只处理能匹配的部分
            min_len = min(len(segment_starts), len(segment_ends))
            segment_starts = segment_starts[:min_len]
            segment_ends = segment_ends[:min_len]

        # 对每段分别计算 masked_mean
        for start, end in zip(segment_starts, segment_ends):
            # 获取这段的索引
            segment_indices = torch.arange(start, end, device=device)
            segment_mask = sample_mask[segment_indices]  # 这段的mask
            segment_tensor = sample_tensor[segment_indices]  # 这段的值

            if segment_mask.sum() > 0:
                # 计算这段的 masked_mean（只考虑mask为1的位置）
                segment_mean = (segment_tensor * segment_mask).sum() / (segment_mask.sum() + 1e-8)
                # 将结果填充到这段内mask为1的位置
                result[b, segment_indices] = segment_mean * segment_mask

    return result


def compute_agentic_reinforce_return(
    token_level_rewards: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor, mask: Optional[torch.Tensor] = None
):
    """
    计算 REINFORCE 的 return，支持按 mask 分段 discount 衰减。
    每段内所有位置获得相同的折扣累积值（从该段最后位置开始累积）。

    Args:
        token_level_rewards: [batch_size, seq_len] token 级别的奖励
        gamma: discount factor
        lambd: lambda 参数（当前未使用，保留以兼容接口）
        mask: [batch_size, seq_len] mask，1表示有效位置，0表示无效位置。如果为None，则对所有位置计算

    Returns:
        advantages: [batch_size, seq_len] advantages
        returns: [batch_size, seq_len] returns
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        device = token_level_rewards.device
        returns = torch.zeros_like(token_level_rewards, dtype=torch.float32)

        # 如果没有提供 mask，则对所有位置计算（向后兼容）
        if mask is None:
            mask = torch.ones_like(token_level_rewards)

        # 确保 gamma 是标量
        gamma_val = gamma.item() if torch.is_tensor(gamma) else gamma

        # 对每个样本分别处理
        for b in range(batch_size):
            sample_mask = mask[b]  # [seq_len]
            sample_rewards = token_level_rewards[b]  # [seq_len]

            # 找到所有连续的1的段
            # 使用 diff 找到边界：1->0 和 0->1 的位置
            diff = torch.diff(sample_mask.float(), prepend=torch.tensor([0.0], device=device))

            # 找到段的开始位置（0->1，diff==1）
            segment_starts = torch.where(diff == 1)[0]

            # 找到段的结束位置（1->0，diff==-1）
            segment_ends = torch.where(diff == -1)[0]

            # 如果最后一个位置是1，需要添加结束位置
            if len(sample_mask) > 0 and sample_mask[-1] == 1:
                segment_ends = torch.cat([segment_ends, torch.tensor([gen_len], device=device)])

            # 计算该段从最后位置开始的累积折扣奖励
            cumulative_return = 0.0
            # 对每段分别计算 discounted return
            for start, end in zip(segment_starts.flip(-1), segment_ends.flip(-1)):
                start_idx = start.item()
                end_idx = end.item()
                segment_len = end_idx - start_idx

                cumulative_return = sample_rewards[end_idx - 1].item() + gamma_val * cumulative_return

                # 该段内所有位置都设置为这个累积值
                returns[b, start_idx:end_idx] = cumulative_return

        advantages = returns

    return advantages, returns


@torch.no_grad()
def agentic_compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    response_mask=None,
    pipeline_config=None,
):
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]
    if response_mask.sum() == 0:
        whiten_rewards = False
        whiten_advantages = False
        logger.info("Warning: domain final_response_mask.sum() == 0! All masked_whiten will be skipped.")

    # Check OPD config
    is_pure_opd = getattr(pipeline_config, "is_pure_opd", False) if pipeline_config else False
    use_opd = getattr(pipeline_config, "use_opd", False) if pipeline_config else False
    opd_kl_coef = getattr(pipeline_config, "opd_kl_coef", 1.0) if pipeline_config else 1.0

    # Compute KL divergence for OPD modes
    kld = None
    if is_pure_opd or use_opd:
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"] if getattr(pipeline_config, "enable_old_logprobs_recompute", False) else data.batch["infer_logprobs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=getattr(pipeline_config, "kl_penalty", "kl"),
        )

    # For pure OPD mode, advantage is directly -kld
    if is_pure_opd:
        advantages = -kld
        returns = advantages
        data.batch["raw_advantages"] = advantages
    else:
        token_level_rewards = data.batch["token_level_rewards"].float()
        if whiten_rewards:
            token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
        token_level_rewards = token_level_rewards * response_mask
        data.batch["token_level_rewards"] = token_level_rewards
        if adv_estimator == "gae":
            values = data.batch["values"].float()
            data.batch["values"] = values * response_mask
            advantages, returns = compute_gae_advantage_return(
                token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
            )
        elif adv_estimator in ["reinforce", "grpo", "gigpo", "step_reinforce"]:
            advantages, returns = compute_reinforce_return(
                token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
            )
        elif adv_estimator in ["agentic_reinforce"]:
            advantages, returns = compute_agentic_reinforce_return(
                token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd, mask=response_mask
            )
        else:
            raise NotImplementedError

        data.batch["raw_advantages"] = advantages

        # Apply mixed OPD mode
        if use_opd:
            advantages = advantages - opd_kl_coef * kld

    if whiten_advantages:
        # TODO whiten过程中是否要考虑response的长度？
        advantages = masked_whiten(values=advantages, mask=response_mask)
    advantages = advantages * response_mask

    if advantage_clip is not None:
        adv_clip_frac = compute_clip_fraction(values=advantages, clip_min=-advantage_clip, clip_max=advantage_clip)
        data.meta_info["metrics"] = {"critic/advantage_clip_frac": adv_clip_frac}
        advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data
