import os.path as osp
import time

#import cv2
import numpy as np

from rlkit.samplers.data_collector.path_collector import VAEWrappedEnvPathCollector
from rlkit.torch.her.her import HERTrainer

from rlkit.util.video import VideoSaveFunction

from rlkit.launchers.common import *

def train_vae(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)

def grill_her_td3_offpolicy_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_her_td3_experiment_offpolicy_online_vae(variant['grill_variant'])

def grill_her_td3_experiment_offpolicy_online_vae(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    import rlkit.samplers.rollout_functions as rf
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.td3.td3 import TD3Trainer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.vae.online_vae_offpolicy_algorithm import OnlineVaeOffpolicyAlgorithm

    import gc
    gc.collect() # Ashvin: this line for a GPU memory error

    grill_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset=uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset=None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )
    es = get_exploration_strategy(variant, env)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae

    replay_buffer_class = variant.get("replay_buffer_class", OnlineVaeRelabelingBuffer)
    replay_buffer = replay_buffer_class(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,

        **variant['replay_buffer_kwargs']
    )
    replay_buffer.representation_size = vae.representation_size

    vae_trainer_class = variant.get("vae_trainer_class", ConvVAETrainer)
    print('vae_trainer_class', vae_trainer_class.__name__)
    vae_trainer = vae_trainer_class(
        variant['vae_train_data'],
        variant['vae_test_data'],
        env.vae,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        env,
        expl_policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    algorithm = OnlineVaeOffpolicyAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)

    algorithm.pretrain()
    algorithm.train()
