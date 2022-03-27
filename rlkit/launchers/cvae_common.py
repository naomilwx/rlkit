import os.path as osp
import time

import cv2
import numpy as np

from torch.utils import data

# from rlkit.samplers.data_collector import VAEWrappedEnvPathCollector
# from rlkit.torch.her.her import HERTrainer
# from rlkit.torch.sac.policies import MakeDeterministic
# from rlkit.torch.sac.sac import SACTrainer
# from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm

# from rlkit.torch.grill.video_gen import VideoSaveFunction

def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        grill_variant['env_id'] = env_id
        train_vae_variant['generate_vae_dataset_kwargs']['env_id'] = env_id
    else:
        env_class = variant['env_class']
        env_kwargs = variant['env_kwargs']
        train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = (
            env_class
        )
        train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = (
            env_kwargs
        )
        grill_variant['env_class'] = env_class
        grill_variant['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = (
        init_camera
    )
    train_vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['imsize'] = imsize
    grill_variant['imsize'] = imsize
    grill_variant['init_camera'] = init_camera


def train_vae_and_update_variant(variant):
    from rlkit.core import logger
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(train_vae_variant,
                                                       return_data=True)
        if grill_variant.get('save_vae_data', False):
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        grill_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if grill_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs']
            )
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data


def train_vae(variant, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
        ConvDynamicsVAE,
        SpatialAutoEncoder,
        AutoEncoder,
    )
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch
    import gym
    print('running train_vae')
    beta = variant["beta"]
    representation_size = variant.get("representation_size", variant.get("latent_sizes", None))
    use_linear_dynamics = variant.get('use_linear_dynamics', False)
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn', generate_vae_dataset)
    variant['generate_vae_dataset_kwargs']['use_linear_dynamics'] = use_linear_dynamics
    variant['generate_vae_dataset_kwargs']['batch_size'] = variant['algo_kwargs']['batch_size']
    print('running generate_vae_dataset_fctn')
    train_dataset, test_dataset, info = generate_vae_dataset_fctn(variant['generate_vae_dataset_kwargs'])

    if use_linear_dynamics:
        action_dim = train_dataset.data['actions'].shape[2]

    logger.save_extra_data(info)
    print('done saving')
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    if 'context_schedule' in variant:
        schedule = variant['context_schedule']
        if type(schedule) is dict:
            context_schedule = PiecewiseLinearSchedule(**schedule)
        else:
            context_schedule = ConstantSchedule(schedule)
        variant['algo_kwargs']['context_schedule'] = context_schedule
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    if variant['algo_kwargs'].get('is_auto_encoder', False):
        model = AutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    elif variant.get('use_spatial_auto_encoder', False):
        model = SpatialAutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    else:
        vae_class = variant.get('vae_class', ConvVAE)
        if use_linear_dynamics:
            model = vae_class(representation_size, decoder_output_activation=decoder_activation, action_dim=action_dim,**variant['vae_kwargs'])
        else:
            model = vae_class(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    model.to(ptu.device)

    vae_trainer_class = variant.get('vae_trainer_class', ConvVAETrainer)
    trainer = vae_trainer_class(model, beta=beta,
                       beta_schedule=beta_schedule,
                       **variant['algo_kwargs'])
    save_period = variant['save_period']

    dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
    print('vae num epoch', variant['num_epochs'])
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        trainer.train_epoch(epoch, train_dataset)
        trainer.test_epoch(epoch, test_dataset)

        if should_save_imgs:
            trainer.dump_reconstructions(epoch)
            trainer.dump_samples(epoch)
            if dump_skew_debug_plots:
                trainer.dump_best_reconstruction(epoch)
                trainer.dump_worst_reconstruction(epoch)
                trainer.dump_sampling_histogram(epoch)

        stats = trainer.get_diagnostics()
        for k, v in stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        trainer.end_epoch(epoch)

        if epoch % 50 == 0:
            logger.save_itr_params(epoch, model)
    logger.save_extra_data(model, 'vae.pkl', mode='pickle')
    if return_data:
        return model, train_dataset, test_dataset
    print('end of vae_train')
    return model



def generate_vae_dataset(variant):
    print(variant)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs',None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)
    batch_size = variant.get('batch_size', 128)
    test_p = variant.get('test_p', 0.9)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    dataset_path = variant.get('dataset_path', None)
    oracle_dataset_using_set_to_goal = variant.get('oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_rollout_data_set_to_goal = variant.get('random_rollout_data_set_to_goal', True)
    random_and_oracle_policy_data=variant.get('random_and_oracle_policy_data', False)
    random_and_oracle_policy_data_split=variant.get('random_and_oracle_policy_data_split', 0)
    policy_file = variant.get('policy_file', None)
    n_random_steps = variant.get('n_random_steps', 100)
    vae_dataset_specific_env_kwargs = variant.get('vae_dataset_specific_env_kwargs', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    non_presampled_goal_img_is_garbage = variant.get('non_presampled_goal_img_is_garbage', None)

    conditional_vae_dataset = variant.get('conditional_vae_dataset', False)
    use_env_labels = variant.get('use_env_labels', False)
    use_linear_dynamics = variant.get('use_linear_dynamics', False)
    save_trajectories = variant.get('save_trajectories', False)
    save_trajectories = save_trajectories or use_linear_dynamics or conditional_vae_dataset

    tag = variant.get('tag', '')

    # assert N % n_random_steps == 0, "Fix N/horizon or dataset generation will fail"

    from multiworld.core.image_env import ImageEnv, unormalize_image
    import rlkit.torch.pytorch_util as ptu
    from rlkit.util.io import load_local_or_remote_file
    from rlkit.data_management.dataset  import (
        TrajectoryDataset, ImageObservationDataset, InitialObservationDataset,
        EnvironmentDataset, ConditionalDynamicsDataset, InitialObservationNumpyDataset,
        InfiniteBatchLoader,
    )

    info = {}
    if dataset_path is not None:
        dataset = load_local_or_remote_file(dataset_path)
        dataset = dataset.item()
        N = dataset['observations'].shape[0] * dataset['observations'].shape[1]
        n_random_steps = dataset['observations'].shape[1]
    else:
        print('loading data')
        if env_kwargs is None:
            env_kwargs = {}
        if save_file_prefix is None:
            save_file_prefix = env_id
        if save_file_prefix is None:
            save_file_prefix = env_class.__name__
        filename = "/tmp/{}_N{}_{}_imsize{}_random_oracle_split_{}{}.npy".format(
            save_file_prefix,
            str(N),
            init_camera.__name__ if init_camera and hasattr(init_camera, '__name__') else '',
            imsize,
            random_and_oracle_policy_data_split,
            tag,
        )

        if use_cached and osp.isfile(filename):
            dataset = np.load(filename)
            if conditional_vae_dataset:
                dataset = dataset.item()
            print("loaded data from saved file", filename)
        else:
            now = time.time()

            if env_id is not None:
                import gym
                import multiworld
                multiworld.register_all_envs()
                env = gym.make(env_id)
            else:
                if vae_dataset_specific_env_kwargs is None:
                    vae_dataset_specific_env_kwargs = {}
                for key, val in env_kwargs.items():
                    if key not in vae_dataset_specific_env_kwargs:
                        vae_dataset_specific_env_kwargs[key] = val
                env = env_class(**vae_dataset_specific_env_kwargs)

            if not isinstance(env, ImageEnv):
                env = ImageEnv(
                    env,
                    imsize,
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                    non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
                )
            else:
                imsize = env.imsize
                env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
            env.reset()
            info['env'] = env
            if random_and_oracle_policy_data:
                policy_file = load_local_or_remote_file(policy_file)
                policy = policy_file['policy']
                policy.to(ptu.device)
            if random_rollout_data:
                from rlkit.exploration_strategies.ou_strategy import OUStrategy
                policy = OUStrategy(env.action_space)

            if save_trajectories:
                dataset = {
                    'observations': np.zeros((N // n_random_steps, n_random_steps, imsize * imsize * num_channels), dtype=np.uint8),
                    'actions': np.zeros((N // n_random_steps, n_random_steps, env.action_space.shape[0]), dtype=np.float),
                    'env': np.zeros((N // n_random_steps, imsize * imsize * num_channels), dtype=np.uint8),
                    }
            else:
                dataset = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
            labels = []
            for i in range(N):
                print(i, 'out of ', N)
                if random_and_oracle_policy_data:
                    num_random_steps = int(N*random_and_oracle_policy_data_split)
                    if i < num_random_steps:
                        env.reset()
                        for _ in range(n_random_steps):
                            obs = env.step(env.action_space.sample())[0]
                    else:
                        obs = env.reset()
                        policy.reset()
                        for _ in range(n_random_steps):
                            policy_obs = np.hstack((
                                obs['state_observation'],
                                obs['state_desired_goal'],
                            ))
                            action, _ = policy.get_action(policy_obs)
                            obs, _, _, _ = env.step(action)
                elif random_rollout_data: #ADD DATA WHERE JUST PUCK MOVES
                    if i % n_random_steps == 0:
                        env.reset()
                        policy.reset()
                        env_img = env._get_obs()['image_observation']
                        if random_rollout_data_set_to_goal:
                            env.set_to_goal(env.get_goal())
                    obs = env._get_obs()
                    u = policy.get_action_from_raw_action(env.action_space.sample())
                    env.step(u)
                elif oracle_dataset_using_set_to_goal:
                    goal = env.sample_goal()
                    env.set_to_goal(goal)
                    obs = env._get_obs()
                else:
                    env.reset()
                    for _ in range(n_random_steps):
                        obs = env.step(env.action_space.sample())[0]

                img = obs['image_observation']
                if use_env_labels:
                    labels.append(obs['label'])
                if save_trajectories:
                    dataset['observations'][i // n_random_steps, i % n_random_steps, :] = unormalize_image(img)
                    dataset['actions'][i // n_random_steps, i % n_random_steps, :] = u
                    dataset['env'][i // n_random_steps, :] = unormalize_image(env_img)
                else:
                    dataset[i, :] = unormalize_image(img)

                if show:
                    img = img.reshape(3, imsize, imsize).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    # radius = input('waiting...')
            print("done making training data", filename, time.time() - now)
            np.save(filename, dataset)
            np.save(filename[:-4] + 'labels.npy', np.array(labels))

    info['train_labels'] = []
    info['test_labels'] = []

    if use_linear_dynamics and conditional_vae_dataset:
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        train_dataset = ConditionalDynamicsDataset({
            'observations': dataset['observations'][:n, :, :],
            'actions': dataset['actions'][:n, :, :],
            'env': dataset['env'][:n, :]
        })
        test_dataset = ConditionalDynamicsDataset({
            'observations': dataset['observations'][n:, :, :],
            'actions': dataset['actions'][n:, :, :],
            'env': dataset['env'][n:, :]
        })

        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        indices = np.arange(num_trajectories)
        np.random.shuffle(indices)
        train_i, test_i = indices[:n], indices[n:]

        try:
            train_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][train_i, :, :],
                'actions': dataset['actions'][train_i, :, :],
                'env': dataset['env'][train_i, :]
            })
            test_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][test_i, :, :],
                'actions': dataset['actions'][test_i, :, :],
                'env': dataset['env'][test_i, :]
            })
        except:
            train_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][train_i, :, :],
                'actions': dataset['actions'][train_i, :, :],

            })
            test_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][test_i, :, :],
                'actions': dataset['actions'][test_i, :, :],
            })
    elif use_linear_dynamics:
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        train_dataset = TrajectoryDataset({
            'observations': dataset['observations'][:n, :, :],
            'actions': dataset['actions'][:n, :, :]
        })
        test_dataset = TrajectoryDataset({
            'observations': dataset['observations'][n:, :, :],
            'actions': dataset['actions'][n:, :, :]
        })
    elif conditional_vae_dataset:
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        indices = np.arange(num_trajectories)
        np.random.shuffle(indices)
        train_i, test_i = indices[:n], indices[n:]

        if 'env' in dataset:
            train_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][train_i, :, :],
                'env': dataset['env'][train_i, :]
            })
            test_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][test_i, :, :],
                'env': dataset['env'][test_i, :]
            })
        else:
            train_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][train_i, :, :],
            })
            test_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][test_i, :, :],
            })

        train_batch_loader_kwargs = variant.get(
            'train_batch_loader_kwargs',
            dict(batch_size=batch_size, num_workers=0, )
        )
        test_batch_loader_kwargs = variant.get(
            'test_batch_loader_kwargs',
            dict(batch_size=batch_size, num_workers=0, )
        )

        train_data_loader = data.DataLoader(train_dataset,
            shuffle=True, drop_last=True, **train_batch_loader_kwargs)
        test_data_loader = data.DataLoader(test_dataset,
            shuffle=True, drop_last=True, **test_batch_loader_kwargs)

        train_dataset = InfiniteBatchLoader(train_data_loader)
        test_dataset = InfiniteBatchLoader(test_data_loader)
    else:
        n = int(N * test_p)
        train_dataset = ImageObservationDataset(dataset[:n, :])
        test_dataset = ImageObservationDataset(dataset[n:, :])
    return train_dataset, test_dataset, info

def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrapper import VAEWrappedEnv, ConditionalVAEWrappedEnv
    from rlkit.util.io import load_local_or_remote_file
    from rlkit.torch.vae.conditional_conv_vae import CVAE, ConditionalConvVAE

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only', False)
    presampled_goals_path = variant.get('presampled_goals_path', None)
    vae = load_local_or_remote_file(vae_path) if type(vae_path) is str else vae_path
    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    env=vae_env,
                    env_id=variant.get('env_id', None),
                    **variant['goal_generation_kwargs']
                )
                del vae_env
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
            del image_env
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
                **variant.get('image_env_kwargs', {})
            )
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                presampled_goals = presampled_goals,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            print("Presampling all goals only")
        else:
            if isinstance(vae, CVAE) or isinstance(vae, ConditionalConvVAE):
                vae_env = ConditionalVAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            else:
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            if presample_image_goals_only:
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **variant['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Presampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env

    return env


def get_exploration_strategy(variant, env):
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy
    # from rlkit.exploration_strategies.noop import NoopStrategy

    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    # elif exploration_type == 'noop':
    #     es = NoopStrategy(
    #         action_space=env.action_space
    #     )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es


def grill_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_acheived_goal'


def get_state_experiment_video_save_function(rollout_function, env, policy, variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.core import logger
    from rlkit.envs.vae_wrapper import temporary_mode
    from rlkit.torch.skewfit import dump_video
    logdir = logger.get_snapshot_dir()
    save_period = variant.get('save_video_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    if do_state_exp:
        imsize = variant.get('imsize')
        dump_video_kwargs['imsize'] = imsize
        image_env = ImageEnv(
            env,
            imsize,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                dump_video(image_env, policy, filename, rollout_function,
                           **dump_video_kwargs)
    else:
        image_env = env
        dump_video_kwargs['imsize'] = env.imsize

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
                filename = osp.join(logdir,
                                    'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
    return save_video
