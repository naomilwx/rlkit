import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers
from rlkit.launchers.cvae_experiments import grill_her_td3_offpolicy_online_vae_full_experiment
from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerTwoObjectEnv
from rlkit.torch.vae.conditional_conv_vae import DeltaCVAE
from rlkit.torch.vae.conditional_vae_trainer import DeltaCVAETrainer
from rlkit.data_management.online_conditional_vae_replay_buffer import \
        OnlineConditionalVaeRelabelingBuffer

from examples.ccrig.env import default_multiobj_env
env, args = default_multiobj_env()

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerReachXYEnv-v1',
        # env_class=env,
        # env_kwargs=args,
        grill_variant=dict(
            save_video=False,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=0,
            ),
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=50,
            algo_kwargs=dict(
                batch_size=128,
                num_epochs=350,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=1000, # was 4000
                vae_training_schedule=vae_schedules.never_train,
                oracle_data=False,
                vae_save_period=25,
                parallel_vae_train=False,
                dataset_path=None,
                rl_offpolicy_num_training_steps=0,
            ),
            td3_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1.0,
                # tau=1e-2,
                policy_learning_rate=1e-4
            ),
            replay_buffer_class=OnlineConditionalVaeRelabelingBuffer,
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode='vae_prior',
            ),
            exploration_goal_sampling_mode='vae_prior',
            evaluation_goal_sampling_mode='reset_of_env',
            normalize=False,
            render=False,
            exploration_type='epgaussian',
            exploration_noise=0.3,
            exploration_random_prob=0.3,
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                epsilon=0.05,
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            vae_trainer_class=DeltaCVAETrainer,
        ),
        train_vae_variant=dict(
            latent_sizes=(4, 4),
            beta=10,
            beta_schedule_kwargs=dict(
                x_values=(0, 1000),
                y_values=(1, 100),
            ),
            context_schedule=1,
            num_epochs=300, # was 1500
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                N=10000,
                n_random_steps=10,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
                tag="ccrig_tuning_orig_network",
            ),
            vae_trainer_class=DeltaCVAETrainer,
            vae_class=DeltaCVAE,
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture_with_more_hidden_layers,
                decoder_distribution='gaussian_identity_variance',
            ),

            algo_kwargs=dict(
                start_skew_epoch=100,
                is_auto_encoder=False,
                batch_size=32, # was 128
                lr=1e-3, #1E-4
                skew_config=dict(
                    method='vae_prob',
                    power=-1,
                ),
                weight_decay=1e-3,
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=25,
        ),

        logger_variant=dict(
            tensorboard=True,
        ),
    )

    # search_space = {
    #     'train_vae_variant.latent_sizes': [(6, 4),],
    #     'train_vae_variant.context_schedule': [
    #         1.0,
    #     ],
    #     'train_vae_variant.beta_schedule_kwargs': [
    #         dict(x_values=(0, 1500,), y_values=(1, 50)),
    #     ],
    #     'train_vae_variant.algo_kwargs.batch_size': [128, ],
    #     'train_vae_variant.algo_kwargs.lr': [1e-4, ],
    #     'train_vae_variant.algo_kwargs.weight_decay': [1e-4, ],
    #     'grill_variant.algo_kwargs.num_trains_per_train_loop':[1000,],
    #     'grill_variant.algo_kwargs.batch_size': [128,],
    #     'grill_variant.exploration_noise': [0.3],
    # }
    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'ccrig-reach'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_offpolicy_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True
          )
