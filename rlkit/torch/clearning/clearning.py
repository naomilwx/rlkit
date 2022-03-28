from collections import OrderedDict
from email import policy

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

class CLearningTrainer(TorchTrainer):
    def __init__(
            self,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            target_policy,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,
            batch_size=128,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy
        self.target_policy = target_policy
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_learning_rate,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.batch_size = batch_size

    def train_from_torch(self, batch):
        """
        Critic operations.
        """

        # next_actions = self.target_policy(next_obs)
        # noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        # noise = torch.clamp(
        #     noise,
        #     -self.target_policy_noise_clip,
        #     self.target_policy_noise_clip
        # )
        # noisy_next_actions = next_actions + noise

        # target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        # target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        # target_q_values = torch.min(target_q1_values, target_q2_values)
        # q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        # q_target = q_target.detach()

        # q1_pred = self.qf1(obs, actions)
        # bellman_errors_1 = (q1_pred - q_target) ** 2
        # qf1_loss = bellman_errors_1.mean()

        # q2_pred = self.qf2(obs, actions)
        # bellman_errors_2 = (q2_pred - q_target) ** 2
        # qf2_loss = bellman_errors_2.mean()
        critic_loss = self.critic_loss(batch)

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        critic_loss.backward()
        self.qf2_optimizer.step()

        # policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_loss = self.policy_loss(batch)
            # policy_actions = self.policy(obs)
            # q_output = self.qf1(obs, policy_actions)
            # policy_loss = - q_output.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['Critic Loss'] = np.mean(ptu.get_numpy(critic_loss))
            # self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Bellman Errors 1',
            #     ptu.get_numpy(bellman_errors_1),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Bellman Errors 2',
            #     ptu.get_numpy(bellman_errors_2),
            # ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            trained_policy=self.policy,
            target_policy=self.target_policy,
        )

    def critic_loss(self, batch,
        w_clipping=20.0,
        relabel_next_prob=0.5,
        relabel_future_prob=0.0):
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']

        if w_clipping is None:
            w_clipping = 1 / (1 - self.reward_scale)
        
        next_actions = self.target_policy(next_obs)    
        target_q1_values = self.target_qf1(next_obs, next_actions)
        target_q2_values = self.target_qf2(next_obs, next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)

        w = target_q_values / (1 - target_q_values)
        w = w.detach()
        if w_clipping >= 0:
            w = torch.clamp(w, 0, w_clipping)
        
        half_batch = self.batch_size // 2
        float_batch_size = float(self.batch_size)
        num_next = int(np.round(float_batch_size * relabel_next_prob))
        num_future = int(np.round(float_batch_size * relabel_future_prob))

        # weights = torch.cat([torch.full((num_next,), (1 - self.reward_scale)),
        #                     torch.full((num_future,), 1.0),
        #                     (1 + self.reward_scale * w)[half_batch:]],
        #                     axis=0)

        y = self.reward_scale * w / (1 + self.reward_scale * w)
        td_targets = rewards + (1 - rewards) * y
        td_targets = td_targets.detach()
        if relabel_future_prob > 0:
            td_targets = np.concat([np.ones(half_batch),
                                td_targets[half_batch:]], axis=0)

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        qf1_loss = self.qf_criterion(q1_pred, td_targets)
        qf2_loss = self.qf_criterion(q2_pred, td_targets)
        critic_loss = qf1_loss + qf2_loss
        
        if len(critic_loss.shape) > 1:
            # Sum over the time dimension.
            critic_loss = torch.sum(critic_loss, dim=(1, len(critic_loss.shape) - 1))

        # TODO
        # agg_loss = common.aggregate_losses(
        #   per_example_loss=critic_loss,
        #   sample_weight=weights,
        #   regularization_loss=(self.qf1.losses +
        #                        self.qf1.losses))
        # critic_loss = agg_loss.total_loss

        if self._need_to_update_eval_statistics:
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(td_targets),
            ))

        return critic_loss
    
    def policy_loss(self, batch):
        obs = batch['observations']
        
        sampled_actions = self.policy(obs)

        target_q1_values = self.qf1(obs, sampled_actions)
        target_q2_values = self.qf2(obs, sampled_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        policy_loss = -1.0 * target_q_values

        policy_loss = torch.sum(policy_loss)
        return policy_loss