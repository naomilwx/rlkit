from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from multiworld.core.image_env import normalize_image
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.data import (
    ImageDataset,
    InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)
from rlkit.util.ml_util import ConstantSchedule
import collections
import time

# TODO: Fix type of train dataset
class VAETrainer(object):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            lr=None,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            use_linear_dynamics=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
    ):
        #TODO:steven fix pickling
        assert not use_parallel_dataloading, "Have to fix pickling the dataloaders first"

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        if skew_config is None:
            skew_config = {}
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        if is_auto_encoder:
            self.beta = 0
        if lr is None:
            if is_auto_encoder:
                lr = 1e-2
            else:
                lr = 1e-3
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None or is_auto_encoder:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot
        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params,
            lr=self.lr,
            weight_decay=weight_decay,
        )

        self.batch_size = batch_size
        self.use_parallel_dataloading = use_parallel_dataloading
        self.train_data_workers = train_data_workers
        self.skew_dataset = skew_dataset
        self.skew_config = skew_config
        self.start_skew_epoch = start_skew_epoch
        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs
        
        if self.skew_dataset:
            self._train_weights = self._compute_train_weights()
        else:
            self._train_weights = None

        if use_parallel_dataloading:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                sampler=InfiniteRandomSampler(self.train_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=train_data_workers,
                pin_memory=True,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                sampler=InfiniteRandomSampler(self.test_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            self.train_dataloader = iter(self.train_dataloader)
            self.test_dataloader = iter(self.test_dataloader)

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )
        self.linearity_weight = linearity_weight
        self.distance_weight = distance_weight
        self.loss_weights = loss_weights

        self.use_linear_dynamics = use_linear_dynamics
        self._extra_stats_to_log = None

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)

    @property
    def log_dir(self):
        return logger.get_snapshot_dir()
    
    # def update_train_weights(self):
    #     if self.skew_dataset:
    #         self._train_weights = self._compute_train_weights()
    #         if self.use_parallel_dataloading:
    #             self.train_dataloader = DataLoader(
    #                 self.train_dataset,
    #                 sampler=InfiniteWeightedRandomSampler(self.train_dataset, self._train_weights),
    #                 batch_size=self.batch_size,
    #                 drop_last=False,
    #                 num_workers=self.train_data_workers,
    #                 pin_memory=True,
    #             )
    #             self.train_dataloader = iter(self.train_dataloader)
    
    def _compute_train_weights(self):
        # TODO: fix this
        method = self.skew_config.get('method', 'squared_error')
        power = self.skew_config.get('power', -1)
        batch_size = 512
        size = self.train_dataset.shape[0]
        next_idx = min(batch_size, size)
        cur_idx = 0
        weights = np.zeros(size)
        while cur_idx < self.train_dataset.shape[0]:
            idxs = np.arange(cur_idx, next_idx)
            data = self.train_dataset[idxs, :]
            if method == 'vae_prob':
                data = normalize_image(data)
                weights[idxs] = compute_p_x_np_to_np(self.model, data, power=power, **self.priority_function_kwargs)
            else:
                raise NotImplementedError('Method {} not supported'.format(method))
            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, size)

        if method == 'vae_prob':
            weights = relative_probs_from_log_probs(weights)
        return weights

    def get_dataset_stats(self, data):
        torch_input = ptu.from_numpy(normalize_image(data))
        mus, log_vars = self.model.encode(torch_input)
        mus = ptu.get_numpy(mus)
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def _kl_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
        mu, log_var = self.model.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def _reconstruction_squared_error_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
        recons, *_ = self.model(torch_input)
        error = torch_input - recons
        return ptu.get_numpy((error ** 2).sum(dim=1))

    def set_vae(self, vae):
        self.model = vae
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_batch(self, train=True, epoch=None):
        # Not actually used except in debug_statistics
        if self.use_parallel_dataloading:
            if not train:
                dataloader = self.test_dataloader
            else:
                dataloader = self.train_dataloader
            samples = next(dataloader).to(ptu.device)
            return samples

        dataset = self.train_dataset if train else self.test_dataset
        skew = False
        if epoch is not None:
            skew = (self.start_skew_epoch < epoch)
        if train and self.skew_dataset and skew:
            probs = self._train_weights / np.sum(self._train_weights)
            ind = np.random.choice(
                len(probs),
                self.batch_size,
                p=probs,
            )
        else:
            ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = normalize_image(dataset[ind, :])
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            samples = samples - self.train_data_mean
        return ptu.from_numpy(samples)

    def get_debug_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return ptu.from_numpy(X), ptu.from_numpy(Y)

    def train_epoch(self, epoch, dataset, batches=100):
        start_time = time.time()
        for b in range(batches):
            self.train_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["train/epoch_duration"].append(time.time() - start_time)
        # self.update_train_weights()

    def test_epoch(self, epoch, dataset, batches=10):
        start_time = time.time()
        for b in range(batches):
            self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    def compute_loss(self, epoch, batch, test=False):
        prefix = "test/" if test else "train/"

        beta = float(self.beta_schedule.get_value(epoch))
        if "x_t" in batch:
            obs = batch["x_t"]
        else:
            obs = batch["observations"]
        reconstructions, obs_distribution_params, latent_distribution_params = self.model(obs)
        log_prob = self.model.logprob(obs, obs_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)
        loss = -1 * log_prob + beta * kle

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (obs, reconstructions)

        return loss

    def train_batch(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(epoch, batch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(epoch, batch, True)

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats

    def dump_scatterplot(self, z, epoch):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.log(__file__ + ": Unable to load matplotlib. Consider "
                                  "setting do_scatterplot to False")
            return
        dim_and_stds = [(i, np.std(z[:, i])) for i in range(z.shape[1])]
        dim_and_stds = sorted(
            dim_and_stds,
            key=lambda x: x[1]
        )
        dim1 = dim_and_stds[-1][0]
        dim2 = dim_and_stds[-2][0]
        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, dim1], z[:, dim2], marker='o', edgecolor='none')
        if self.model.dist_mu is not None:
            x1 = self.model.dist_mu[dim1:dim1 + 1]
            y1 = self.model.dist_mu[dim2:dim2 + 1]
            x2 = (
                    self.model.dist_mu[dim1:dim1 + 1]
                    + self.model.dist_std[dim1:dim1 + 1]
            )
            y2 = (
                    self.model.dist_mu[dim2:dim2 + 1]
                    + self.model.dist_std[dim2:dim2 + 1]
            )
        plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
        axes = plt.gca()
        axes.set_xlim([-6, 6])
        axes.set_ylim([-6, 6])
        axes.set_title('dim {} vs dim {}'.format(dim1, dim2))
        plt.grid(True)
        save_file = osp.join(self.log_dir, 'scatter%d.png' % epoch)
        plt.savefig(save_file)

class ConvVAETrainer(VAETrainer):
    def dump_reconstructions(self, epoch):
        obs, reconstructions = self.eval_data["test/last_batch"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1, self.input_channels, self.imsize, self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                self.input_channels,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def debug_statistics(self):
        """
        Given an image $$x$$, samples a bunch of latents from the prior
        $$z_i$$ and decode them $$\hat x_i$$.
        Compare this to $$\hat x$$, the reconstruction of $$x$$.
        Ideally
         - All the $$\hat x_i$$s do worse than $$\hat x$$ (makes sure VAE
           isn???t ignoring the latent)
         - Some $$\hat x_i$$ do better than other $$\hat x_i$$ (tests for
           coverage)
        """
        debug_batch_size = 64
        data = self.get_batch(train=False)
        reconstructions, _, _ = self.model(data)
        img = data[0]
        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, self.representation_size)
        random_imgs, _ = self.model.decode(samples)
        random_mses = (random_imgs - img_repeated) ** 2
        mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)
        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            ptu.get_numpy(random_mses),
        ))
        stats['debug/MSE of reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        return stats

    def dump_samples(self, epoch):
        self.model.eval()
        sample = ptu.randn(64, self.representation_size)
        sample = self.model.decode(sample)[0].cpu()
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(
            sample.data.view(64, self.input_channels, self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )

    '''
    SkewFit Debug Stats
    '''

    def dump_sampling_histogram(self, epoch):
        import matplotlib.pyplot as plt
        if self._train_weights is None:
            self._train_weights = self._compute_train_weights()
        weights = torch.from_numpy(self._train_weights)
        samples = ptu.get_numpy(torch.multinomial(
            weights, len(weights), replacement=True
        ))
        plt.clf()
        n, bins, patches = plt.hist(samples, bins=np.arange(0, len(weights), 1))
        plt.xlabel('Indices')
        plt.ylabel('Number of Samples')
        plt.title('VAE Priority Histogram')
        save_file = osp.join(self.log_dir, 'hist{}.png'.format(
            epoch))
        plt.savefig(save_file)

        samples = ptu.get_numpy(torch.multinomial(
            weights, self.batch_size, replacement=True
        ))
        plt.clf()
        n, bins, patches = plt.hist(samples, bins=np.arange(0, len(weights), 1))
        plt.xlabel('Indices')
        plt.ylabel('Number of Samples')
        plt.title('VAE Priority Histogram Batch')
        save_file = osp.join(self.log_dir, 'hist_batch{}.png'.format(
            epoch))
        plt.savefig(save_file)

    def dump_best_reconstruction(self, epoch, num_shown=10):
        idx_and_weights = self._get_sorted_idx_and_train_weights()
        idxs = [i for i, _ in idx_and_weights[:num_shown]]
        self._dump_imgs_and_reconstructions(idxs, 'best{}.png'.format(epoch))

    def dump_worst_reconstruction(self, epoch, num_shown=10):
        idx_and_weights = self._get_sorted_idx_and_train_weights()
        idx_and_weights = idx_and_weights[::-1]
        idxs = [i for i, _ in idx_and_weights[:num_shown]]
        self._dump_imgs_and_reconstructions(idxs, 'worst{}.png'.format(epoch))

    def _dump_imgs_and_reconstructions(self, idxs, filename):
        imgs = []
        recons = []
        for i in idxs:
            img_np = self.train_dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(self.log_dir, filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=len(idxs),
        )

    def log_loss_under_uniform(self, model, data, priority_function_kwargs):
        import torch.nn.functional as F
        log_probs_prior = []
        log_probs_biased = []
        log_probs_importance = []
        kles = []
        mses = []
        for i in range(0, data.shape[0], self.batch_size):
            img = normalize_image(data[i:min(data.shape[0], i + self.batch_size), :])
            torch_img = ptu.from_numpy(img)
            reconstructions, obs_distribution_params, latent_distribution_params = self.model(torch_img)

            priority_function_kwargs['sampling_method'] = 'true_prior_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_prior = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'biased_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_biased = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'importance_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_importance = (log_p - log_q + log_d).mean()

            kle = model.kl_divergence(latent_distribution_params)
            mse = F.mse_loss(torch_img, reconstructions, reduction='elementwise_mean')
            mses.append(mse.item())
            kles.append(kle.item())
            log_probs_prior.append(log_prob_prior.item())
            log_probs_biased.append(log_prob_biased.item())
            log_probs_importance.append(log_prob_importance.item())

        logger.record_tabular("Uniform Data Log Prob (True Prior)", np.mean(log_probs_prior))
        logger.record_tabular("Uniform Data Log Prob (Biased)", np.mean(log_probs_biased))
        logger.record_tabular("Uniform Data Log Prob (Importance)", np.mean(log_probs_importance))
        logger.record_tabular("Uniform Data KL", np.mean(kles))
        logger.record_tabular("Uniform Data MSE", np.mean(mses))

    def dump_uniform_imgs_and_reconstructions(self, dataset, epoch):
        idxs = np.random.choice(range(dataset.shape[0]), 4)
        filename = 'uniform{}.png'.format(epoch)
        imgs = []
        recons = []
        for i in idxs:
            img_np = dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(self.log_dir, filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=4,
        )

    def _get_sorted_idx_and_train_weights(self):
        if self._train_weights is None:
            self._train_weights = self._compute_train_weights()
        idx_and_weights = zip(range(len(self._train_weights)),
                              self._train_weights)
        return sorted(idx_and_weights, key=lambda x: x[1])

class ConvVAEGradientPenaltyTrainer(ConvVAETrainer):
    def compute_loss(self, epoch, batch, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        obs = batch["observations"]
        reconstructions, obs_distribution_params, latent_distribution_params = self.model(obs)
        log_prob = self.model.logprob(obs, obs_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)

        # import pdb; pdb.set_trace()
        output = torch.sum(reconstructions)
        grads = torch.autograd.grad([output], latent_distribution_params[0:1], retain_graph=True)
        gradient_norm = torch.mean(grads[0].norm(dim=1))
        gradient_norm_weight = self.loss_weights["gradient_norm"]

        loss = -1 * log_prob + beta * kle + gradient_norm_weight * gradient_norm # + self.linearity_weight * linear_dynamics_loss + self.distance_weight * state_distance_loss

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())
        self.eval_statistics[prefix + "gradient_norm"].append(gradient_norm.item())
        # self.eval_statistics[prefix + "dynamics_loss"].append(linear_dynamics_loss.item())
        # self.eval_statistics[prefix + "distance_loss"].append(state_distance_loss.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (obs, reconstructions)

        return loss

    def train_batch(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(epoch, batch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(epoch, batch, True)


def compute_p_x_np_to_np(
    model,
    data,
    power,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    # assert data.dtype == np.float64, 'images should be normalized'
    assert power >= -1 and power <= 0, 'power for skew-fit should belong to [-1, 0]'

    log_p, log_q, log_d = compute_log_p_log_q_log_d(
        model,
        data,
        decoder_distribution,
        num_latents_to_sample,
        sampling_method
    )

    if sampling_method == 'importance_sampling':
        log_p_x = (log_p - log_q + log_d).mean(dim=1)
    elif sampling_method == 'biased_sampling' or sampling_method == 'true_prior_sampling':
        log_p_x = log_d.mean(dim=1)
    else:
        raise EnvironmentError('Invalid Sampling Method Provided')
    log_p_x_skewed = power * log_p_x
    return ptu.get_numpy(log_p_x_skewed)

def relative_probs_from_log_probs(log_probs):
    """
    Returns relative probability from the log probabilities. They're not exactly
    equal to the probability, but relative scalings between them are all maintained.

    For correctness, all log_probs must be passed in at the same time.
    """
    probs = np.exp((log_probs - log_probs.mean())/log_probs.std())
    print('log probs', np.min(log_probs), np.max(log_probs), log_probs.mean(), log_probs.std())
    print(np.min(probs),np.max(probs))

    assert not np.any(probs <= 0), 'choose a smaller power'

    return probs


def compute_log_p_log_q_log_d(
    model,
    data,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    # assert data.dtype == np.float64, 'images should be normalized'
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs)
    batch_size = data.shape[0]
    representation_size = model.representation_size
    log_p, log_q, log_d = ptu.zeros((batch_size, num_latents_to_sample)), ptu.zeros(
        (batch_size, num_latents_to_sample)), ptu.zeros((batch_size, num_latents_to_sample))
    true_prior = Normal(ptu.zeros((batch_size, representation_size)),
                        ptu.ones((batch_size, representation_size)))
    mus, logvars = latent_distribution_params
    for i in range(num_latents_to_sample):
        if sampling_method == 'importance_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'biased_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'true_prior_sampling':
            latents = true_prior.rsample()
        else:
            raise EnvironmentError('Invalid Sampling Method Provided')

        stds = logvars.exp().pow(.5)
        vae_dist = Normal(mus, stds)
        log_p_z = true_prior.log_prob(latents).sum(dim=1)
        log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=1)
        if decoder_distribution == 'bernoulli':
            decoded = model.decode(latents)[0]
            log_d_x_given_z = torch.log(imgs * decoded + (1 - imgs) * (1 - decoded) + 1e-8).sum(dim=1)
        elif decoder_distribution == 'gaussian_identity_variance':
            _, obs_distribution_params = model.decode(latents)
            dec_mu, dec_logvar = obs_distribution_params
            dec_var = dec_logvar.exp()
            decoder_dist = Normal(dec_mu, dec_var.pow(.5))
            log_d_x_given_z = decoder_dist.log_prob(imgs).sum(dim=1)
        else:
            raise EnvironmentError('Invalid Decoder Distribution Provided')

        log_p[:, i] = log_p_z
        log_q[:, i] = log_q_z_given_x
        log_d[:, i] = log_d_x_given_z
    return log_p, log_q, log_d
