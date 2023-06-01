# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import vcse

class ICM(nn.Module):
    """
    Same as ICM, with a trunk to save memory for KNN
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, icm_rep_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, icm_rep_dim),
                                   nn.LayerNorm(icm_rep_dim), nn.Tanh())

        self.forward_net = nn.Sequential(
            nn.Linear(icm_rep_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, icm_rep_dim))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * icm_rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        obs = self.trunk(obs)
        next_obs = self.trunk(next_obs)
        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error

    def get_rep(self, obs):
        rep = self.trunk(obs)
        return rep

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb , beta, knn_k,do_vcse):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.beta = beta
        self.knn_k = knn_k
        self.do_vcse = do_vcse

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.critic_extrinsic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_extrinsic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device).to(device)
        self.critic_extrinsic_target.load_state_dict(self.critic_extrinsic.state_dict())

        self.action_dim = action_shape[0]
        self.obs_dim = self.encoder.repr_dim

        self.icm = ICM(self.obs_dim, self.action_dim, hidden_dim,
                       50).to(self.device)

        # optimizers
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=lr)
        
        self.icm.train()

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_extrinsic_opt = torch.optim.Adam(self.critic_extrinsic.parameters(), lr=lr)


        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.vcse = vcse.VCSE(knn_k,self.device)
        self.se = vcse.PBE(False, 0, knn_k, False, False,
                             self.device)

        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)

        self.train()
        self.critic_target.train()

    def update_icm(self, obs, action, next_obs):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()
        self.icm_opt.zero_grad()
        loss.backward()
        self.icm_opt.step()
        if self.use_tb or self.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics,loss

    def compute_intr_reward_vcse(self, obs, value):
        rep = self.icm.get_rep(obs)
        reward,_,_,_, _, _ = self.vcse(rep,value)
        reward = reward.reshape(-1, 1)
        return reward

    def compute_intr_reward_se(self, obs):
        rep = self.icm.get_rep(obs)
        reward = self.se(rep)
        reward = reward.reshape(-1, 1)
        return reward

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_critic_extrinsic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_extrinsic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic_extrinsic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_extrinsic_target_q'] = target_Q.mean().item()
            metrics['critic_extrinsic_q1'] = Q1.mean().item()
            metrics['critic_extrinsic_q2'] = Q2.mean().item()
            metrics['critic_extrinsic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.critic_extrinsic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_extrinsic_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        metr, icm_loss = self.update_icm(obs.detach(), action, next_obs.detach())
        metrics.update(metr)

        with torch.no_grad():
            intr_act = self.actor(obs, 0).mean
            Q1, Q2 = self.critic_extrinsic(obs,intr_act)
            Q = torch.min(Q1, Q2)
            self.layerNorm = torch.nn.LayerNorm(Q.size(0)).to(self.device)
            Q = Q.reshape(-1)
            Q = self.layerNorm(Q) 
            Q = Q.reshape(-1,1)
            if self.do_vcse :
                intrinsic_reward = self.compute_intr_reward_vcse(obs, Q)
                self.s_ent_stats.update(intrinsic_reward)
            else :
                intrinsic_reward = self.compute_intr_reward_se(obs)
                self.s_ent_stats.update(intrinsic_reward)
                intrinsic_reward = intrinsic_reward / self.s_ent_stats.mean
        
        intrinsic_reward = self.beta * intrinsic_reward
        extrinsic_reward = reward
        reward = reward + intrinsic_reward
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
            metrics['extrinsic_reward'] = extrinsic_reward.mean().item()
            metrics['intrinsic_reward'] = intrinsic_reward.mean().item()
        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs.detach(), step))

        metrics.update(
            self.update_critic_extrinsic(obs.detach(), action, extrinsic_reward, discount, next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        # update critic target
        utils.soft_update_params(self.critic_extrinsic, self.critic_extrinsic_target,
                                self.critic_target_tau)
        
        return metrics
