from itertools import chain
import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None, use_entropy_reward=False, use_value_condition=False):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, use_entropy_reward, use_value_condition)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                            alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_extr_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            if self.acmodel.recurrent:
                _, _, _, value = self.acmodel(sb.obs, memory * sb.mask , get_extrinsic_value = True)
            else:
                _, _, _, value = self.acmodel(sb.obs,get_extrinsic_value = True)

            if self.use_entropy_reward:
                if self.use_value_condition :
                    sb_obs = sb.obs.image.transpose(1, 3).transpose(2, 3)
                    # compute value-conditional state entropy with random_encoder
                    src_feats = self.random_encoder(sb_obs)
                    if self.use_batch:
                        tgt_feats = src_feats.clone()[:,:,0,0]
                    else:
                        if self.full:
                            tgt_feats = torch.tensor(self.replay_buffer, device=self.device)
                        else:
                            tgt_feats = torch.tensor(self.replay_buffer[:self.idx], device=self.device)    
                    value_dist = value
                    # NOTE: a trick to normalize values using the samples from mini-batch
                    #       parameters will be initialized at every step
                    #       but not a neat implementation
                    self.layerNorm = torch.nn.LayerNorm(value_dist.size(0)).to(self.device)
                    value_dist = value_dist.reshape(-1)
                    value_dist = self.layerNorm(value_dist)
                    value_dist = value_dist.reshape(-1,1)
                    s_ent = self.compute_value_condition_state_entropy(src_feats[:,:,0,0], tgt_feats,value_dist, average_entropy=False)[:,0]
                    # we do not normalize intrinsic rewards, but update stats for logging purpose
                    self.s_ent_stats.update(s_ent)

                    sb_advantage = sb.advantage + self.beta * s_ent
                    sb_returnn = sb.returnn + self.beta * s_ent               
                else :
                    sb_obs = sb.obs.image.transpose(1, 3).transpose(2, 3)
                    # compute state entropy with random_encoder
                    src_feats = self.random_encoder(sb_obs)
                    if self.use_batch:
                        tgt_feats = src_feats.clone()[:,:,0,0]
                    else:
                        if self.full:
                            tgt_feats = torch.tensor(self.replay_buffer, device=self.device)
                        else:
                            tgt_feats = torch.tensor(self.replay_buffer[:self.idx], device=self.device)
                    s_ent = self.compute_state_entropy(src_feats[:,:,0,0], tgt_feats, average_entropy=True)[:,0]
                    s_ent = torch.log(s_ent + 1.0)
                    # normalize s_ent
                    self.s_ent_stats.update(s_ent)
                    norm_state_entropy = s_ent / self.s_ent_stats.std
                    s_ent = norm_state_entropy
                    sb_advantage = sb.advantage + self.beta * s_ent
                    sb_returnn = sb.returnn + self.beta * s_ent
                   
            else:
                sb_advantage = sb.advantage
                sb_returnn = sb.returnn

            extr_sb_returnn = sb.extr_returnn
            # Compute loss

            if self.acmodel.recurrent:
                dist, value, memory, extr_value = self.acmodel(sb.obs, memory * sb.mask,get_extrinsic_value = True)
            else:
                dist, value, _, extr_value = self.acmodel(sb.obs,get_extrinsic_value = True)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb_advantage).mean()

            value_loss = (value - sb_returnn).pow(2).mean()
            
            extr_value_loss = (extr_value - extr_sb_returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * (value_loss + extr_value_loss)

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_extr_value_loss += extr_value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_extr_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm,
            "update_extr_value_loss": update_extr_value_loss
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes

