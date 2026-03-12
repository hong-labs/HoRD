import torch
from torch import distributions, nn
from hydra.utils import instantiate
from hord.agents.common.mlp import MultiHeadedMLP
from hord.global_config import Config


class PPOActor(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.logstd = nn.Parameter(
            torch.ones(num_out, device=("cuda" if torch.cuda.is_available() else "cpu")) * config.actor_logstd,
            requires_grad=False,
        )
        self.mu: MultiHeadedMLP = instantiate(self.config.mu_model, num_out=num_out)

    def forward(self, input_dict):
        mu = self.mu(input_dict)
        mu = torch.tanh(mu)
        std = torch.exp(self.logstd)
        dist = distributions.Normal(mu, std)
        return dist


class PPOModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create networks
        self._actor: PPOActor = instantiate(
            self.config.actor,
        )
        self._critic: MultiHeadedMLP = instantiate(
            self.config.critic,
        )

    def get_action_and_value(self, input_dict: dict):
        dist = self._actor(input_dict)
        action = dist.sample()
        
        # DR Action mode: use zero value because single-step RL does not require a value function.
        if Config.use_delta and not Config.freeze_delta:
            batch_size = action.shape[0]
            value = torch.zeros(batch_size, device=action.device)
        else:
            value = self._critic(input_dict).flatten()

        logstd = self._actor.logstd
        std = torch.exp(logstd)

        neglogp = self.neglogp(action, dist.mean, std, logstd)
        return action, neglogp, value.flatten()

    def act(self, input_dict: dict, mean: bool = True) -> torch.Tensor:
        dist = self._actor(input_dict)
        if mean:
            return dist.mean
        return dist.sample()

    @staticmethod
    def neglogp(x, mean, std, logstd):
        dist = distributions.Normal(mean, std)
        return -dist.log_prob(x).sum(dim=-1)
