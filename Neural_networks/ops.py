from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, List
from numpy import ndarray
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=0.2, name=None, device=None, eps=1e-5, weight_decay=0):

        defaults = dict(idx=idx, lr=lr, w=w, agents=agents, name=name, device=device,
                        eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @classmethod
    def cls_collect_params_grads(cls, optimizer: Optimizer, independent: bool = False):
        var_s = []
        grads = []
        for group in optimizer.param_groups:
            if independent:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    var_s.append(p.data.clone().detach())
                    grads.append(p.grad.data.clone().detach())
                return var_s, grads
            for p in group['params']:
                if p.grad is None:
                    continue
                var_s.append(p.data)
                grads.append(p.grad.data)
        return var_s, grads

    def collect_params_grads(self, independent: bool = False):
        return self.cls_collect_params_grads(self, independent)

    def collect_lr(self):
        for group in self.param_groups:
            return group["lr"]

    def collect_prev_lr(self):
        for group in self.param_groups:
            return group["prev_lr"]

    @property
    def _device(self) -> torch.device:
        return self.param_groups[0]["device"]

    @property
    def _w(self) -> ndarray:
        return self.param_groups[0]["w"]

    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        """step method in Optimizer class"""


class GD(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self,
             lr_constant: float,
             ) -> None:

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                p.data = p.data - lr_constant * p.grad.data
                continue

        return None


class DSGD(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def collect_grad(cls, params: Iterable) -> List[Tensor]:
        grads = []
        for p in params:
            if p.grad is None:
                continue
            grads.append(p.grad.data)
        return grads

    @classmethod
    def collect_x(cls, params: Iterable) -> List[Tensor]:
        var_x: list[Tensor] = []
        for p in params:
            if p.grad is None:
                continue
            var_x.append(p.data.clone().detach())
        return var_x

    def step(self,
             lr_list: Sequence[float],
             switching_k: int,
             lr_constant: float,
             k: int,
             vars: dict[int, list[Tensor]],
             grads: dict[int, list[Tensor]],
             participate_agents: list[int] = None,
             w_partial: ndarray = None,
             ) -> None:

        for group in self.param_groups:
            idx = group['idx']
            agents = group["agents"]
            device = group["device"]
            w = group["w"]
            
            # Use partial participation if specified
            if participate_agents is not None and w_partial is not None:
                # Only participating agents are involved
                active_agents = participate_agents
                w_matrix = w_partial
                # Map global agent idx to local idx in participating agents
                if idx in participate_agents:
                    local_idx = participate_agents.index(idx)
                else:
                    # Non-participating agent, skip update
                    continue
            else:
                # Full participation (original behavior)
                active_agents = list(range(agents))
                w_matrix = w
                local_idx = idx

            printed_switch_msg = False
            sub = 0
            for i, p in enumerate(group['params']):
                summat_x: Tensor = torch.zeros_like(p).to(device)
                if k == 0:
                    for local_j, j in enumerate(active_agents):
                        temp_var: Tensor = vars[j][i + sub] - lr_list[j] * grads[j][i + sub]
                        summat_x += w_matrix[local_idx, local_j] * temp_var
                    p.data = summat_x
                    continue
                if switching_k is not None and k >= switching_k:
                    if not printed_switch_msg and k == switching_k:
                        print("Switch to the same stepsize", switching_k)
                        printed_switch_msg = True
                    for local_j, j in enumerate(active_agents):
                        temp_var: Tensor = vars[j][i + sub] - lr_constant * grads[j][i + sub]
                        summat_x += w_matrix[local_idx, local_j] * temp_var
                    p.data = summat_x
                else:
                    for local_j, j in enumerate(active_agents):
                        temp_var: Tensor = vars[j][i + sub] - lr_list[j] * grads[j][i + sub]
                        summat_x += w_matrix[local_idx, local_j] * temp_var
                    p.data = summat_x

        return None


class FedAdaptiveBase(Base):
    """Base class for adaptive federated optimizers (FedAdaGrad, FedYogi, FedAdam)"""
    def __init__(self, params, idx, w, agents, lr=0.2, name=None, device=None, 
                 eps=1e-5, weight_decay=0, beta1=0.9, beta2=0.99, tau=1e-3):
        
        defaults = dict(idx=idx, lr=lr, w=w, agents=agents, name=name, device=device,
                        eps=eps, weight_decay=weight_decay, beta1=beta1, beta2=beta2, tau=tau)
        
        # Initialize using parent's __init__ (Optimizer), not Base.__init__
        Optimizer.__init__(self, params, defaults)
        
        # Initialize momentum and second moment buffers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['m'] = torch.zeros_like(p.data)  # momentum
                state['v'] = torch.zeros_like(p.data)  # second moment

    @classmethod
    def collect_x(cls, params: Iterable) -> List[Tensor]:
        var_x: list[Tensor] = []
        for p in params:
            if p.grad is None:
                continue
            var_x.append(p.data.clone().detach())
        return var_x

    @classmethod
    def collect_grad(cls, params: Iterable) -> List[Tensor]:
        grads = []
        for p in params:
            if p.grad is None:
                continue
            grads.append(p.grad.data)
        return grads

    @abstractmethod
    def _update_second_moment(self, v_old: Tensor, delta_squared: Tensor, beta2: float) -> Tensor:
        """Update second moment - different for each algorithm"""
        pass

    def step(self,
             local_lr: float,
             server_lr: float,
             vars_after_local: dict[int, list[Tensor]],
             participate_agents: list[int],
             ) -> None:
        """
        Federated adaptive step with partial participation
        
        Args:
            local_lr: Learning rate for local SGD steps (not used here, for interface consistency)
            server_lr: Server learning rate (eta in algorithm)
            vars_after_local: Dictionary of model parameters after K local steps for each agent
            participate_agents: List of participating agent indices
        """

        for group in self.param_groups:
            idx = group['idx']
            device = group["device"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            tau = group["tau"]
            
            num_participate = len(participate_agents)
            
            for i, p in enumerate(group['params']):
                state = self.state[p]
                
                # Store x_t (current global model)
                x_t = p.data.clone()
                
                # Compute pseudo-gradients Delta_i = x_{i,K} - x_t for participating clients
                # and average them: Delta_t = (1/|S|) * sum_{i in S} Delta_i
                delta_t = torch.zeros_like(p.data).to(device)
                
                for j in participate_agents:
                    # Pseudo-gradient: Delta_i = x_{i,K} - x_t
                    delta_i = vars_after_local[j][i] - x_t
                    delta_t += delta_i
                
                delta_t /= num_participate  # Average over participating clients
                
                # Update momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * Delta_t
                state['m'] = beta1 * state['m'] + (1 - beta1) * delta_t
                
                # Update second moment (algorithm-specific)
                delta_squared = delta_t ** 2
                state['v'] = self._update_second_moment(state['v'], delta_squared, beta2)
                
                # Server update: x_{t+1} = x_t + eta * m_t / (sqrt(v_t) + tau)
                p.data = x_t + server_lr * state['m'] / (torch.sqrt(state['v']) + tau)

        return None


class FedAdaGrad(FedAdaptiveBase):
    """FedAdaGrad: v_t = v_{t-1} + Delta_t^2"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_second_moment(self, v_old: Tensor, delta_squared: Tensor, beta2: float) -> Tensor:
        # FedAdaGrad: v_t = v_{t-1} + Delta_t^2
        return v_old + delta_squared


class FedYogi(FedAdaptiveBase):
    """FedYogi: v_t = v_{t-1} - (1-beta2) * Delta_t^2 * sign(v_{t-1} - Delta_t^2)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_second_moment(self, v_old: Tensor, delta_squared: Tensor, beta2: float) -> Tensor:
        # FedYogi: v_t = v_{t-1} - (1-beta2) * Delta_t^2 * sign(v_{t-1} - Delta_t^2)
        return v_old - (1 - beta2) * delta_squared * torch.sign(v_old - delta_squared)


class FedAdam(FedAdaptiveBase):
    """FedAdam: v_t = beta2 * v_{t-1} + (1-beta2) * Delta_t^2"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_second_moment(self, v_old: Tensor, delta_squared: Tensor, beta2: float) -> Tensor:
        # FedAdam: v_t = beta2 * v_{t-1} + (1-beta2) * Delta_t^2
        return beta2 * v_old + (1 - beta2) * delta_squared


class FedAvgM(Base):
    """FedAvg with Momentum (FedAvgM)"""
    def __init__(self, params, idx, w, agents, lr=0.2, name=None, device=None, 
                 eps=1e-5, weight_decay=0, beta=0.9, server_lr=1.0):
        
        defaults = dict(idx=idx, lr=lr, w=w, agents=agents, name=name, device=device,
                        eps=eps, weight_decay=weight_decay, beta=beta, server_lr=server_lr)
        
        # Initialize using Optimizer's __init__
        Optimizer.__init__(self, params, defaults)
        
        # Initialize server-side momentum buffer (global gradient estimate g)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['server_momentum'] = torch.zeros_like(p.data)  # g in the algorithm

    @classmethod
    def collect_x(cls, params: Iterable) -> List[Tensor]:
        var_x: list[Tensor] = []
        for p in params:
            var_x.append(p.data.clone().detach())
        return var_x

    def step(self,
             local_lr: float,
             server_lr: float,
             num_local_steps: int,
             vars_after_local: dict[int, list[Tensor]],
             participate_agents: list[int],
             ) -> None:
        """
        FedAvgM server update with partial participation
        
        Args:
            local_lr: Local learning rate η
            server_lr: Server learning rate γ
            num_local_steps: Number of local steps K
            vars_after_local: Model parameters after K local steps for each participating agent
            participate_agents: List of participating agent indices
        """

        for group in self.param_groups:
            idx = group['idx']
            device = group["device"]
            
            num_participate = len(participate_agents)
            
            for i, p in enumerate(group['params']):
                state = self.state[p]
                
                # Store x^r (current global model)
                x_r = p.data.clone()
                
                # Aggregate local updates: g^{r+1} = 1/(ηNK) * Σ_{i in S} (x^r - x_i^{r,K})
                pseudo_grad = torch.zeros_like(p.data).to(device)
                
                for j in participate_agents:
                    # Compute (x^r - x_i^{r,K})
                    model_diff = x_r - vars_after_local[j][i]
                    pseudo_grad += model_diff
                
                # Normalize by ηNK (local_lr * num_participate * num_local_steps)
                pseudo_grad /= (local_lr * num_participate * num_local_steps)
                
                # Update server momentum: g^{r+1} (this is the aggregated pseudo-gradient)
                state['server_momentum'] = pseudo_grad
                
                # Server update: x^{r+1} = x^r - γ * g^{r+1}
                p.data = x_r - server_lr * state['server_momentum']

        return None




