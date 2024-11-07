import torch
import abc
import torch.nn.functional as F
from common.ops import batchify, gather_by_index, unbatchify_and_gather, unbatchify

def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done out-of-place.
    Ref: https://github.com/togethercomputer/stripedhyena/blob/7e13f618027fea9625be1f2d2d94f9a361f6bd02/stripedhyena/sample.py#L6
    """
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, float("-inf"))


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done out-of-place.
    Ref: https://github.com/togethercomputer/stripedhyena/blob/7e13f618027fea9625be1f2d2d94f9a361f6bd02/stripedhyena/sample.py#L14
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))

def process_logits(
    logits: torch.Tensor,
    mask: torch.Tensor = None,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    tanh_clipping: float = 0,
    mask_logits: bool = True,
):
    """Convert logits to log probabilities with additional features like temperature scaling, top-k and top-p sampling.

    Note:
        We convert to log probabilities instead of probabilities to avoid numerical instability.
        This is because, roughly, softmax = exp(logits) / sum(exp(logits)) and log(softmax) = logits - log(sum(exp(logits))),
        and avoiding the division by the sum of exponentials can help with numerical stability.
        You may check the [official PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html).

    Args:
        logits: Logits from the model (batch_size, num_actions).
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        temperature: Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation).
        top_p: Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751). Remove tokens that have a cumulative probability
            less than the threshold 1 - top_p (lower tail of the distribution). If 0, do not perform.
        top_k: Top-k sampling, i.e. restrict sampling to the top k logits. If 0, do not perform. Note that we only do filtering and
            do not return all the top-k logits here.
        tanh_clipping: Tanh clipping (https://arxiv.org/abs/1611.09940).
        mask_logits: Whether to mask logits of infeasible actions.
    """

    # Tanh clipping from Bello et al. 2016
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    # In RL, we want to mask the logits to prevent the agent from selecting infeasible actions
    if mask_logits:
        assert mask is not None, "mask must be provided if mask_logits is True"
        logits[~mask] = float("-inf")

    logits = logits / temperature  # temperature scaling

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # safety check
        logits = modify_logits_for_top_k_filtering(logits, top_k)

    if top_p > 0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
        logits = modify_logits_for_top_p_filtering(logits, top_p)

    # Compute log probabilities
    return F.log_softmax(logits, dim=-1)

def calculate_entropy(logprobs):
    """Calculate the entropy of the log probabilities distribution
    logprobs: Tensor of shape [batch, decoder_steps, num_actions]
    """
    logprobs = torch.nan_to_num(logprobs, nan=0.0)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)  # [batch, decoder steps]
    entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
    assert entropy.isfinite().all(), "Entropy is not finite"
    return entropy

class DecodingStrategy(metaclass=abc.ABCMeta):
    """Base class for decoding strategies. Subclasses should implement the :meth:`_step` method.
    Includes hooks for pre and post main decoding operations.

    Args:
        temperature: Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation). Defaults to 1.0.
        top_p: Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751). Defaults to 0.0.
        top_k: Top-k sampling, i.e. restrict sampling to the top k logits. If 0, do not perform. Defaults to 0.
        mask_logits: Whether to mask logits of infeasible actions. Defaults to True.
        tanh_clipping: Tanh clipping (https://arxiv.org/abs/1611.09940). Defaults to 0.
        multistart: Whether to use multistart decoding. Defaults to False.
        multisample: Whether to use sampling decoding. Defaults to False.
        num_starts: Number of starts for multistart decoding. Defaults to None.
    """

    name = "base"

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        mask_logits: bool = True,
        tanh_clipping: float = 0,
        multistart: bool = False,
        multisample: bool = False,
        num_starts = None,
        select_start_nodes_fn = None,
        improvement_method_mode: bool = False,
        select_best: bool = False,
        store_all_logp: bool = True,
        **kwargs,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        self.multistart = multistart
        self.multisample = multisample
        self.num_starts = num_starts
        self.select_start_nodes_fn = select_start_nodes_fn
        self.improvement_method_mode = improvement_method_mode
        self.select_best = select_best
        self.store_all_logp = store_all_logp
        # initialize buffers
        self.actions = []
        self.logprobs = []

    def reset(self):
        self.actions = []
        self.logprobs = []

    @abc.abstractmethod
    def _step(self, logprobs,mask,td,action = None,**kwargs):
        raise NotImplementedError("Must be implemented by subclass")

    def pre_decoder_hook(self, td, env, action = None):
        pass

    def post_decoder_hook(self, td, env):
        assert (
            len(self.logprobs) > 0
        ), "No logprobs were collected because all environments were done. Check your initial state"
        logprobs = torch.stack(self.logprobs, 1)
        actions = torch.stack(self.actions, 1)
        return logprobs, actions, td, env

    def step( self, logits, mask, td=None, action = None, **kwargs):
        """Main decoding operation. This method should be called in a loop until all sequences are done.

        Args:
            logits: Logits from the model.
            mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
            td: TensorDict containing the current state of the environment.
            action: Optional action to use, e.g. for evaluating log probabilities.
        """
        if not self.mask_logits:  # set mask_logit to None if mask_logits is False
            mask = None

        logprobs = process_logits(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.mask_logits,
        )
        logprobs, selected_action, td = self._step(
            logprobs, mask, td, action=action, **kwargs
        )

        # directly return for improvement methods, since the action for improvement methods is finalized in its own policy
        if self.improvement_method_mode:
            return logprobs, selected_action
        # for others
        if not self.store_all_logp:
            logprobs = gather_by_index(logprobs, selected_action, dim=1)
        td.set("action", selected_action)
        self.actions.append(selected_action)
        self.logprobs.append(logprobs)
        return td

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select the action with the highest probability."""
        # [BS], [BS]
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert (
                not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"

        return selected

    @staticmethod
    def sampling(logprobs, mask=None):
        """Sample an action with a multinomial distribution given by the log probabilities."""
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)

        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                selected = probs.multinomial(1).squeeze(1)
            assert (
                not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"

        return selected

    def _select_best(self, logprobs, actions, td, env):
        rewards = env.get_reward(td, actions)
        _, max_idxs = unbatchify(rewards, self.num_starts).max(dim=-1)

        actions = unbatchify_and_gather(actions, max_idxs, self.num_starts)
        logprobs = unbatchify_and_gather(logprobs, max_idxs, self.num_starts)
        td = unbatchify_and_gather(td, max_idxs, self.num_starts)

        return logprobs, actions, td, env
    
class Greedy(DecodingStrategy):
    name = "greedy"

    def _step(self, logprobs, mask, td, **kwargs):
        selected = self.greedy(logprobs, mask)
        return logprobs, selected, td

class Sampling(DecodingStrategy):
    name = "sampling"

    def _step(self, logprobs, mask, td, **kwargs):
        """Sample an action with a multinomial distribution given by the log probabilities."""
        selected = self.sampling(logprobs, mask)
        return logprobs, selected, td


class Evaluate(DecodingStrategy):
    name = "evaluate"

    def _step(self,logprobs,mask,td,action,**kwargs):
        """The action is provided externally, so we just return the action"""
        selected = action
        return logprobs, selected, td
    
def get_decoding_strategy(decoding_strategy, **config):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "evaluate": Evaluate,
    }
    return strategy_registry.get(decoding_strategy, Sampling)(**config)