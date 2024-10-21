import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
import torch
from common.ops import get_log_likelihood
from .encoder import Encoder 
from .decoder import Decoder 
from .decode_stragegy import get_decoding_strategy

def calculate_entropy(logprobs):
    """Calculate the entropy of the log probabilities distribution
    logprobs: Tensor of shape [batch, decoder_steps, num_actions]
    """
    logprobs = torch.nan_to_num(logprobs, nan=0.0)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)  # [batch, decoder steps]
    entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
    assert entropy.isfinite().all(), "Entropy is not finite"
    return entropy

class AttentionModelPolicy(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(AttentionModelPolicy, self).__init__()

        self.encoder = Encoder(embed_dim=embed_dim, num_layers=num_encoder_layers, num_heads=num_heads)
        self.decoder = Decoder(embed_dim=embed_dim, num_heads=num_heads)

        self.embed_dim = embed_dim

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td,
        env,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)
            step += 1
            if step > max_steps:
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict