import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
import torch
from common.ops import get_log_likelihood
from .encoder import Encoder 
from .decoder import Decoder 
from .decode_stragegy import get_decoding_strategy, calculate_entropy
from common.nb_utils import run_parallel, refine_routes, prob_idxs
from common.ops import gather_by_index
import numpy as np


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
        calc_reward: bool = False,
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
            decode_type = f"{phase}_decode_type"

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            # store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            store_all_logp=True,
            **decoding_kwargs,
        )

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden)
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
        
        actions = torch.stack(decode_strategy.actions, dim=1)
        actions = actions.cpu().numpy().astype(np.int32)
        actions_org = actions.copy()
        actions = run_parallel(refine_routes, actions, td["demand"].cpu().numpy(), max_vehicles=5)
        max_pad = max(map(lambda x: len(x), actions))
        actions_np = np.array([np.pad(x, (0, max_pad - len(x)), 'constant') for x in actions])

        out = {}
        out["actions"] = torch.tensor(actions_np, dtype=torch.int64, device=td.device)
        
        if calc_reward:
            td.set("reward", env.get_reward(td, out["actions"]))
            out["reward"] = td["reward"]

        if phase != 'infer':
            logprobs = torch.stack(decode_strategy.logprobs, dim=1)        
            idxs = np.array(run_parallel(prob_idxs, actions_org, actions_np))
            idxs = torch.tensor(idxs, dtype=torch.int64, device=td.device)
            logprobs = gather_by_index(logprobs, idxs, dim=1)

            out['log_likelihood'] = get_log_likelihood(
                logprobs, out["actions"], td.get("mask", None), return_sum_log_likelihood)

            if return_entropy:
                out["entropy"] = calculate_entropy(logprobs)
            if return_hidden:
                out["hidden"] = hidden
            if return_init_embeds:
                out["init_embeds"] = init_embeds

        return out