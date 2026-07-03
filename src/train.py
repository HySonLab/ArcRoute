import random
import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import numpy as np
import argparse
from env.env import CARPEnv
from policy.policy import AttentionModelPolicy
from trainers.grpo import GRPO
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for HDCARP on OSM data")

    parser.add_argument('--group_size', type=int, default=8)
    parser.add_argument('--reward_mode', type=str, default='vector')
    parser.add_argument('--seed', type=int, default=6868)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_vehicle', type=str, default='2,3,5,7,10',
                        help='Fleet size(s) sampled per instance: "3" or "2,3,5,7,10".')
    parser.add_argument('--variant', type=str, default='P')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints/grpo_osm')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--path_train_data', type=str, default='data/osm_train')
    parser.add_argument('--path_val_data',   type=str, default='data/osm_val')
    parser.add_argument('--path_test_data',  type=str, default='data/osm_test')
    parser.add_argument('--curriculum', type=str, default=None,
                        choices=['small', 'medium', 'large'],
                        help='Load only this curriculum phase of training buckets. '
                             'small=[20_30,20_40]  medium=[30_45..50_75,40_80]  '
                             'large=[50_100,80_120,80_133]. Default: all buckets.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Checkpoint to load policy weights from (warm-start). '
                             'Loads weights only — epoch/optimizer state is NOT restored.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fleet = [int(x) for x in str(args.num_vehicle).split(",")]
    fleet = fleet[0] if len(fleet) == 1 else fleet
    print(f"Fleet M (sampled per instance): {fleet}")
    print(f"reward_mode={args.reward_mode}  group_size={args.group_size}")

    from utils.data_utils import CURRICULUM
    train_buckets = CURRICULUM[args.curriculum] if args.curriculum else None
    print(f"curriculum={args.curriculum or 'all'}  train_buckets={train_buckets}")

    env = CARPEnv(num_vehicle=fleet, variant=args.variant, reward_mode=args.reward_mode)

    policy = AttentionModelPolicy(
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        tanh_clipping=10,
        P=env.P,  # decoder's per-class features must match the env's class count
    )

    metrics = {"train": ["reward", "loss", "T1_mean", "T2_mean", "T3_mean"]}

    if args.resume_from:
        import os
        assert os.path.isfile(args.resume_from), f"resume_from not found: {args.resume_from}"
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        sd = {k[len("policy."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("policy.")}
        policy.load_state_dict(sd, strict=True)
        print(f"Loaded policy weights from {args.resume_from} (epoch/optimizer reset)")

    model = GRPO(
        env, policy,
        path_train_data=args.path_train_data,
        path_val_data=args.path_val_data,
        path_test_data=args.path_test_data,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        metrics=metrics,
        group_size=args.group_size,
        train_buckets=train_buckets,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="{epoch:03d}",
        save_top_k=1,
        save_last=True,
        monitor="val/lex_best",
        mode="max",
    )

    trainer = Trainer(
        max_epochs=args.max_epoch,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir="outputs",
        callbacks=[checkpoint_callback],
        precision="32-true",
    )

    trainer.fit(model)
