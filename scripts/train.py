import torch
torch.set_float32_matmul_precision('medium')
# Decoder pointer attention uses q_len=1: Flash/MemEfficient SDPA hit CUDA grid
# limits at large batch×heads and are slower for q_len=1 (tiling buys nothing).
# Math backend has no kernel limits and is ~11% faster per sample here.
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
    parser = argparse.ArgumentParser(description="GRPO training for HDCARP")

    parser.add_argument('--group_size', type=int, default=8,
                        help='K samples/instance (group size).')
    parser.add_argument('--reward_mode', type=str, default='vector',
                        help='scalar|vector (default: vector for GRPO).')
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--train_data_size', type=int, default=100000)
    parser.add_argument('--val_data_size', type=int, default=10000)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_loc', type=int, default=20)
    parser.add_argument('--num_arc', type=int, default=20)
    parser.add_argument('--num_vehicle', type=str, default='3',
                        help='Fleet size(s): "3" or comma list "2,3,5,7,10".')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Multi-size ladder: "nloc:narc,..." e.g. '
                             '"20:40,30:60,40:80,50:100,40:120".')
    parser.add_argument('--variant', type=str, default='U')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints/clP_ladder')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--path_train_data', type=str, default="data/train_data.data")
    parser.add_argument('--path_val_data', type=str, default="data/val_data.data")
    parser.add_argument('--path_test_data', type=str, default="data/test_data.data")
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Checkpoint to load policy weights from (curriculum warm-start). '
                             'Loads weights only — epoch/optimizer state is NOT restored.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sizes = None
    if args.sizes:
        sizes = [tuple(int(x) for x in pair.split(":")) for pair in args.sizes.split(",")]
        print(f"Multi-size training over (num_loc, num_arc) buckets: {sizes}")

    fleet = [int(x) for x in str(args.num_vehicle).split(",")]
    fleet = fleet[0] if len(fleet) == 1 else fleet
    print(f"Fleet M (swept in reward, policy M-agnostic): {fleet}")
    print(f"Algo=grpo  reward_mode={args.reward_mode}  group_size={args.group_size}")

    env = CARPEnv(num_loc=args.num_loc, num_arc=args.num_arc, num_vehicle=fleet,
                  variant=args.variant, sizes=sizes, reward_mode=args.reward_mode)

    policy = AttentionModelPolicy(
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
    )

    metrics = {"train": ["reward", "loss", "entropy",
                         "T1_mean", "T2_mean", "T3_mean"]}

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
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        dataloader_num_workers=args.num_workers,
        metrics=metrics,
        reload_train_dataloader=4,
        group_size=args.group_size,
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
        precision="bf16-mixed",
    )

    trainer.fit(model)
