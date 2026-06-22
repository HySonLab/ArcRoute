import torch
import numpy as np
import argparse
from env.env import CARPEnv
from policy.policy import AttentionModelPolicy
from rl.ppo import PPO
from rl.grpo import GRPO
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for PPO on CARPEnv")
    
    # Add arguments
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'grpo'],
                        help='RL algorithm: ppo (weighted-sum + critic, default) or '
                             'grpo (group lexicographic rank, critic-free).')
    parser.add_argument('--group_size', type=int, default=8,
                        help='K samples/instance for GRPO (group size).')
    parser.add_argument('--reward_mode', type=str, default=None,
                        help='scalar|vector. Auto = vector if algo==grpo else scalar.')
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=1000, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--mini_batch_size', type=int, default=256, help='Mini-batch size')
    parser.add_argument('--train_data_size', type=int, default=100000, help='Training data size')
    parser.add_argument('--val_data_size', type=int, default=10000, help='Validation data size')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_loc', type=int, default=20, help='Number of nodes')
    parser.add_argument('--num_arc', type=int, default=20, help='Number of arcs')
    parser.add_argument('--num_vehicle', type=str, default='3',
                        help='Fleet size(s): an int "3" OR a comma list "2,3,5,7,10". '
                             'A list mixes M PER-INSTANCE (M is swept in the reward/'
                             'Scheduler; the policy stays M-agnostic). dynamic_plan Phase 3.')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Phase 6 multi-size training: "nloc:narc,..." e.g. '
                             '"20:40,30:60,40:80,50:100,40:120". Overrides num_loc/num_arc.')
    parser.add_argument('--variant', type=str, default='U', help='Environment variant')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/60arcs', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for data loader') 
    parser.add_argument('--accelerator', type=str, default='gpu', help='Training accelerator')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    parser.add_argument('--path_train_data', type=str, default="data/train_data.data", help='path_train_data')
    parser.add_argument('--path_val_data', type=str, default="data/val_data.data", help='path_val_data')
    parser.add_argument('--path_test_data', type=str, default="data/test_data.data", help='path_test_data')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Phase 6: parse the optional size ladder "nloc:narc,nloc:narc,...".
    sizes = None
    if args.sizes:
        sizes = [tuple(int(x) for x in pair.split(":")) for pair in args.sizes.split(",")]
        print(f"Multi-size training over (num_loc, num_arc) buckets: {sizes}")

    # Phase 3: parse fleet "2,3,5,7,10" -> list (mixed per-instance) or single int.
    fleet = [int(x) for x in str(args.num_vehicle).split(",")]
    fleet = fleet[0] if len(fleet) == 1 else fleet
    print(f"Fleet M (swept in reward, policy M-agnostic): {fleet}")

    # D2 Phase 4: reward_mode auto-selects vector for GRPO (it ranks the T-vector),
    # scalar for PPO (the old -(T.w) reward). Explicit --reward_mode overrides.
    reward_mode = args.reward_mode or ('vector' if args.algo == 'grpo' else 'scalar')
    print(f"Algo={args.algo}  reward_mode={reward_mode}"
          + (f"  group_size={args.group_size}" if args.algo == 'grpo' else ''))

    # Initialize environment
    env = CARPEnv(num_loc=args.num_loc, num_arc=args.num_arc, num_vehicle=fleet,
                  variant=args.variant, sizes=sizes, reward_mode=reward_mode)

    # Initialize policy
    policy = AttentionModelPolicy(
                    embed_dim=args.embed_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_heads=args.num_heads)

    # D2 Phase 4: select the CLASS by --algo (no flag inside ppo.py). GRPO logs
    # per-objective T1/T2/T3 means so the learning curve shows T2/T3 descending.
    Model = GRPO if args.algo == 'grpo' else PPO
    extra = {'group_size': args.group_size} if args.algo == 'grpo' else {}
    metrics = ({"train": ["reward", "loss", "surrogate_loss", "entropy",
                          "T1_mean", "T2_mean", "T3_mean"]}
               if args.algo == 'grpo' else
               {"train": ["reward", "loss", "surrogate_loss", "value_loss", "entropy"]})

    # Initialize RL model
    model = Model(env,
                policy,
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
                **extra)
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir,
                                          filename="{epoch:03d}",
                                          save_top_k=1,
                                          save_last=True,
                                          monitor="val/reward",
                                          mode="max")
    
    trainer = Trainer(
        max_epochs=args.max_epoch,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)