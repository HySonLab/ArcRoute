import torch
import numpy as np
import argparse
from env.env import CARPEnv
from policy.policy import AttentionModelPolicy
from rl.ppo import PPO
from rl.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for PPO on CARPEnv")
    
    # Add arguments
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=1000, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--mini_batch_size', type=int, default=256//8, help='Mini-batch size')
    parser.add_argument('--train_data_size', type=int, default=100000, help='Training data size')
    parser.add_argument('--val_data_size', type=int, default=10000, help='Validation data size')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_loc', type=int, default=60, help='Number of nodes')
    parser.add_argument('--num_arc', type=int, default=60, help='Number of arcs')
    parser.add_argument('--variant', type=str, default='U', help='Environment variant')
    parser.add_argument('--checkpoint_dir', type=str, default='/usr/local/rsa/cpkts/60U2', help='Checkpoint directory')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Training accelerator')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize environment
    env = CARPEnv(generator_params={'num_loc': args.num_loc, 'num_arc': args.num_arc}, variant=args.variant)
    
    # Initialize policy
    policy = AttentionModelPolicy(
                    embed_dim=args.embed_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_heads=args.num_heads)

    # Initialize PPO model
    model = PPO(env, 
                policy,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                train_data_size=args.train_data_size,
                val_data_size=args.val_data_size)
    

    # _model = PPO.load_from_checkpoint('/usr/local/rsa/cpkts/best60U3.ckpt')
    # model.policy.load_state_dict(_model.policy.state_dict())
    # model.critic.load_state_dict(_model.critic.state_dict())

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir,
                                          filename="{epoch:03d}",
                                          save_top_k=3,
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