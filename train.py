import torch
import numpy as np
from env.env import CARPEnv
from policy.policy import AttentionModelPolicy
from rl.ppo import PPO
from rl.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    torch.manual_seed(6868)
    np.random.seed(6868)

    env = CARPEnv(generator_params={"num_loc": 20, "num_arc": 20})
    
    policy = AttentionModelPolicy(
                embed_dim=128,
                num_encoder_layers=12,
                num_heads=8)

    model = PPO(env, 
                policy,
                batch_size=512*4,
                mini_batch_size=256,
                train_data_size=1000000,
                val_data_size=10000
                ) 

    # _model = PPO.load_from_checkpoint('/home/project/cpkts/epoch=008.ckpt')
    # model.policy.load_state_dict(_model.policy.state_dict())
    # model.critic.load_state_dict(_model.critic.state_dict())

    checkpoint_callback = ModelCheckpoint(dirpath="../cpkts/cl1", # save to checkpoints/
                                        filename="{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor="val/reward", # monitor validation reward
                                        mode="max") # maximize validation reward
    
    trainer = Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)