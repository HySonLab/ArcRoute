import torch
import lightning as pl
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

# 1️⃣ DataModule: Quản lý dữ liệu MNIST
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        MNIST(root="data", train=True, download=True)  # Tải dataset nếu chưa có

    def setup(self, stage=None):
        dataset = MNIST(root="data", train=True, transform=self.transform)
        self.train_data, self.val_data = random_split(dataset, [55000, 5000])
        self.test_data = MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4)

# 2️⃣ Model: Mạng MLP đơn giản
class MLP(pl.LightningModule):
    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)  # Flatten ảnh
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# 3️⃣ Training
data_module = MNISTDataModule(batch_size=64)
model = MLP()

trainer = pl.Trainer(max_epochs=5, accelerator="gpu" if torch.cuda.is_available() else "cpu")
trainer.fit(model, data_module)
