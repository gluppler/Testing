import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from safetensors.torch import save_file
import requests
import os

# ── Model ────────────────────────────────────────────────────────────────────

class SVHNCNN(nn.Module):
    """CNN for SVHN classification (Opacus compatible)."""

    def __init__(self):
        super(SVHNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── Config ───────────────────────────────────────────────────────────────────

BASE_URL      = os.environ.get("BASE_URL", "http://localhost:5000")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS        = 20
BATCH_SIZE    = 256
LR            = 1e-3
MAX_GRAD_NORM = 1.0       # DP clipping norm
EPSILON       = 10.0      # privacy budget (higher = more accuracy, less privacy)
DELTA         = 1e-5      # DP delta — keep < 1/len(train_dataset)

# ── Data ─────────────────────────────────────────────────────────────────────

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD  = (0.1980, 0.2010, 0.1970)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD)
])

train_dataset = datasets.SVHN("data", split='train', download=True, transform=transform)
test_dataset  = datasets.SVHN("data", split='test',  download=True, transform=transform)

# Opacus requires drop_last=True and a non-zero generator
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ── Training with Opacus DP ───────────────────────────────────────────────────

model     = SVHNCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"Training with DP | ε={EPSILON}, δ={DELTA} | device={DEVICE}\n")

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    eps = privacy_engine.get_epsilon(DELTA)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} | loss={running_loss/len(train_loader):.4f} | "
          f"test_acc={acc:.4f} | ε={eps:.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────

# Opacus wraps the model in _module — unwrap for clean state_dict
save_file(model._module.state_dict(), "dp_model.safetensors")
print("\nSaved → dp_model.safetensors")

# ── Submit ────────────────────────────────────────────────────────────────────

with open("dp_model.safetensors", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/validate",
        files={"model": ("dp_model.safetensors", f, "application/octet-stream")}
    )

result = response.json()
print("\n── Validation Result ──────────────────────")
print(f"  Passed:   {result.get('passed')}")
print(f"  Accuracy: {result.get('accuracy')}")
print(f"  MIA Adv:  {result.get('mia_advantage')}")
print(f"  Time:     {result.get('evaluation_time')}s")
if result.get("flag"):
    print(f"\n  🚩 FLAG: {result['flag']}")
