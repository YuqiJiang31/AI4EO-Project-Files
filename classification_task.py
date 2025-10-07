import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import random
import numpy as np

NUM_CLASSES = 21
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DATA_DIR = 'UCMerced_HR_RealESRGAN'
SEED = 42
BEST_MODEL_PATH = "best_resnet18_RealESRGAN.pth"

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Use ImageNet normalization stats
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
# Validation transform
val_dataset.dataset.transform = data_transforms['val']

# Disable parallel workers: num_workers=0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
except:
    model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Start training with {DATA_DIR} (num_classes={NUM_CLASSES})...")
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels)

    epoch_acc = corrects.double().item() / len(val_dataset)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save({
            "model_state": model.state_dict(),
            "acc": best_acc,
            "epoch": epoch + 1,
            "classes": class_names
        }, BEST_MODEL_PATH)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {epoch_loss:.4f}  Val Acc: {epoch_acc:.4f}  Best: {best_acc:.4f}")

print("Training finished! Best val accuracy:", f"{best_acc:.4f}")
print(f"Best model saved: {BEST_MODEL_PATH}")