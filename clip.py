from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Configuration
LOADER_PATCH_SIZE = 32






# ---------- 1. Preprocessing transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # 调整到模型常见输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet Mean
        std=[0.229, 0.224, 0.225]    # ImageNet standard variance
    )
])

# ---------- 2. Load Dataset ----------
train_dataset = datasets.Flowers102(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

val_dataset = datasets.Flowers102(
    root="./data",
    split="val",
    download=False,
    transform=transform
)

test_dataset = datasets.Flowers102(
    root="./data",
    split="test",
    download=False,
    transform=transform
)

# ---------- 3. 封装为 DataLoader ----------
train_loader = DataLoader(train_dataset, batch_size=LOADER_PATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=LOADER_PATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=LOADER_PATCH_SIZE, shuffle=False)