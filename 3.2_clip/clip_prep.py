import gc
import os
from itertools import islice

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from transformers import pipeline, CLIPModel, CLIPProcessor
from tqdm import tqdm


class Cfg:
    model_id: str = "openai/clip-vit-base-patch32"
    batch_size: int = 32
    epochs: int = 40
    seed: int = 42

    # -------- Optim & Loss ----------
    lr_head: float = 1e-3  # 线性头
    wd_head: float = 1e-4
    lr_lora: float = 1e-4  # LoRA 注入层
    wd_lora: float = 1e-2
    lambda_text: float = 0.3  # 文本对齐辅助损失权重

    early_stopping: bool = True
    early_stop_patience = 3
    early_stop_minimum_improvement: float = 0.02

    # -------- LoRA ----------
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target: tuple = ("q_proj", "k_proj", "v_proj", "out_proj")  # 只对注意力投影层做LoRA
    # 也可扩展到 MLP 内部 proj，但注意稳定性

    amp: bool = True

# Configuration
LOADER_PATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda Availability:{torch.cuda.is_available()} Training on {device}")

cfg = Cfg()

PREPROCESS_DATA_ROOT = "data/preprocessed"
torch.manual_seed(cfg.seed)
model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_id).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_id)


# ---------- 2. Load Dataset ----------
def collate_pil(batch):
    # batch: List[ (PIL.Image.Image, int) ]
    images, labels = zip(*batch)  # images: tuple of PIL, labels: tuple of int
    return list(images), torch.tensor(labels)  # 让 processor 接收 list[PIL]，labels 变成 LongTensor


def preprocess_dataset(split="train", data_root='data', outputdir="data/preprocessed", batchsize=cfg.batch_size,
                       fp16=False):
    os.makedirs(outputdir, exist_ok=True)
    dataset = datasets.Flowers102(root=data_root, split=split, download=True)
    classes = dataset.classes
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True if split == "train" else False, num_workers=0,
                        collate_fn=collate_pil)

    # Allocate RAM
    N = len(dataset)
    C, H, W = 3, 224, 224
    pixels = torch.empty((N, C, H, W), dtype=torch.float32 if not fp16 else torch.float16)
    labels = torch.empty(N, dtype=torch.long)

    print(f"Preprocessing {split} data...")

    index = 0
    with torch.no_grad():
        for images, y in tqdm(loader, desc=f"Preprocessing {split}"):
            pix = processor(images=images, return_tensors="pt")['pixel_values']
            if fp16:
                pix = pix.half()
            b = pix.size(0)  # patch size
            pixels[index:index + b] = pix
            labels[index:index + b] = y
            index += b
            del pix

    out_path = os.path.join(outputdir, f"{split}.pt")
    torch.save({"pixel_values": pixels, "labels": labels, "fp16": fp16, "classes": classes}, out_path)
    print(f"Saved → {out_path} (pixels: {pixels.shape}, dtype={pixels.dtype})")

    print("Performing Garbage Cleaning...")
    del pixels, labels
    torch.cuda.empty_cache()
    gc.collect()


class CacheDataset:
    def __init__(self, split="train", root="data/preprocessed"):
        path = os.path.join(root, f"{split}.pt")
        try:
            obj = torch.load(path, map_location="cpu", mmap=True)  # 仅 2.5+ 可用
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        self.pixel_values = obj["pixel_values"]
        self.labels = obj["labels"]
        self.fp16 = bool(obj.get("fp16", False))
        self.classes = obj['classes']

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        return self.pixel_values[idx], self.labels[idx]


if __name__ == "__main__":
    if not os.path.exists(os.path.join(PREPROCESS_DATA_ROOT, "train.pt")):
        preprocess_dataset(split="train", data_root="data", outputdir=PREPROCESS_DATA_ROOT, fp16=False)
    if not os.path.exists(os.path.join(PREPROCESS_DATA_ROOT, "val.pt")):
        preprocess_dataset(split="val", data_root="data", outputdir=PREPROCESS_DATA_ROOT, fp16=False)
    if not os.path.exists(os.path.join(PREPROCESS_DATA_ROOT, "test.pt")):
        preprocess_dataset(split="test", data_root="data", outputdir=PREPROCESS_DATA_ROOT, fp16=False)

    # Load dataset

