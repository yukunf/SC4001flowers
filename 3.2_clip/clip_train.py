import gc
import os
from itertools import islice

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from transformers import pipeline, CLIPModel, CLIPProcessor, CLIPVisionModel
from tqdm import tqdm
from clip_prep import CacheDataset, DataLoader, cfg, processor, clip_model, device

promptTemplate = {
    "A photo of {}.",
    "A photo of flower {}.",
    "Botanic picture of {}",
    "A example picture of type {}"
}
# Use more templates to reduce sensitivity to other contexts

@torch.no_grad()
def build_text_embeddings(names):
    embs = []
    for name in tqdm(names, desc="TextEmbed"):
        prompts = [t.format(name.replace("_"," ")) for t in promptTemplate] # insert class names
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        te = clip_model.get_text_features(**inputs)     # [T, D]
        te = te / te.norm(dim=-1, keepdim=True)
        embs.append(te.mean(dim=0))                     # [D]
    text = torch.stack(embs, dim=0)                     # [C, D]
    return text / text.norm(dim=-1, keepdim=True)


def build_head_and_optim(clip_model: CLIPModel):
    feat_dim = clip_model.config.projection_dim  # ViT-B/32 = 512
    head = nn.Linear(feat_dim, 102).to(device)

    lora_params = lora_injection(clip_model, target_names=cfg.lora_target)
    clip_model.to(device)

    # 2 parameter groups: LoRA and linear head
    optim = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": cfg.lr_head, "weight_decay": cfg.wd_head},
            {"params": lora_params, "lr": cfg.lr_lora, "weight_decay": cfg.wd_lora},
        ]
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda" and cfg.amp))
    return head, optim, scaler




def get_image_feats(images):
    # images are actual tensor now
    # assume it's preprocessed
    if isinstance(images, torch.Tensor):
        pixel_values = images.to(device, dtype=torch.float16 if (device == "cuda" and cfg.amp) else torch.float32)
    else:
        inputs = processor(images=images, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]

    feats = clip_model.get_image_features(pixel_values=pixel_values)  # [B, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def supervised_logits(feats):
    return head(feats)  # [B, 102]



def text_logits(feats):
    # perform cosine similarity with text embedding.
    return (feats @ text_embs.T) * clip_model.logit_scale.exp()


class LoRALinearLayer(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):

        super().__init__()
        self.base = base  # linear layer frozen for training LoRA parameters
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        dt = base.weight.dtype

        if r > 0:
            self.lora_A = nn.Linear(base.in_features, r, bias=False).to(dev, dtype=dt)
            self.lora_B = nn.Linear(r, base.out_features, bias=False).to(dev, dtype=dt)
            self.dropout = nn.Dropout(dropout)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B.weight)  # set B to 0, avoid any bias introduced.
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

            # Frozen
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        if self.r > 0:
            return self.base(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
        else:
            return self.base(x)


# LoRA Injection with warped LoRA layer shown above.

def lora_injection(clip_model: nn.Module, target_names=("q_proj", "k_proj", "v_proj", "out_proj")):
    """
    """
    assert isinstance(clip_model.vision_model, CLIPVisionModel.__mro__[0].__class__) or hasattr(clip_model,
                                                                                                "vision_model")
    lora_params = []
    for name, module in clip_model.vision_model.named_modules():
        # injection to clip/transformer attention layer: q_proj/k_proj/v_proj/out_proj
        for t in target_names:
            if hasattr(module, t):
                lin = getattr(module, t)
                if isinstance(lin, nn.Linear):
                    lora_lin = LoRALinearLayer(lin, r=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
                    setattr(module, t, lora_lin)
                    lora_params += list(lora_lin.lora_A.parameters()) + list(lora_lin.lora_B.parameters())
    # Freeze the parameters
    for p in clip_model.vision_model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True
    return lora_params


class EarlyStopper:
    def __init__(self):
        self.counter = 0
        self.last_loss = 0
        self.patience = cfg.early_stop_patience
        self.enable = cfg.early_stopping
        self.delta = cfg.early_stop_minimum_improvement

    def report(self, loss):
        if self.last_loss - loss < self.delta:
            self.counter += 1
        else:
            self.counter = 0
        self.last_loss = loss

    def stop_flag(self):
        return self.enable and (self.counter >= self.patience)


def run_epoch(loader: DataLoader, train: bool = True):
    if train:
        head.train()
        clip_model.train()
    else:
        head.eval()
        clip_model.eval()

    total, correct_cls, correct_txt = 0, 0, 0
    loss_sum = 0.0
    for images, labels in tqdm(loader, desc="Train" if train else "Eval"):
        labels = labels.to(device)
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda" and cfg.amp)):
            feats = get_image_feats(images)  # [B, D]

            logits_cls = supervised_logits(feats)  # logits of classification score from linear layer head
            loss_cls = ce(logits_cls, labels)

            logits_txt = text_logits(feats)  # logits of text embedding trained in transformer
            loss_txt = ce(logits_txt, labels)
            # Alignment between text and img.

            loss = loss_cls + cfg.lambda_text * loss_txt  # weighted

        if train:
            # backward propagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Stats

        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
        correct_cls += (logits_cls.argmax(dim=-1) == labels).sum().item()
        correct_txt += (logits_txt.argmax(dim=-1) == labels).sum().item()

    loss_avg = loss_sum / total

    return {
        "loss": loss_avg,
        "acc_cls": correct_cls / total,  # 线性头准确率
        "acc_txt": correct_txt / total,  # 文本读出准确率（zero-shot 风格）
    }



if __name__ == '__main__':
    train_set = CacheDataset(split="train")
    val_set = CacheDataset(split="val")
    test_set = CacheDataset(split="test")

    classname = val_set.classes

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
        # workers should be 4, but got problems in notebook
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    head, optimizer, scaler = build_head_and_optim(clip_model)
    text_embs = build_text_embeddings(classname)
    ce = torch.nn.CrossEntropyLoss()

    best_val = -1.0
    best_head = None
    earlystop = EarlyStopper()
    for ep in range(1, cfg.epochs + 1):
        training = run_epoch(train_loader, train=True)
        val = run_epoch(val_loader, train=False)
        print(f"[{ep}/{cfg.epochs}] "
              f"Train: loss={training['loss']:.4f} acc_cls={training['acc_cls']:.4f} acc_txt={training['acc_txt']:.4f} | "
              f"Val:   loss={val['loss']:.4f} acc_cls={val['acc_cls']:.4f} acc_txt={val['acc_txt']:.4f}")

        if val["acc_cls"] > best_val:
            best_val = val["acc_cls"]
            best_head = {k: v.detach().cpu() for k, v in
                         head.state_dict().items()}  # Detach the parameters from autograd (keeps weights only)
        earlystop.report(val['loss'])
        if earlystop.stop_flag():
            print(f"Early stop triggered...Exiting on epoch {ep}")
            break

    if best_head is not None:
        head.load_state_dict({k: v.to(device) for k, v in best_head.items()})
    te = run_epoch(test_loader, train=False)
    print(f"Test: loss={te['loss']:.4f}  acc_cls={te['acc_cls']:.4f}  acc_txt={te['acc_txt']:.4f}")

