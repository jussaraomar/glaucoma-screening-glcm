
# glcm/src/train_eyepacs_resnet50.py

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from pathlib import Path

from glcm.utils.data_loader_eyepacs import get_dataloaders_eyepacs
from glcm.config.params import LEARNING_RATE, EPOCHS


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, _, class_to_idx = get_dataloaders_eyepacs()
    print("class_to_idx:", class_to_idx)

    # class weights from dataset
    train_ds = train_loader.dataset
    counts = [0, 0]
    for _, y in train_ds.samples:
        counts[y] += 1
    print("Train class counts:", counts)

    # model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # weighted loss
    w0 = 1.0 / counts[0]
    w1 = 1.0 / counts[1]
    weights = torch.tensor([w0, w1], device=device)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Loss weights:", weights.detach().cpu().numpy())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    #  Mixed precision scaler 
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path("glcm/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "resnet50_eyepacs_best.pt"
    last_path = save_dir / "resnet50_eyepacs_last.pt"

    best_val_acc = 0.0
    patience = 3
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"EyePACS Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            # Mixed precision backward + optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=float(loss.item()), acc=float(correct / total))

        train_loss = running_loss / total
        train_acc = correct / total

        # validate 
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST to {best_path} (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no val improvement for {patience} epochs).")
                break

    torch.save(model.state_dict(), last_path)
    print(f"Saved LAST to {last_path}")
    print(f"Best validation accuracy was {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
