# glcm/utils/data_loader_eyepacs.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from glcm.config.params import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from glcm.config.paths import EYEPACS_TRAIN_DIR, EYEPACS_VAL_DIR, EYEPACS_TEST_DIR


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])


def _imagefolder_with_rg_nrg(path, train: bool):
    """
    EyePACS folders may be named RG/NRG or rg/nrg.
    ImageFolder reads folder names as class names automatically.
    """
    return datasets.ImageFolder(str(path), transform=get_transforms(train=train))


def get_datasets_eyepacs():
    train_ds = _imagefolder_with_rg_nrg(EYEPACS_TRAIN_DIR, train=True)
    val_ds   = _imagefolder_with_rg_nrg(EYEPACS_VAL_DIR, train=False)
    test_ds  = _imagefolder_with_rg_nrg(EYEPACS_TEST_DIR, train=False)
    return train_ds, val_ds, test_ds


def get_dataloaders_eyepacs():
    train_ds, val_ds, test_ds = get_datasets_eyepacs()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


    return train_loader, val_loader, test_loader, train_ds.class_to_idx
