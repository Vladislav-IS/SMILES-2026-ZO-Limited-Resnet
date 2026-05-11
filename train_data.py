from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import numpy as np

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY=True
MAX_SAMPLES=8192
N_CLASSES=100
SEED=0xDEAD

def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY, # True
        download=True,
        transform=get_transforms(train=True),
    )
    class_indices = {cls: [] for cls in range(N_CLASSES)}
    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)
    samples_per_class = MAX_SAMPLES // N_CLASSES
    remainder = MAX_SAMPLES % N_CLASSES
    balanced = []
    for cls in range(N_CLASSES):
        n = samples_per_class + (1 if cls < remainder else 0)
        balanced += class_indices[cls][:n]
    np.random.seed(SEED)
    np.random.shuffle(balanced)
    train_dataset = Subset(dataset, balanced)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train
    )

    return train_dataset, train_loader
