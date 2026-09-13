"""Factory helpers for pluggable dataset and split contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

DatasetBuilder = Callable[[object, bool], Dataset]


def _load_index_manifest(path_value: str | None, *, field_name: str) -> list[int]:
    """Load a JSON index manifest used by the repair split contract."""
    if path_value is None:
        raise ValueError(f"Repair-only training requires `{field_name}` to identify the defect set.")

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Bug index manifest not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        indices = payload.get("indices")
    else:
        indices = payload
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError("Bug index manifest must be a JSON list of integers or `{\"indices\": [...]}`.")
    return indices


def _build_cifar10_dataset(cfg, train: bool) -> Dataset:
    """Build a CIFAR-10 dataset with torchvision."""
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required to build CIFAR-10 dataloaders."
        ) from exc

    mean = tuple(cfg.dataset.get("mean", [0.4914, 0.4822, 0.4465]))
    std = tuple(cfg.dataset.get("std", [0.2470, 0.2435, 0.2616]))
    image_size = int(cfg.dataset.input_resolution)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    data_root = cfg.dataset.get("data_root", "data")
    download = bool(cfg.dataset.get("download", False))
    return datasets.CIFAR10(root=data_root, train=train, download=download, transform=transform)


class _TinyImageNetValDataset(Dataset):
    """Tiny-ImageNet validation loader supporting annotation-file layout."""

    def __init__(self, root: Path, transform) -> None:
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        wnids_path = root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Tiny-ImageNet wnids file not found: {wnids_path}")
        class_names = [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        class_to_idx = {name: index for index, name in enumerate(class_names)}

        val_dir = root / "val"
        annotations_path = val_dir / "val_annotations.txt"
        images_dir = val_dir / "images"
        if annotations_path.exists() and images_dir.exists():
            for line in annotations_path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                image_path = images_dir / image_name
                if image_path.exists() and class_name in class_to_idx:
                    self.samples.append((image_path, class_to_idx[class_name]))
            return

        # Fallback: allow an already-reorganized ImageFolder-style validation tree.
        try:
            from torchvision import datasets
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torchvision is required to build Tiny-ImageNet datasets.") from exc
        image_folder = datasets.ImageFolder(val_dir, transform=transform)
        self.samples = [(Path(path), label) for path, label in image_folder.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class _TinyImageNetTrainDataset(Dataset):
    """Tiny-ImageNet train loader using wnids.txt class order for stable labels."""

    def __init__(self, root: Path, transform) -> None:
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        wnids_path = root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Tiny-ImageNet wnids file not found: {wnids_path}")
        class_names = [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        class_to_idx = {name: index for index, name in enumerate(class_names)}

        train_dir = root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {train_dir}")

        for class_name in class_names:
            class_dir = train_dir / class_name / "images"
            if not class_dir.exists():
                continue
            label = class_to_idx[class_name]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file():
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _build_tiny_imagenet_dataset(cfg, train: bool) -> Dataset:
    """Build a Tiny-ImageNet dataset."""
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required to build Tiny-ImageNet dataloaders."
        ) from exc

    mean = tuple(cfg.dataset.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg.dataset.get("std", [0.229, 0.224, 0.225]))
    image_size = int(cfg.dataset.input_resolution)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    data_root = Path(str(cfg.dataset.get("data_root", "data/tiny-imagenet-200")))
    if not data_root.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet data root not found: {data_root}. "
            "Set dataset.data_root to the extracted tiny-imagenet-200 directory."
        )

    train_root = data_root / "train"
    val_root = data_root / "val"
    test_root = data_root / "test"

    if train:
        return _TinyImageNetTrainDataset(data_root, transform=transform)

    if val_root.exists():
        return _TinyImageNetValDataset(data_root, transform=transform)

    # Smoke-test fallback for archives that only ship train/test without labeled val.
    # This keeps the pipeline runnable, but evaluation/bug manifests then come from train.
    if test_root.exists() and not val_root.exists():
        return datasets.ImageFolder(train_root, transform=transform)

    return _TinyImageNetValDataset(data_root, transform=transform)


def _build_gtsrb_dataset(cfg, train: bool) -> Dataset:
    """Build a GTSRB dataset with torchvision."""
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required to build GTSRB dataloaders."
        ) from exc

    mean = tuple(cfg.dataset.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg.dataset.get("std", [0.229, 0.224, 0.225]))
    image_size = int(cfg.dataset.input_resolution)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    data_root = str(cfg.dataset.get("data_root", "data"))
    download = bool(cfg.dataset.get("download", False))
    split = "train" if train else "test"
    return datasets.GTSRB(root=data_root, split=split, download=download, transform=transform)


def _build_imagefolder_dataset(cfg, train: bool) -> Dataset:
    """Build a generic ImageFolder-style dataset with train/eval subdirectories."""
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required to build ImageFolder dataloaders."
        ) from exc

    mean = tuple(cfg.dataset.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg.dataset.get("std", [0.229, 0.224, 0.225]))
    image_size = int(cfg.dataset.input_resolution)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    data_root = Path(str(cfg.dataset.get("data_root", "data/imagefolder_dataset")))
    if not data_root.exists():
        raise FileNotFoundError(f"ImageFolder data root not found: {data_root}")

    train_split = str(cfg.dataset.get("train_split", "train"))
    eval_split = str(cfg.dataset.get("clean_split", "test"))
    split_dir = data_root / (train_split if train else eval_split)
    if not split_dir.exists():
        raise FileNotFoundError(f"ImageFolder split directory not found: {split_dir}")

    return datasets.ImageFolder(str(split_dir), transform=transform, allow_empty=True)


def _build_tabular_pt_dataset(cfg, train: bool) -> Dataset:
    """Build a tabular dataset from serialized tensor payloads."""
    from torch.utils.data import TensorDataset

    data_root = Path(str(cfg.dataset.get("data_root", "data/tabular_dataset")))
    if not data_root.exists():
        raise FileNotFoundError(f"Tabular data root not found: {data_root}")

    train_file = str(cfg.dataset.get("train_file", "train.pt"))
    eval_file = str(cfg.dataset.get("eval_file", "eval.pt"))
    split_path = data_root / (train_file if train else eval_file)
    if not split_path.exists():
        raise FileNotFoundError(f"Tabular split file not found: {split_path}")

    payload = torch.load(split_path, map_location="cpu")
    if isinstance(payload, dict):
        inputs = payload.get("inputs")
        labels = payload.get("labels")
    else:
        inputs, labels = payload
    if inputs is None or labels is None:
        raise ValueError(
            f"Tabular payload at {split_path} must provide `inputs` and `labels`."
        )

    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(1)
    return TensorDataset(inputs, labels)


DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "cifar10": _build_cifar10_dataset,
    "gtsrb": _build_gtsrb_dataset,
    "imagefolder": _build_imagefolder_dataset,
    "lisa_signs": _build_imagefolder_dataset,
    "tiny_imagenet": _build_tiny_imagenet_dataset,
    "tiny-imagenet": _build_tiny_imagenet_dataset,
    "tt100k_signs": _build_imagefolder_dataset,
    "tsinghua_traffic_light": _build_imagefolder_dataset,
    "tabular_pt": _build_tabular_pt_dataset,
    "acas_xu_tabular": _build_tabular_pt_dataset,
}


def register_dataset_builder(name: str, builder: DatasetBuilder) -> None:
    """Register a new dataset builder."""
    DATASET_BUILDERS[name.lower()] = builder


def build_dataset(cfg, train: bool) -> Dataset:
    """Build the configured dataset split."""
    dataset_name = str(cfg.dataset.name).lower()
    try:
        builder = DATASET_BUILDERS[dataset_name]
    except KeyError as exc:
        raise NotImplementedError(f"Dataset builder not implemented for: {cfg.dataset.name}") from exc
    return builder(cfg, train)


def _subset(dataset: Dataset, indices: list[int] | None) -> Dataset:
    """Return a dataset subset if indices are provided."""
    if indices is None:
        return dataset
    return Subset(dataset, indices)


def _loader_kwargs(num_workers: int, *, shuffle: bool) -> dict:
    """Return DataLoader kwargs tuned for GPU training throughput."""
    workers = max(int(num_workers), 0)
    kwargs = {
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def build_repair_dataloaders(cfg) -> dict[str, DataLoader]:
    """Build dataloaders for repair-only training and evaluation."""
    train_dataset = build_dataset(cfg, train=True)
    test_dataset = build_dataset(cfg, train=False)

    support_indices = _load_index_manifest(cfg.data.get("bug_indices_path"), field_name="data.bug_indices_path")
    bug_train_path = cfg.data.get("bug_train_indices_path")
    bug_train_indices = (
        _load_index_manifest(bug_train_path, field_name="data.bug_train_indices_path")
        if bug_train_path is not None
        else support_indices
    )
    bug_eval_path = cfg.data.get("bug_val_indices_path")
    if bug_eval_path is None:
        bug_eval_path = cfg.data.get("bug_eval_indices_path")
    bug_eval_field = "data.bug_val_indices_path" if cfg.data.get("bug_val_indices_path") is not None else "data.bug_eval_indices_path"
    bug_eval_indices = (
        _load_index_manifest(bug_eval_path, field_name=bug_eval_field)
        if bug_eval_path is not None
        else support_indices
    )
    clean_eval_indices = cfg.data.get("clean_eval_indices_path")
    clean_eval_subset = None
    if clean_eval_indices is not None:
        clean_eval_subset = _load_index_manifest(clean_eval_indices, field_name="data.clean_eval_indices_path")

    batch_size = int(cfg.train_loop.batch_size)
    eval_batch_size = int(cfg.evaluation.get("batch_size", batch_size))
    num_workers = int(cfg.runtime.get("num_workers", 0))

    bug_train_dataset = _subset(test_dataset, bug_train_indices)
    bug_eval_dataset = _subset(test_dataset, bug_eval_indices)
    clean_train_dataset = train_dataset
    clean_eval_dataset = _subset(test_dataset, clean_eval_subset)

    return {
        "bug_train": DataLoader(bug_train_dataset, batch_size=batch_size, **_loader_kwargs(num_workers, shuffle=True)),
        "bug_eval": DataLoader(bug_eval_dataset, batch_size=eval_batch_size, **_loader_kwargs(num_workers, shuffle=False)),
        "clean_train": DataLoader(clean_train_dataset, batch_size=eval_batch_size, **_loader_kwargs(num_workers, shuffle=True)),
        "clean_eval": DataLoader(clean_eval_dataset, batch_size=eval_batch_size, **_loader_kwargs(num_workers, shuffle=False)),
    }


def build_classification_dataloaders(cfg) -> dict[str, DataLoader]:
    """Build standard train/eval dataloaders for backbone classification."""
    train_dataset = build_dataset(cfg, train=True)
    test_dataset = build_dataset(cfg, train=False)

    batch_size = int(cfg.train_loop.batch_size)
    eval_batch_size = int(cfg.evaluation.get("batch_size", batch_size))
    num_workers = int(cfg.runtime.get("num_workers", 0))

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, **_loader_kwargs(num_workers, shuffle=True)),
        "eval": DataLoader(test_dataset, batch_size=eval_batch_size, **_loader_kwargs(num_workers, shuffle=False)),
    }
