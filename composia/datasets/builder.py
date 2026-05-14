from .mixed_dataset import MixedDataset
from .nuscenes_dataset import nuScenesDataset


__dataset_cls__ = {
    "nuScenesDataset": nuScenesDataset,
    "MixedDataset": MixedDataset,
}

__all__ = ["__dataset_cls__"]
