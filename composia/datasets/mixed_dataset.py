import random
from typing import Any, Dict
from torch.utils.data import Dataset
from itertools import accumulate
from videox_fun.utils.utils import import_cls                     


class MixedDataset(Dataset):
    """
    A wrapper that combines multiple datasets behind one loader interface.
    """
    def __init__(
        self,
        dataset_config,
        mode: str = "train",
    ):
        super().__init__()
        self.dataset_cls = []
        dataset_lens = []
        shared_kwargs = dataset_config.get("kwargs", {})
        sub_dataset_names = dataset_config.get("sub_dataset_names", [])
        for name in sub_dataset_names:
            data_cfg = dataset_config.get(name, {})
            kwargs = data_cfg.get("kwargs", {})
            kwargs.update(shared_kwargs)
            kwargs["mode"] = mode
            data_cls = import_cls(data_cfg["type"])(**kwargs)
            dataset_lens.append(len(data_cls))
            self.dataset_cls.append(data_cls)
        
        self.dataset_lens = list(accumulate(dataset_lens))
        self.length = self.dataset_lens[-1]
        self.camera_names = shared_kwargs.get("camera_names", [])
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, kwargs) -> Dict[str, Any]:
        """
        Get a video sample.
        """
        idx = kwargs.get("idx", 0) % self.length
        # Locate the owning sub-dataset for this global index.
        dataset_idx = next(i for i, v in enumerate(self.dataset_lens) if v > idx)
        if dataset_idx > 0:
            idx -= self.dataset_lens[dataset_idx - 1]

        ## multi-res training
        validation_mode = kwargs.get("validation_mode", None)
        # if validation_mode is not None:
        conditions = kwargs.get("conditions", [])
        # else:
        #     resolution = kwargs["resolution"]
        #     # resolution: [T, H, W]
        #     _, H_sample, W_sample = resolution
        #     if H_sample == 256:
        #         conditions = kwargs.get("low_conditions", [])
        #     else:
        #         conditions = kwargs.get("high_conditions", [])
        #     kwargs["conditions"] = conditions
        # If the selected dataset does not support all requested conditions,
        # fall back to another dataset that does.
        if not all([cond in self.dataset_cls[dataset_idx].valid_conditions for cond in conditions]):
            valid_dataset_indices = [
                i for i, ds in enumerate(self.dataset_cls) if all([cond in ds.valid_conditions for cond in conditions])
            ]
            if len(valid_dataset_indices) == 0:
                raise ValueError(f"No dataset found with conditions: {conditions}")
            dataset_idx = random.choice(valid_dataset_indices)
            idx = random.randint(0, len(self.dataset_cls[dataset_idx]) - 1)
        
        kwargs["idx"] = idx
        return self.dataset_cls[dataset_idx].__getitem__(kwargs)
