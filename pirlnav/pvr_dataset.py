import torch
import zarr
import numpy as np

from habitat.core.registry import registry

from torch.utils.data import Dataset, DataLoader, IterableDataset


class ZarrDataset(IterableDataset):
    """IterableDataset for zarr arrays. This a) doesn't require loading the entire
    dataset into memory, and b) prevents random access, which is apparently slow
    for zarr."""

    def __init__(self, zarr_array_dict, start=None, end=None):
        self._zarr_array_dict = zarr_array_dict
        total_length = zarr_array_dict[next(iter(zarr_array_dict))].shape[0]
        self._start = start or 0
        self._end = end or total_length

    def __len__(self):
        return self._end - self._start

    def __iter__(self):
        generators = {
            k: v.islice(self._start, self._end)
            for k, v in self._zarr_array_dict.items()
        }
        for _ in range(len(self)):
            yield {k: next(v) for k, v in generators.items()}


class DictDataset(Dataset):
    """StackDataset is not available in PyTorch 1.12.1, so implementing here."""

    def __init__(self, array_dict):
        # array_dict should map names to numpy arrays
        self._array_dict = array_dict

    def __len__(self):
        return len(self._array_dict[next(iter(self._array_dict))])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._array_dict.items()}


def get_pvr_dataset(
    pvr_dataset_path,
    nv_dataset_path,
    pvr_keys=None,
    nv_keys=None,
    in_memory=False,
):
    pvr_dataset = zarr.open(pvr_dataset_path, mode="r")
    nv_dataset = zarr.open(nv_dataset_path, mode="r")

    pvr_keys = list(pvr_dataset["data"].keys()) if pvr_keys is None else pvr_keys
    nv_keys = list(nv_dataset["data"].keys()) if nv_keys is None else nv_keys

    # If loading into memory, convert to numpy arrays:
    f = np.array if in_memory else lambda x: x
    pvr_arrays = {k: f(v) for k, v in pvr_dataset["data"].items() if k in pvr_keys}
    nv_arrays = {k: f(v) for k, v in nv_dataset["data"].items() if k in nv_keys}

    arrays = {**pvr_arrays, **nv_arrays}

    dataset_cls = DictDataset if in_memory else ZarrDataset
    return dataset_cls(arrays)


if __name__ == "__main__":
    dataset = get_pvr_dataset(
        "pvr_data/one_percent/clip_data",
        "pvr_data/one_percent/non_visual_data",
        pvr_keys=["last_two_hidden_layers"],
        nv_keys=None,
        in_memory=False,
    )

    dataloader = DataLoader(dataset, batch_size=8)

    for batch in dataloader:
        print(f"compass {batch['compass']}")
