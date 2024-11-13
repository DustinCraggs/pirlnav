from bisect import bisect_left
import json
import torch
import zarr
import itertools
import numpy as np

from habitat.core.registry import registry

from torch.utils.data import Dataset, DataLoader, IterableDataset


class ZarrDataset(IterableDataset):
    """IterableDataset for zarr arrays. This a) doesn't require loading the entire
    dataset into memory, and b) prevents random access, which is apparently slow
    for zarr."""

    def __init__(self, zarr_array_dict, start=None, end=None, auto_reset=True):
        self._zarr_array_dict = zarr_array_dict
        total_length = zarr_array_dict[next(iter(zarr_array_dict))].shape[0]
        self._start = start or 0
        self._end = end or total_length
        self._auto_reset = auto_reset

    def __len__(self):
        return self._end - self._start

    def __iter__(self):
        generators = self._get_generators()

        while True:
            try:
                yield {k: next(v) for k, v in generators.items()}
            except StopIteration:
                if self._auto_reset:
                    generators = self._get_generators()
                else:
                    raise

    def _get_generators(self):
        return {
            k: v.islice(self._start, self._end)
            for k, v in self._zarr_array_dict.items()
        }


class DictDataset(Dataset):
    """StackDataset is not available in PyTorch 1.12.1, so implementing here."""

    def __init__(self, array_dict):
        # array_dict should map names to numpy arrays
        self._array_dict = array_dict

    def __len__(self):
        return len(self._array_dict[next(iter(self._array_dict))])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._array_dict.items()}


class BatchedCyclicIterableDataset(IterableDataset):
    """Iterates over multiple IterableDatasets in parallel, yielding batches containing
    one sample from each dataset."""

    def __init__(self, datasets):
        self._datasets = datasets

    def __len__(self):
        return min(len(dataset) for dataset in self._datasets)

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self._datasets]
        batch = []
        while True:
            for i, iterator in enumerate(iterators):
                try:
                    batch.append(next(iterator))
                except StopIteration:
                    iterators[i] = iter(self._datasets[i])
                    batch.append(next(iterator))
            yield {k: torch.stack([x[k] for x in batch]) for k in batch[0]}


def create_pvr_dataset_splits(
    pvr_dataset_path,
    nv_dataset_path,
    num_splits=1,
    pvr_keys=None,
    nv_keys=None,
):
    # This is probably overcomplicated given that the number of steps would be pretty
    # similar even if splitting by episode.
    with open(f"{pvr_dataset_path}/ep_index.json") as f:
        ep_index = json.load(f)

    dataset = get_pvr_dataset(pvr_dataset_path, nv_dataset_path, pvr_keys, nv_keys)

    # Need to divide dataset into episodes (i.e. episodes should not be broken across
    # multiple splits):
    ep_boundaries = [*[ep_info["row"] for ep_info in ep_index], len(dataset)]
    target_split_length = len(dataset) // num_splits
    target_end_rows = [target_split_length * (i + 1) for i in range(num_splits)]

    end_rows = [
        ep_boundaries[bisect_left(ep_boundaries, target)] for target in target_end_rows
    ]

    datasets = []
    start_row = 0
    print(f"Total length: {len(dataset)}")
    for end_row in end_rows:
        print(f"start_row {start_row}, end_row {end_row}")
        dataset = get_pvr_dataset(
            pvr_dataset_path,
            nv_dataset_path,
            pvr_keys,
            nv_keys,
            start_idx=start_row,
            end_idx=end_row,
        )
        datasets.append(dataset)
        start_row = end_row

    return datasets


def print_dataset_head(
    pvr_dataset_path,
    nv_dataset_path,
    num_splits=1,
    pvr_keys=None,
    nv_keys=None,
):
    with open(f"{pvr_dataset_path}/ep_index.json") as f:
        ep_index = json.load(f)

    dataset = get_pvr_dataset(pvr_dataset_path, nv_dataset_path, pvr_keys, nv_keys)
    print(ep_index[:50])

    for i, x in zip(range(3000), dataset):
        print(f"step {i}")
        print(f"done {x['done']}")
        print(f"objectgoal {x['objectgoal']}")
        print(f"reward {x['reward']}")
        print(f"compass {x['compass']}")
        print(f"gps {x['gps']}")
        print(f"mean_pvr {x['last_two_hidden_layers'].mean()}")

        print()


def get_pvr_dataset(
    pvr_dataset_path,
    nv_dataset_path,
    pvr_keys=None,
    nv_keys=None,
    start_idx=None,
    end_idx=None,
):
    pvr_dataset = zarr.open(pvr_dataset_path, mode="r")
    nv_dataset = zarr.open(nv_dataset_path, mode="r")

    pvr_keys = list(pvr_dataset["data"].keys()) if pvr_keys is None else pvr_keys
    nv_keys = list(nv_dataset["data"].keys()) if nv_keys is None else nv_keys

    pvr_arrays = {k: v for k, v in pvr_dataset["data"].items() if k in pvr_keys}
    nv_arrays = {k: v for k, v in nv_dataset["data"].items() if k in nv_keys}

    arrays = {**pvr_arrays, **nv_arrays}

    dataset = ZarrDataset(arrays, start=start_idx, end=end_idx)
    return dataset


if __name__ == "__main__":
    dataset = get_pvr_dataset(
        "pvr_data/one_percent/clip_data",
        "pvr_data/one_percent/non_visual_data",
        pvr_keys=["last_two_hidden_layers"],
        nv_keys=None,
    )

    dataloader = DataLoader(dataset, batch_size=8)

    for batch in dataloader:
        print(f"compass {batch['compass']}")
