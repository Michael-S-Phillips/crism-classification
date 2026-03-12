import torch
import numpy as np
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MRRAL_DIR = '/mnt/crism/MRDR'
HDR_FILES = sorted(glob.glob(os.path.join(MRRAL_DIR, 'mc*/t*mrral*.hdr')))


def test_dataset_yields_correct_shape():
    """Each yielded patch should be (7, 7, 59) float32."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(HDR_FILES[:5], patch_size=7)
    it = iter(ds)
    patch = next(it)
    assert patch.shape == (7, 7, 59), f"Got {patch.shape}"
    assert patch.dtype == torch.float32


def test_dataset_values_clipped():
    """Values should be in [0.0, 0.5] after normalization."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(HDR_FILES[:5], patch_size=7)
    it = iter(ds)
    for _ in range(20):
        patch = next(it)
        assert patch.min().item() >= 0.0, f"Min below 0: {patch.min()}"
        assert patch.max().item() <= 0.5, f"Max above 0.5: {patch.max()}"


def test_dataset_no_nodata_in_output():
    """No 65535 values should appear in yielded patches."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(HDR_FILES[:5], patch_size=7)
    it = iter(ds)
    for _ in range(10):
        patch = next(it)
        assert not (patch == 65535.0).any(), "Nodata value 65535 found in output"


def test_dataloader_multiworker():
    """DataLoader with num_workers=2 should yield patches without error."""
    from data.global_patch_dataset import CRISMGlobalPatchDataset
    from torch.utils.data import DataLoader
    ds = CRISMGlobalPatchDataset(HDR_FILES[:10], patch_size=7)
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    batch = next(iter(loader))
    assert batch.shape == (4, 7, 7, 59)
