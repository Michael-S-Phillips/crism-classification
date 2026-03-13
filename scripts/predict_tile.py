"""
Run inference on a full CRISM mrrsu tile and write per-class GeoTIFF outputs.

Usage:
    conda run -n crism python scripts/predict_tile.py \
        --tile_id t0503 \
        --model logreg \
        --checkpoint checkpoints/logreg_model.pkl
"""
import os, sys, argparse, pickle, logging
import numpy as np
import rasterio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.extract_pixels import NODATA_VALUE
from data.label_parser import CLASSES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SKLEARN_MODELS = {'logreg', 'svc', 'rf', 'xgb', 'lgbm'}
TORCH_PIXEL_MODELS = {'mlp'}
TORCH_PATCH_MODELS = {'cnn', 'vit'}


def predict_tile(
    tile_id: str,
    mrrsu_path: str,
    checkpoint_path: str,
    model_type: str,
    output_dir: str,
    patch_size: int = 7,
    batch_size: int = 4096,
):
    """
    Predict mineral probabilities for all valid pixels in a mrrsu tile.
    Writes one GeoTIFF per class + a best_class.tif.
    """
    os.makedirs(os.path.join(output_dir, tile_id), exist_ok=True)

    logger.info(f"Loading tile: {mrrsu_path}")
    with rasterio.open(mrrsu_path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)
        profile = src.profile.copy()
        H, W = src.height, src.width

    # Impute NaN bands to 0.0 (e.g. BD640_2 / band 5 is undefined in CRISM MRDR)
    np.nan_to_num(data, nan=0.0, copy=False)

    # Exclude pixels where any band is NODATA (65535)
    nodata_mask = (data >= NODATA_VALUE).any(axis=0)
    valid_mask = ~nodata_mask  # (H, W)
    rows, cols = np.where(valid_mask)
    X = data[:, rows, cols].T  # (n_valid, n_bands)

    logger.info(f"Valid pixels: {len(rows):,} / {H * W:,}")

    # --- Load model and predict ---
    if model_type in SKLEARN_MODELS:
        with open(checkpoint_path, 'rb') as f:
            models = pickle.load(f)
        y_score = _sklearn_predict(models, X, model_type)

    elif model_type in TORCH_PIXEL_MODELS:
        y_score = _torch_pixel_predict(checkpoint_path, model_type, X, batch_size)

    elif model_type in TORCH_PATCH_MODELS:
        y_score = _torch_patch_predict(
            checkpoint_path, model_type, mrrsu_path, rows, cols, patch_size, batch_size
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Write output GeoTIFFs ---
    out_profile = profile.copy()
    out_profile.update(count=1, dtype='float32', driver='GTiff')

    prob_volume = np.zeros((len(CLASSES), H, W), dtype=np.float32)
    prob_volume[:, rows, cols] = y_score.T

    tile_out_dir = os.path.join(output_dir, tile_id)
    for cls_idx, cls_name in enumerate(CLASSES):
        out_path = os.path.join(tile_out_dir, f'{cls_name}_prob.tif')
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(prob_volume[cls_idx:cls_idx + 1])

    # Best class map (argmax, 0-indexed)
    best_class = np.full((H, W), -1, dtype=np.int16)
    best_class[valid_mask] = prob_volume[:, valid_mask].argmax(axis=0)
    best_profile = out_profile.copy()
    best_profile.update(dtype='int16', nodata=-1)
    with rasterio.open(os.path.join(tile_out_dir, 'best_class.tif'), 'w', **best_profile) as dst:
        dst.write(best_class[np.newaxis])

    logger.info(f"Predictions written to {tile_out_dir}")


def _sklearn_predict(models, X, model_type):
    if model_type in ('xgb', 'lgbm'):
        return np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    elif model_type == 'svc':
        m = models[0]
        raw = m.decision_function(X)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        return 1 / (1 + np.exp(-raw))
    else:
        proba_list = models[0].predict_proba(X)
        return np.stack([p[:, 1] for p in proba_list], axis=1)


def _torch_pixel_predict(checkpoint_path, model_type, X, batch_size):
    from models.mlp import MLP
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = MLP(n_features=X.shape[1], n_classes=len(CLASSES))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scores = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size])
        with torch.no_grad():
            scores.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(scores)


def _torch_patch_predict(checkpoint_path, model_type, mrrsu_path, rows, cols, patch_size, batch_size):
    from models.cnn import SpectralSpatialCNN
    from models.vit import SpectralViT

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    half = patch_size // 2

    if model_type == 'cnn':
        model = SpectralSpatialCNN(patch_size=patch_size)
    else:
        model = SpectralViT(patch_size=patch_size)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    scores = []
    with rasterio.open(mrrsu_path) as src:
        H, W = src.height, src.width
        for i in range(0, len(rows), batch_size):
            batch_patches = []
            for r, c in zip(rows[i:i + batch_size], cols[i:i + batch_size]):
                r0, r1 = max(0, r - half), min(H, r + half + 1)
                c0, c1 = max(0, c - half), min(W, c + half + 1)
                window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                patch = src.read(window=window).astype(np.float32)
                patch[patch >= NODATA_VALUE] = 0.0
                patch = np.nan_to_num(patch)
                full = np.zeros((src.count, patch_size, patch_size), np.float32)
                full[:, half - (r - r0):half - (r - r0) + patch.shape[1],
                        half - (c - c0):half - (c - c0) + patch.shape[2]] = patch
                batch_patches.append(full)
            xb = torch.tensor(np.stack(batch_patches))
            with torch.no_grad():
                scores.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_id', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--patch_size', type=int, default=7)
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    from config_loader import load_config
    cfg = load_config(cfg_path)

    from data.extract_pixels import find_tile_pairs
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}

    if args.tile_id not in mrrsu_map:
        print(f"Tile {args.tile_id} not found.")
        sys.exit(1)

    predict_tile(
        tile_id=args.tile_id,
        mrrsu_path=mrrsu_map[args.tile_id],
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        output_dir=cfg['predictions_dir'],
        patch_size=args.patch_size,
    )


if __name__ == '__main__':
    main()
