"""
Evaluate top-N model ensemble on the val or test split.

Usage:
    conda run -n crism python scripts/evaluate_ensemble.py \
        --checkpoints checkpoints/scnn_best_best.pt checkpoints/svit_best_best.pt

    # Evaluate val split
    conda run -n crism python scripts/evaluate_ensemble.py \
        --checkpoints checkpoints/scnn_best_best.pt --split val
"""
import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model_from_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    name = os.path.basename(ckpt_path)
    if 'scnn' in name or 'spectral_cnn' in name:
        from models.spectral_cnn import SpectralCNN1D
        model = SpectralCNN1D(n_bands=59, n_classes=5)
    elif 'svit' in name or 'spectral_vit' in name:
        from models.spectral_transformer import SpectralTransformer
        model = SpectralTransformer(n_bands=59, n_classes=5)
    elif 'cnn' in name:
        from models.cnn import SpectralSpatialCNN
        model = SpectralSpatialCNN(n_bands=60, n_classes=5, patch_size=7)
    elif 'mlp' in name:
        from models.mlp import MLP
        model = MLP(n_features=60, n_classes=5)
    else:
        raise ValueError(f"Cannot infer model type from filename: {name}")
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model.to(device)


def predict(model, df, device, batch_size=1024):
    from torch.utils.data import DataLoader
    from data.dataset import CRISMSpectralDataset, CRISMPixelDataset, MRRAL_BAND_COLS
    if MRRAL_BAND_COLS[0] in df.columns:
        ds = CRISMSpectralDataset(df)
    else:
        ds = CRISMPixelDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for feats, _, _ in loader:
            logits = model(feats.to(device))
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    args = parser.parse_args()

    from config_loader import load_config
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data (prefer mrral if available)
    mrral_path = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    mrrsu_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    if os.path.exists(mrral_path):
        df = pd.read_parquet(mrral_path)
        logging.info(f"Using mrral parquet: {mrral_path}")
    else:
        df = pd.read_parquet(mrrsu_path)
        logging.info(f"Using mrrsu parquet: {mrrsu_path}")
    test_df = df[df['split'] == args.split]
    logging.info(f"Split '{args.split}': {len(test_df)} pixels")

    from data.dataset import LABEL_COLS
    y_true = test_df[LABEL_COLS].values.astype('float32')
    conf_tiers = test_df['confidence_tier'].tolist()

    all_scores = []
    for ckpt_path in args.checkpoints:
        logging.info(f"Loading {ckpt_path}")
        model = load_model_from_checkpoint(ckpt_path, device)
        scores = predict(model, test_df, device)
        all_scores.append(scores)
        from evaluation.metrics import compute_full_metrics
        m = compute_full_metrics(y_true, scores, conf_tiers)
        logging.info(f"  {os.path.basename(ckpt_path)}: mAP={m['mAP']:.4f}")
        for cls, ap in m['per_class_ap'].items():
            logging.info(f"    {cls}: AP={ap:.4f}")

    if len(all_scores) > 1:
        ensemble_scores = np.mean(all_scores, axis=0)
        from evaluation.metrics import compute_full_metrics
        m = compute_full_metrics(y_true, ensemble_scores, conf_tiers)
        logging.info(f"\nEnsemble ({len(all_scores)} models): mAP={m['mAP']:.4f}")
        for cls, ap in m['per_class_ap'].items():
            logging.info(f"  {cls}: AP={ap:.4f}")


if __name__ == '__main__':
    main()
