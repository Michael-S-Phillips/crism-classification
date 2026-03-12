import os, pytest, tempfile
import numpy as np

def test_predict_tile_produces_geotiffs(tmp_path):
    """Smoke test: run inference on the test tile, check outputs exist."""
    from scripts.predict_tile import predict_tile
    import glob

    # Use first test tile
    import pandas as pd
    from config_loader import load_config
    cfg = load_config()

    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'pixels.parquet'))
    test_tile = df[df['split'] == 'test']['tile_id'].iloc[0]

    from data.extract_pixels import find_tile_pairs
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}

    # Use a saved sklearn checkpoint
    ckpts = glob.glob(os.path.join(cfg['checkpoints_dir'], 'logreg_model.pkl'))
    if not ckpts:
        pytest.skip("No logreg checkpoint found. Run train.py --model logreg first.")

    out_dir = str(tmp_path / 'predictions')
    predict_tile(
        tile_id=test_tile,
        mrrsu_path=mrrsu_map[test_tile],
        checkpoint_path=ckpts[0],
        model_type='logreg',
        output_dir=out_dir,
    )

    from data.label_parser import CLASSES
    for cls in CLASSES:
        assert os.path.exists(os.path.join(out_dir, test_tile, f'{cls}_prob.tif'))
    assert os.path.exists(os.path.join(out_dir, test_tile, 'best_class.tif'))
