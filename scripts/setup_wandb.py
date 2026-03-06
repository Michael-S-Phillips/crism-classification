"""
Interactive W&B setup. Run once before training.

Usage:
    conda run -n crism python scripts/setup_wandb.py
"""
import os, sys, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    print("=== Weights & Biases Setup ===")
    print("You need a free W&B account at https://wandb.ai")
    print()

    api_key = input("Paste your W&B API key (from https://wandb.ai/authorize): ").strip()
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)

    wandb.login(key=api_key)

    entity = input("W&B username or team name (leave blank for default): ").strip() or None

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config.yaml'
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg['wandb']['entity'] = entity
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\nW&B configured: project=crism-mineral-classification, entity={entity}")
    print("Test with: conda run -n crism python -c \"import wandb; wandb.init(project='crism-mineral-classification')\"")

if __name__ == '__main__':
    main()
