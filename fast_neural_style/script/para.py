"""Report parameters in a metadata-bearing Ti60 checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    from neural_style.transformer_net import load_checkpoint
    model, metadata = load_checkpoint(args.model)
    print(f"Metadata: {metadata}")
    print(f"Trainable parameters (excludes BN buffers): {sum(p.numel() for p in model.parameters())}")
    print(f"Folded inference parameters: {sum(p.numel() for p in model.fused().parameters())}")


if __name__ == "__main__":
    main()
