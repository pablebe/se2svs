"""Download se2svs fine-tuned model checkpoints from Hugging Face.

Collection: https://hf.co/collections/pablebe/se2svs-model-checkpoints

Downloads all checkpoints into checkpoints/se2svs/<model>/ so that the
folder structure matches what the other scripts expect.

Usage:
    python download_checkpoints.py
    python download_checkpoints.py --token <hf_token>   # for private repos
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

# Maps (hf_repo_id, filename) -> local destination directory
CHECKPOINTS = [
    ("pablebe/se2svs_sgm_full",       "epoch=900-sdr=8.35.ckpt",  "checkpoints/se2svs/sgm_full"),
    # Note: base pretrained models (sgmse_pretrained, bsrnn_pretrained) are downloaded separately;
    # see README for links.
    ("pablebe/se2svs_sgm_lora_r16",   "epoch=533-sdr=7.63.ckpt",  "checkpoints/se2svs/sgm_lora_r16"),
    ("pablebe/se2svs_sgm_scratch",    "epoch=522-sdr=7.26.ckpt",  "checkpoints/se2svs/sgm_scratch"),
    ("pablebe/se2svs_bsrnn_full",     "epoch=378-sdr=10.03.ckpt", "checkpoints/se2svs/bsrnn_full"),
    ("pablebe/se2svs_bsrnn_lora_r16", "epoch=503-sdr=8.94.ckpt",  "checkpoints/se2svs/bsrnn_lora_r16"),
    ("pablebe/se2svs_bsrnn_lora_r32",  "epoch=544-sdr=9.05.ckpt",  "checkpoints/se2svs/bsrnn_lora_r32"),
    ("pablebe/se2svs_bsrnn_lora_r128","epoch=489-sdr=9.43.ckpt",  "checkpoints/se2svs/bsrnn_lora_r128"),
    ("pablebe/se2svs_bsrnn_scratch",  "epoch=480-sdr=8.25.ckpt",  "checkpoints/se2svs/bsrnn_scratch"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download se2svs checkpoints from Hugging Face")
    parser.add_argument("--token", default=None, help="Hugging Face access token (for private repos)")
    args = parser.parse_args()

    for repo_id, filename, local_dir in CHECKPOINTS:
        dest = Path(local_dir) / filename
        if dest.exists():
            print(f"[skip] {dest} already exists")
            continue

        print(f"[download] {repo_id}/{filename} -> {dest}")
        os.makedirs(local_dir, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            token=args.token,
        )
        print(f"[done]  {dest}")


if __name__ == "__main__":
    main()
