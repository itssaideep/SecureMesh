#!/usr/bin/env python3
# scripts/run_all_ablations.py
"""Automated execution of the SecureMesh-SCE Paper Ablation Suite.

Runs the complete 5x5 Attacker x Defender matrix across multiple seeds,
saving structured JSONL, CSV, and summary JSON files to experiments/results/.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from securemesh_sce.experiments.matrix_runner import run_matrix_experiment


def main():
    parser = argparse.ArgumentParser(description="Run SecureMesh-SCE Ablation Suite")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes per cell")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202], help="List of seeds")
    parser.add_argument("--duration", type=int, default=50, help="Steps per episode")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Results directory")

    args = parser.parse_args()

    print("\n=======================================================")
    print("      SecureMesh-SCE: Paper Ablation Study Suite       ")
    print("=======================================================")
    print(f"Episodes: {args.episodes} | Steps: {args.duration} | Seeds: {args.seeds}")
    print(f"Output Directory: {args.output_dir}\n")

    results = run_matrix_experiment(
        episodes=args.episodes,
        seeds=args.seeds,
        duration=args.duration,
        output_dir=args.output_dir,
    )

    print("\n[SUCCESS] All ablation cells completed successfully.")


if __name__ == "__main__":
    main()
