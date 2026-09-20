#!/usr/bin/env python3
# scripts/run_hardware_suite.py
"""Automated runner for Scenarios 1 to 5 on physical ESP8266 hardware.

Waits for any currently running experiment to finish, then executes
all 5 scenarios sequentially with the physical hardware bridge enabled,
and aggregates the results into a comparative report.

Usage:
    python scripts/run_hardware_suite.py [--episodes 5] [--wait-pid PID]
"""

import os
import sys
import time
import json
import psutil
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from securemesh_sce.experiments.runner import run_experiment


SCENARIOS = [
    ("SCE-001", "experiments/scenarios/01_baseline.yaml"),
    ("SCE-002", "experiments/scenarios/02_rl_attacker.yaml"),
    ("SCE-003", "experiments/scenarios/03_rl_defender.yaml"),
    ("SCE-004", "experiments/scenarios/04_co_evolution.yaml"),
    ("SCE-005", "experiments/scenarios/05_llm_vs_llm.yaml"),
]



def wait_for_process(pid: int):
    """Wait until a specific PID has terminated."""
    if not pid:
        return
    try:
        proc = psutil.Process(pid)
        print(f"[*] Waiting for existing experiment (PID {pid}: {proc.name()}) to finish...")
        while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            time.sleep(2)
        print(f"[+] Process {pid} has completed.\n")
    except psutil.NoSuchProcess:
        print(f"[*] Process {pid} has already finished.\n")
    except Exception as e:
        print(f"[!] Warning waiting for process {pid}: {e}\n")


def generate_comparison_report(results: dict, output_path: Path):
    """Generate a clean markdown report comparing all 5 scenarios."""
    lines = [
        "# SecureMesh-SCE: Physical Hardware Experiment Suite (Scenarios 1-5)",
        f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Hardware Device:** ESP8266 NodeMCU (`{os.getenv('ESP8266_DEVICE_IP', '192.168.0.111')}`)  ",
        f"**Backend Ingestion:** `{os.getenv('BACKEND_URL', 'http://192.168.0.106:8000/api/logs/')}`  ",
        "",
        "---",
        "",
        "## Summary Results Table",
        "",
        "| Scenario | Name | Attacker | Defender | ASR | Detection Rate | Availability | Cost | Hypothesis | Verdict |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for sc_id, data in results.items():
        name = data.get("name", "")
        atk = data.get("attacker", "").upper()
        dfn = data.get("defender", "").upper()
        agg = data.get("aggregate_metrics", {})
        hyp = data.get("hypothesis_result", {})

        asr = f"{agg.get('attack_success_rate', 0.0):.4f}"
        dr = f"{agg.get('detection_rate', 0.0):.4f}"
        avail = f"{agg.get('service_availability', 0.0):.4f}"
        cost = f"{agg.get('intervention_cost', 0.0):.2f}"
        verdict = f"**{hyp.get('verdict', 'N/A')}**"
        hyp_str = f"{hyp.get('metric', '')} {hyp.get('direction', '')} {hyp.get('threshold', '')}"

        lines.append(f"| `{sc_id}` | {name} | {atk} | {dfn} | {asr} | {dr} | {avail} | {cost} | `{hyp_str}` | {verdict} |")

    lines.extend([
        "",
        "---",
        "",
        "## Scenario Key Insights",
        "",
        "1. **SCE-001 (Baseline: Scripted vs Static)**: Establishes the non-adaptive benchmark where fixed rule-based defenders face predictable kill-chains.",
        "2. **SCE-002 (RL Attacker: PPO vs Static)**: Demonstrates the ability of an adaptive PPO policy to discover evasion paths against rigid static rules.",
        "3. **SCE-003 (RL Defender: Scripted vs RL)**: Evaluates whether an autonomous RL agent can suppress attack impact while avoiding unnecessary service downtime.",
        "4. **SCE-004 (Co-Evolution: PPO vs RL)**: Simulates simultaneous two-player Markov game dynamics with both offensive and defensive policies adapting concurrently.",
        "5. **SCE-005 (Generative Duel: LLM vs LLM)**: Explores generative AI adversaries and defenders using Google Gemini 3.5 Flash navigating real physical IoT telemetry and containment.",
        "",
        "---",
        "",
        "All telemetry, episode summaries, and hypothesis validations were recorded with live physical hardware in the loop.",
    ])

    report_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[+] Suite comparison report written to: {output_path}")
    print("\n" + report_content + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Scenarios 1-5 on Physical Hardware")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per scenario (default: 5)")
    parser.add_argument("--wait-pid", type=int, default=None, help="PID of currently running experiment to wait for")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Wait for any running process
    if args.wait_pid:
        wait_for_process(args.wait_pid)

    all_results = {}

    # 2. Run scenarios 1 to 4 sequentially
    for sc_id, config_path in SCENARIOS:
        summary_file = results_dir / f"{sc_id}_summary.json"
        
        # If SCE-001 was already running and finished just now, load its result if recent
        if sc_id == "SCE-001" and args.wait_pid and summary_file.exists():
            mtime = summary_file.stat().st_mtime
            # If updated in last 10 minutes
            if (time.time() - mtime) < 600:
                print(f"[+] Found fresh result for {sc_id} (completed by PID {args.wait_pid}). Loading...")
                with open(summary_file, "r", encoding="utf-8") as f:
                    all_results[sc_id] = json.load(f)
                continue

        print(f"\n=======================================================")
        print(f"  Executing {sc_id} on Physical ESP8266 Hardware")
        print(f"=======================================================")
        try:
            summary = run_experiment(
                config_path=config_path,
                episodes=args.episodes,
                output_dir=args.output_dir,
                hardware=True,
            )
            all_results[sc_id] = summary
        except Exception as e:
            print(f"[!] Error running {sc_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # 3. Generate combined comparison report
    report_file = results_dir / "physical_hardware_suite_report.md"
    generate_comparison_report(all_results, report_file)


if __name__ == "__main__":
    main()
