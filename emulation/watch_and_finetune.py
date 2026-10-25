#!/usr/bin/env python3
"""
Auto-watcher: monitors for mininet_telemetry.csv completion,
then automatically runs fine-tuning on it.
Run this in a separate terminal while mininet_topo.py is running.
"""
import time, os, sys, subprocess
from pathlib import Path

EMULATION_DIR = Path(__file__).parent
CSV_PATH      = EMULATION_DIR / "mininet_telemetry.csv"
VENV_PYTHON   = EMULATION_DIR.parent / "venv/bin/python"
FINETUNE_SCRIPT = EMULATION_DIR / "finetune_on_mininet.py"

print("👀 Watching for mininet_telemetry.csv...")
print(f"   Path: {CSV_PATH}")
print()

last_size = 0
stable_count = 0
STABLE_NEEDED = 3   # file size unchanged for 3 consecutive checks = done

while True:
    if CSV_PATH.exists():
        size = CSV_PATH.stat().st_size
        rows = 0
        try:
            import pandas as pd
            rows = len(pd.read_csv(CSV_PATH))
        except Exception:
            pass

        if size == last_size and size > 0:
            stable_count += 1
            print(f"\r  📄 {CSV_PATH.name}  |  {size/1e6:.1f} MB  |  ~{rows:,} rows  |  stable {stable_count}/{STABLE_NEEDED}", end="", flush=True)
        else:
            stable_count = 0
            print(f"\r  📄 {CSV_PATH.name}  |  {size/1e6:.1f} MB  |  ~{rows:,} rows  |  writing...", end="", flush=True)

        last_size = size

        if stable_count >= STABLE_NEEDED and size > 1_000_000:  # at least 1MB
            print("\n")
            print("✅ Mininet dataset complete!")
            print(f"   Rows: {rows:,}   Size: {size/1e6:.1f} MB")
            print()
            print("🚀 Starting fine-tuning on Mininet data...")
            print("─" * 50)
            result = subprocess.run(
                [str(VENV_PYTHON), str(FINETUNE_SCRIPT)],
                cwd=str(EMULATION_DIR)
            )
            if result.returncode == 0:
                print()
                print("✅ Fine-tuning complete! Models saved to emulation/mininet_models/")
                print()
                print("🎯 Next step — update live pipeline to use Mininet models:")
                print("   ./start_demo.sh")
            else:
                print("❌ Fine-tuning failed. Check errors above.")
            sys.exit(0)
    else:
        print(f"\r  ⏳ Waiting for Mininet to start writing data...", end="", flush=True)

    time.sleep(10)
