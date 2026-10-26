#!/usr/bin/env python3
"""
Generate Deep-Dive Agent Reasoning Case Studies
==============================================
Runs live autonomous RCA investigations across representative network incident
scenarios using the LangGraph ReAct Agent and Cerebras LLM.
Captures full Thought -> Action -> Observation trajectories and structured RCA reports.
Outputs results to results/agent_case_study_report.md and results/agent_case_studies.json.
"""

import os, sys, json, time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from network.topology import NetworkTopology
from network.scenarios import build_all_scenarios
from agent.tools.diagnostic_tools import ALL_TOOLS, _state, inject_network_state
from agent.react_agent import run_rca_investigation
from agent.prompts import RCA_SYSTEM_PROMPT
from agent.run_agent import get_llm, build_network_state
from benchmark.agent_benchmark import parse_verdict, is_correct

CASE_STUDY_SCENARIOS = [
    {
        "id": "SCEN-01",
        "title": "Case Study 1: Hard Link Failure (Spine-Leaf Core)",
        "fault_type": "link_failure",
        "device": "Spine1",
        "interface": "Ethernet1/3",
        "mse": "1.499763",
        "metrics": {
            "ifOperStatus": 2, "latency_ms": 0.0, "packet_loss_rate": 0.0,
            "ifInOctets": 0, "ifOutOctets": 0, "ifInDiscards": 0, "ifOutDiscards": 0,
            "total_discards": 0, "ifInErrors": 0, "cpu_utilization": 18.5
        }
    },
    {
        "id": "SCEN-03",
        "title": "Case Study 2: Leaf Uplink Congestion & Buffer Overflow",
        "fault_type": "congestion",
        "device": "Leaf1",
        "interface": "Ethernet1/1",
        "mse": "1.404550",
        "metrics": {
            "ifOperStatus": 1, "latency_ms": 80.0, "packet_loss_rate": 0.15,
            "ifInOctets": 75000000000, "ifOutOctets": 75000000000,
            "ifInDiscards": 0, "ifOutDiscards": 12500000, "total_discards": 12500000,
            "ifInErrors": 0, "cpu_utilization": 22.4
        }
    },
    {
        "id": "SCEN-04",
        "title": "Case Study 3: Intermittent Interface Flapping",
        "fault_type": "interface_flap",
        "device": "Leaf2",
        "interface": "Ethernet1/1",
        "mse": "0.870682",
        "metrics": {
            "ifOperStatus": 2, "latency_ms": 0.0, "packet_loss_rate": 0.0,
            "ifInOctets": 0, "ifOutOctets": 0, "ifInDiscards": 0, "ifOutDiscards": 0,
            "total_discards": 0, "ifInErrors": 0, "cpu_utilization": 19.1
        }
    },
    {
        "id": "SCEN-06",
        "title": "Case Study 4: Routing Loop & TTL Exhaustion",
        "fault_type": "routing_loop",
        "device": "Spine1",
        "interface": "Ethernet1/2",
        "mse": "0.547359",
        "metrics": {
            "ifOperStatus": 1, "latency_ms": 120.0, "packet_loss_rate": 0.10,
            "ifInOctets": 74250000000, "ifOutOctets": 74250000000,
            "ifInDiscards": 0, "ifOutDiscards": 0, "total_discards": 0,
            "ifInErrors": 0, "cpu_utilization": 96.8
        }
    }
]

def run_case_studies():
    os.makedirs("results", exist_ok=True)
    report_file = Path("results/agent_case_study_report.md")
    json_file = Path("results/agent_case_studies.json")

    print(f"\n{'═'*65}")
    print("  RUNNING DEEP-DIVE AGENT REASONING CASE STUDIES (4 SCENARIOS)")
    print(f"{'═'*65}\n")

    # Initialize shared network topology and telemetry
    df = pd.read_parquet("data/snmp_telemetry.parquet")
    topo = NetworkTopology()
    topo.build_combined()

    syslogs = []
    syslog_files = sorted(Path("data").glob("syslog*.jsonl")) + sorted(Path("data").glob("syslog*.txt"))
    if syslog_files:
        with open(syslog_files[-1], encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    syslogs.append(obj.get("log", line))
                except Exception:
                    syslogs.append(line)

    build_network_state(df, syslogs, topo, np.zeros(len(df)), df)
    llm = get_llm()

    case_studies = []

    for i, cs in enumerate(CASE_STUDY_SCENARIOS, 1):
        print(f"[{i}/4] Executing {cs['title']} on {cs['device']}/{cs['interface']}...")

        # Reset all links to UP
        all_links = (_state.topology.get("links", []) or []) + (_state.topology.get("edges", []) or [])
        for l in all_links:
            l["status"] = "UP"

        # Inject metrics and scores for this scenario
        _state.metrics.setdefault(cs["device"], {})[cs["interface"]] = cs["metrics"]
        _state.scores.setdefault(cs["device"], {})[cs["interface"]] = float(cs["mse"])

        if cs["metrics"].get("ifOperStatus") == 2:
            for l in all_links:
                if (l.get("source") == cs["device"] and l.get("intf_u") == cs["interface"]) or \
                   (l.get("target") == cs["device"] and l.get("intf_v") == cs["interface"]):
                    l["status"] = "DOWN"

        alert = {
            "device_id": cs["device"],
            "interface_id": cs["interface"],
            "anomaly_type": cs["fault_type"],
            "mse_score": cs["mse"],
            "tier": "1+2",
            "timestamp": "2026-09-20 03:00:00+00:00",
            "ground_truth": cs["fault_type"],
            "metrics": cs["metrics"]
        }

        t0 = time.time()
        result = None
        for attempt in range(4):
            try:
                result = run_rca_investigation(
                    alert=alert,
                    tools=ALL_TOOLS,
                    llm=llm,
                    system_prompt=RCA_SYSTEM_PROMPT,
                    verbose=False,
                )
                break
            except Exception as e:
                print(f"      [Attempt {attempt+1} failed: {e}. Retrying in 5s...]")
                time.sleep(5)

        if result is None:
            print(f"      [!] Failed to investigate {cs['id']} after retries.")
            continue

        elapsed = round(time.time() - t0, 2)
        verdict = parse_verdict(result["final_report"])
        correct = is_correct(verdict, cs["fault_type"])

        print(f"      Verdict: {verdict} | Correct: {correct} | Tools: {len(result['trajectory'])} | {elapsed}s")

        case_studies.append({
            "scenario_id": cs["id"],
            "title": cs["title"],
            "ground_truth": cs["fault_type"],
            "device": cs["device"],
            "interface": cs["interface"],
            "alert": alert,
            "verdict": verdict,
            "is_correct": correct,
            "elapsed_seconds": elapsed,
            "tool_call_count": len(result["trajectory"]),
            "trajectory": result["trajectory"],
            "final_report": result["final_report"],
        })

    # Save JSON case studies
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2)
    print(f"\n[✓] Saved structured case studies to {json_file}")

    # Format Comprehensive Markdown Report
    md_sections = []
    for cs in case_studies:
        traj_md = ""
        for step_idx, step in enumerate(cs["trajectory"], 1):
            args_str = json.dumps(step.get("arguments", {}))
            obs_raw = step.get("observation", "")
            if len(obs_raw) > 280:
                obs_fmt = obs_raw[:280] + " ... [truncated]"
            else:
                obs_fmt = obs_raw

            traj_md += f"""
#### Step {step_idx}: Tool Invocations
* **Action**: `{step.get('tool')}`
* **Arguments**: `{args_str}`
* **Observation**:
```json
{obs_fmt}
```
"""

        status_badge = "✅ CORRECT VERDICT" if cs["is_correct"] else "❌ MISDIAGNOSED"

        md_sections.append(f"""
## {cs['title']}

* **Scenario Identifier**: `{cs['scenario_id']}`
* **Target Entity**: `{cs['device']}` $\\rightarrow$ `{cs['interface']}`
* **Ground Truth Classification**: `{cs['ground_truth']}`
* **Agent Verdict**: `{cs['verdict']}` ({status_badge})
* **Tool Interactions**: {cs['tool_call_count']} calls | **Investigation Time**: {cs['elapsed_seconds']}s

### Diagnostic Investigation Trajectory (ReAct Loop)
{traj_md.strip()}

### Agent Final Incident Report
{cs['final_report'].strip()}

---
""")

    full_md = f"""# Tier 3 Autonomous RCA Agent: Deep-Dive Reasoning Case Studies
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model Provider:** Cerebras API (`qwen-3.8-27b`) / LangGraph ReAct Framework  
**Scope:** Step-by-Step Multi-Tool Diagnostic Trajectories and Autonomous Incident Reports  

---

## Executive Overview

This document records the end-to-end autonomous reasoning trajectory of the Tier 3 LangGraph ReAct agent across four representative network incident scenarios:
1. **Physical Layer**: Complete link failure between Spine and Leaf switches.
2. **Transport/Queue Layer**: Buffer exhaustion and packet drops driven by heavy traffic congestion.
3. **Flapping Interface**: Intermittent physical/optics flapping causing routing oscillations.
4. **Routing Control Plane**: Circular OSPF loop producing severe CPU spikes and TTL exhaustion.

In each incident, the agent autonomously formulates diagnostic hypotheses, invokes specialized network inspection tools (`get_metrics`, `inspect_topology`, `query_syslogs`, `trace_path`, `check_routes`), corroborates findings across independent signals, and generates a structured Root Cause Analysis report with actionable CLI remediations.

---

{''.join(md_sections)}
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"[✓] Saved Case Study Markdown Report to {report_file}")

if __name__ == "__main__":
    run_case_studies()
