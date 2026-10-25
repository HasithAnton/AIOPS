#!/usr/bin/env python3
"""
Corrected Synthetic Mininet Dataset Generator
=============================================
Generates realistic kernel-level network telemetry based on a physically
accurate model of the Spine-Leaf topology used in mininet_topo.py.

Fixes three root bugs in the original mininet_topo.py data:

  BUG 1 — Probe targets h4 (no iperf server) → pings always timeout →
           packet_loss_rate=1.0 for both normal AND anomaly rows.
  FIX 1 — Probe targets h2 (iperf server running) → pings succeed
           normally, measure real loss/latency during faults.

  BUG 2 — Congestion on s2-eth2 is NOT on the h1→h4 probe path →
           latency_ms never changes during congestion fault.
  FIX 2 — Congestion on l1-eth2 (h1's uplink to s1) which IS on the
           h1→h2 probe path → latency spikes correctly.

  BUG 3 — All 20 interfaces labeled with same fault_type even though
           only 1-2 interfaces are actually affected. link_failure shows
           ifOperStatus=2 only 9.9% of the time because 18/20 interfaces
           are unaffected.
  FIX 3 — Label at INTERFACE granularity: only the affected interface(s)
           are labeled as anomaly. Unaffected interfaces in the same
           timestep remain 'none'. This gives clean, accurate labels.

Topology:
    2 spines (s1, s2) × 4 leaf switches (l1-l4) × 4 hosts (h1-h4)
    Background traffic: h1→h2 (10Mbps), h1→h3 (10Mbps)
    Probe: h1 pings h2 every 3 seconds (FIXED from h4)

Output:
    emulation/mininet_telemetry.csv  (overwrites broken version)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

rng = np.random.default_rng(42)

# ─── Topology ─────────────────────────────────────────────────────────────────

SWITCHES = {
    's1': ['s1-eth1', 's1-eth2', 's1-eth3', 's1-eth4'],
    's2': ['s2-eth1', 's2-eth2', 's2-eth3', 's2-eth4'],
    'l1': ['l1-eth1', 'l1-eth2', 'l1-eth3'],
    'l2': ['l2-eth1', 'l2-eth2', 'l2-eth3'],
    'l3': ['l3-eth1', 'l3-eth2', 'l3-eth3'],
    'l4': ['l4-eth1', 'l4-eth2', 'l4-eth3'],
}
ALL_INTFS = [(sw, intf) for sw, intfs in SWITCHES.items() for intf in intfs]
N_INTF = len(ALL_INTFS)  # 20 interfaces

MBPS = 1_000_000  # bytes per second per Mbps

# ─── Base traffic model (bytes/second on each interface, normal state) ─────────
# Background: h1→h2 (10Mbps), h1→h3 (10Mbps). ECMP splits 50/50 via s1/s2.
# Only rx_bytes (ifInOctets) modelled; tx_bytes roughly symmetric.
BASE = {
    # Leaf-to-host interfaces
    'l1-eth1': 20 * MBPS,   # h1 sends 20Mbps total (to h2 + h3)
    'l2-eth1': 10 * MBPS,   # h2 receives 10Mbps
    'l3-eth1': 10 * MBPS,   # h3 receives 10Mbps
    'l4-eth1':  0.01 * MBPS, # h4 only probe traffic
    # Leaf uplinks to s1 (half traffic via s1)
    'l1-eth2': 10 * MBPS,
    'l2-eth2':  5 * MBPS,
    'l3-eth2':  5 * MBPS,
    'l4-eth2':  0.01 * MBPS,
    # Leaf uplinks to s2 (other half via s2)
    'l1-eth3': 10 * MBPS,
    'l2-eth3':  5 * MBPS,
    'l3-eth3':  5 * MBPS,
    'l4-eth3':  0.01 * MBPS,
    # Spine s1 ports (traffic distribution)
    's1-eth1': 10 * MBPS,   # s1 → l1
    's1-eth2':  5 * MBPS,   # s1 → l2 (h1→h2 via s1)
    's1-eth3':  5 * MBPS,   # s1 → l3 (h1→h3 via s1)
    's1-eth4':  0.01 * MBPS, # s1 → l4 (probe only)
    # Spine s2 ports
    's2-eth1': 10 * MBPS,
    's2-eth2':  5 * MBPS,
    's2-eth3':  5 * MBPS,
    's2-eth4':  0.01 * MBPS,
}
BASE_OUT = {k: v * 0.9 for k, v in BASE.items()}  # tx is slightly less than rx

NORMAL_LATENCY = 0.8    # ms
NORMAL_LOSS    = 0.0    # fraction


def gamma_sample(mean, cv=0.25, size=1):
    """Sample from Gamma with given mean and coefficient of variation."""
    shape = 1.0 / (cv ** 2)
    scale = mean / shape
    return rng.gamma(shape, scale, size)


def poisson_discard(delta, discard_rate=0.001):
    """Small random discards proportional to traffic volume."""
    return rng.poisson(max(1, delta * discard_rate))


# ─── Per-interface state ───────────────────────────────────────────────────────

class InterfaceState:
    """Tracks cumulative counters for one (device, interface) pair."""
    def __init__(self, device, intf):
        self.device = device
        self.intf   = intf
        # Cumulative counters (start at plausible baseline)
        mean_in  = BASE.get(intf, 0.1 * MBPS)
        mean_out = BASE_OUT.get(intf, 0.09 * MBPS)
        self.rx_bytes    = int(rng.uniform(1e9, 5e9))
        self.tx_bytes    = int(rng.uniform(1e9, 5e9))
        self.rx_dropped  = int(rng.uniform(1e6, 5e7))
        self.tx_dropped  = int(rng.uniform(1e6, 5e7))
        self.rx_errors   = 0
        self.oper_status = 1  # 1=UP, 2=DOWN

    def step(self, fault_type='none', fault_intf=None, t_in_fault=0, flap_t=0):
        """Advance one second. Returns a telemetry dict."""
        intf     = self.intf
        is_fault = (fault_intf is not None) and (intf in fault_intf)

        mean_in  = BASE.get(intf, 0.1 * MBPS)
        mean_out = BASE_OUT.get(intf, 0.09 * MBPS)
        latency  = NORMAL_LATENCY
        loss     = NORMAL_LOSS
        errors   = 0

        # ── Apply fault effects ─────────────────────────────────────────────
        if is_fault:
            if fault_type == 'link_failure':
                # Affected interface goes DOWN completely
                self.oper_status = 2
                mean_in  = 0
                mean_out = 0

            elif fault_type == 'congestion':
                # tc netem: 80ms delay + 5% loss on l1-eth2 (probe path h1→h2)
                # Outbound traffic still flows but with delay and some drops
                tx_drop_rate = 0.05       # 5% dropped by netem
                mean_out_reduced = mean_out * 0.95
                delta_out = int(gamma_sample(mean_out_reduced)[0]) if mean_out > 0 else 0
                delta_in  = int(gamma_sample(mean_in)[0])          if mean_in  > 0 else 0
                self.rx_bytes   += delta_in
                self.tx_bytes   += delta_out
                tx_drop = int(delta_out * tx_drop_rate)
                self.tx_dropped += tx_drop
                self.rx_dropped += poisson_discard(delta_in, 0.001)
                latency = rng.normal(85, 10)    # ~85ms (netem 80ms + switching)
                loss    = rng.beta(5, 95)       # ~5% loss
                return self._snap(fault_type, latency, loss)

            elif fault_type == 'interface_flap':
                # Link alternates UP/DOWN every 3 seconds
                phase = int(flap_t) % 6   # 3s up, 3s down
                if phase >= 3:
                    self.oper_status = 2
                    mean_in  = 0
                    mean_out = 0
                else:
                    self.oper_status = 1
                    # Brief traffic surge after link comes back up
                    mean_in  = mean_in  * rng.uniform(0.8, 1.2)
                    mean_out = mean_out * rng.uniform(0.8, 1.2)

            elif fault_type == 'mtu_mismatch':
                # MTU=500 on this interface: large frames fragmented/dropped
                # In practice: ~60% of packets are >500 bytes → dropped
                # ifInDiscards (rx_dropped) spikes significantly
                drop_frac = rng.beta(60, 40)    # ~60% drop rate
                delta_in  = int(gamma_sample(mean_in * (1 - drop_frac))[0]) if mean_in > 0 else 0
                delta_out = int(gamma_sample(mean_out)[0]) if mean_out > 0 else 0
                self.rx_bytes   += delta_in
                self.tx_bytes   += delta_out
                # The dropped frames are recorded as rx_dropped
                dropped_bytes = int(gamma_sample(mean_in * drop_frac)[0]) if mean_in > 0 else 0
                self.rx_dropped += dropped_bytes
                self.rx_dropped += poisson_discard(delta_out, 0.001)
                latency = rng.normal(2.5, 0.5)  # slightly higher (PMTU probes)
                loss    = rng.beta(10, 90)       # ~10% probe loss (fragmented probes dropped)
                return self._snap(fault_type, latency, loss)

            elif fault_type == 'packet_loss':
                # tc netem: 40% random loss on l1-eth3 (s2 path, probe goes via s2)
                # Effective throughput reduced by 40%
                loss_frac = rng.beta(40, 60)    # ~40% loss
                mean_in_reduced  = mean_in  * (1 - loss_frac)
                mean_out_reduced = mean_out * (1 - loss_frac)
                delta_in  = int(gamma_sample(mean_in_reduced)[0])  if mean_in_reduced  > 0 else 0
                delta_out = int(gamma_sample(mean_out_reduced)[0]) if mean_out_reduced > 0 else 0
                self.rx_bytes   += delta_in
                self.tx_bytes   += delta_out
                drop_in  = int(gamma_sample(mean_in  * loss_frac)[0]) if mean_in > 0 else 0
                drop_out = int(gamma_sample(mean_out * loss_frac)[0]) if mean_out > 0 else 0
                self.rx_dropped += drop_in
                self.tx_dropped += drop_out
                latency = rng.normal(3.5, 1.0)  # TCP retransmits inflate RTT slightly
                loss    = rng.beta(40, 60)       # ~40% probe loss
                return self._snap(fault_type, latency, loss)

        else:
            # Normal operation or unaffected interface during fault
            self.oper_status = 1

        # ── Normal traffic ──────────────────────────────────────────────────
        delta_in  = int(gamma_sample(mean_in)[0])  if mean_in  > 0 else 0
        delta_out = int(gamma_sample(mean_out)[0]) if mean_out > 0 else 0
        self.rx_bytes   += delta_in
        self.tx_bytes   += delta_out
        self.rx_dropped += poisson_discard(delta_in,  0.001)
        self.tx_dropped += poisson_discard(delta_out, 0.001)

        return self._snap(fault_type if is_fault else 'none', latency, loss)

    def _snap(self, fault_type, latency, loss):
        return {
            'device_id'        : self.device,
            'interface_id'     : self.intf,
            'ifOperStatus'     : self.oper_status,
            'ifInOctets'       : self.rx_bytes,
            'ifOutOctets'      : self.tx_bytes,
            'ifInDiscards'     : self.rx_dropped,
            'ifOutDiscards'    : self.tx_dropped,
            'ifInErrors'       : self.rx_errors,
            'latency_ms'       : max(0.0, latency),
            'packet_loss_rate' : max(0.0, min(1.0, loss)),
            'cpu_utilization'  : float(rng.beta(2, 50)),  # ~4% CPU, varies
            'is_anomaly'       : 0 if fault_type == 'none' else 1,
            'anomaly_type'     : fault_type,
        }


# ─── Fault definitions ────────────────────────────────────────────────────────
# INTERFACE-LEVEL labeling: only the physically affected interface(s) are labeled.
# This is the key fix — not all 20 interfaces are labeled with the fault.

FAULTS = [
    # (fault_type, affected_interfaces, duration_s, description)
    ('link_failure',   {'l1-eth2', 's1-eth1'},               300, 'l1→s1 link down'),
    ('congestion',     {'l1-eth2'},                           300, 'l1→s1 80ms delay+5% loss (probe path)'),
    ('interface_flap', {'l1-eth2', 's1-eth1'},               300, 'l1↔s1 flapping every 3s'),
    ('mtu_mismatch',   {'l1-eth2'},                           300, 'l1→s1 MTU=500'),
    ('packet_loss',    {'l1-eth3', 's2-eth1'},                300, 's2 path 40% loss'),
]

WARMUP_S     = 600   # 10 min normal baseline
RECOVERY_S   = 300   # 5 min recovery between faults


def generate():
    print("[*] Generating corrected synthetic Mininet telemetry...")
    print(f"    {N_INTF} interfaces × topology: {list(SWITCHES.keys())}")

    # Initialize interface states
    states = {(sw, intf): InterfaceState(sw, intf)
              for sw, intfs in SWITCHES.items() for intf in intfs}

    t = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt = timedelta(seconds=1)
    rows = []

    def record_phase(duration, fault_type='none', fault_intf=None, flap_offset=0):
        nonlocal t
        for i in range(duration):
            for (sw, intf), state in states.items():
                row = state.step(
                    fault_type=fault_type,
                    fault_intf=fault_intf,
                    t_in_fault=i,
                    flap_t=i + flap_offset,
                )
                row['timestamp'] = t
                rows.append(row)
            t += dt

    # Phase 0: warmup
    print(f"    Phase 0: warmup  ({WARMUP_S}s normal)")
    record_phase(WARMUP_S)

    # Fault phases
    for fault_type, fault_intf, duration, desc in FAULTS:
        print(f"    Phase: {fault_type:20s} ({duration}s)  [{desc}]")
        record_phase(duration, fault_type=fault_type, fault_intf=fault_intf)
        print(f"    Recovery ({RECOVERY_S}s normal)")
        record_phase(RECOVERY_S)

    df = pd.DataFrame(rows)

    # Compute delta columns (per device+interface)
    df = df.sort_values(['device_id','interface_id','timestamp']).reset_index(drop=True)
    for col in ['ifInOctets','ifOutOctets','ifInDiscards','ifOutDiscards']:
        df[f'delta_{col}'] = (
            df.groupby(['device_id','interface_id'])[col]
              .diff()
              .fillna(0)
              .clip(lower=0)
        )

    total = len(df)
    anoms = df['is_anomaly'].sum()
    print(f"\n    Total records   : {total:,}")
    print(f"    Anomaly records : {anoms:,} ({anoms/total*100:.1f}%)")
    print("\n    Records per fault type:")
    for ft, cnt in df.groupby('anomaly_type').size().items():
        print(f"      {ft:20s}: {cnt:,}")

    # Sanity check — do key signals exist?
    print("\n    === Signal Sanity Check ===")
    print("    latency_ms unique per fault:")
    for ft in df['anomaly_type'].unique():
        sub = df[df['anomaly_type']==ft]['latency_ms']
        print(f"      {ft:20s}: mean={sub.mean():.2f}  std={sub.std():.3f}  unique={sub.nunique()}")

    print("    packet_loss_rate unique per fault:")
    for ft in df['anomaly_type'].unique():
        sub = df[df['anomaly_type']==ft]['packet_loss_rate']
        print(f"      {ft:20s}: mean={sub.mean():.3f}  unique={sub.nunique()}")

    print("    ifOperStatus=2 coverage:")
    for ft in df['anomaly_type'].unique():
        sub = df[df['anomaly_type']==ft]
        pct = (sub['ifOperStatus']==2).mean()*100
        print(f"      {ft:20s}: {pct:.1f}%")

    return df


if __name__ == '__main__':
    df = generate()
    out = 'mininet_telemetry.csv'
    df.to_csv(out, index=False)
    print(f"\n[OK] Saved {len(df):,} rows to {out}")
