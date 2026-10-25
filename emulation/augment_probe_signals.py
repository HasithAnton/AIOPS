"""
augment_probe_signals.py
========================
Takes mininet_telemetry_v2.csv (real Mininet data with correct fault labels
and real SNMP counters, but a broken/flat iperf3 probe: latency=0.5, loss=0.0
everywhere) and injects *realistic* synthetic probe values per fault type.

Rationale
---------
The v2 dataset has:
  - Real SNMP counter data  (ifInOctets, ifOutDiscards, etc.)        ← authentic
  - Correct fault labels    (link_failure, congestion, ...)           ← authentic
  - Broken probe values     (latency=0.5, loss=0.0 everywhere)       ← synthetic fix

We replace the two broken probe columns with distributions that match what
a working iperf3 / ICMP probe WOULD have seen given the tc-netem / tc-qdisc
faults that were injected (80 ms delay for congestion, 40 % loss for
packet_loss, link down for link_failure, MTU=500 for mtu_mismatch).

All other columns are left unchanged.
"""

import numpy as np
import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
IN_CSV  = 'mininet_telemetry_v2.csv'
OUT_CSV = 'mininet_telemetry_augmented.csv'

rng = np.random.default_rng(seed=42)

# ── Probe distributions by fault type ─────────────────────────────────────────
#
# Derived from:
#   • tc netem delay 80ms on l1-eth2 during congestion
#   • tc netem loss  40%  on l1-eth2 during packet_loss
#   • ip link set l1-eth2 down during link_failure (probe can't reach h2)
#   • tc qdisc tbf rate 256kbit burst 32kbit during congestion (causes queuing)
#   • ip link set l2 mtu 500 during mtu_mismatch (fragmentation retries)
#   • interface flapping every 3 s → high transient loss during flap windows
#
# Format:  (latency_dist_fn, loss_dist_fn)
# The functions take `n` and return n values clamped to valid ranges.

def lat_normal(mu, sigma, lo=0.1, hi=2000.0):
    def _f(n): return np.clip(rng.normal(mu, sigma, n), lo, hi)
    return _f

def lat_lognormal(mu, sigma, lo=0.1, hi=2000.0):
    def _f(n): return np.clip(rng.lognormal(mu, sigma, n), lo, hi)
    return _f

def loss_zero(n):
    return np.zeros(n)

def loss_uniform(lo, hi):
    def _f(n): return np.clip(rng.uniform(lo, hi, n), 0.0, 1.0)
    return _f

def loss_beta(a, b):
    def _f(n): return np.clip(rng.beta(a, b, n), 0.0, 1.0)
    return _f


# ── Realistic noise helpers ────────────────────────────────────────────────────
def add_normal_spikes(arr, spike_prob=0.02, spike_scale=5.0):
    """Occasionally a normal probe reading spikes (ARP miss, brief congestion)."""
    mask = rng.random(len(arr)) < spike_prob
    arr[mask] *= (1 + rng.exponential(spike_scale, mask.sum()))
    return arr

def add_partial_recovery(arr, recovery_prob=0.15, recovery_scale=0.4):
    """During a fault, some probe readings show partial recovery toward normal."""
    mask = rng.random(len(arr)) < recovery_prob
    arr[mask] *= rng.uniform(recovery_scale, 1.0, mask.sum())
    return arr

def add_measurement_noise(arr, noise_frac=0.08):
    """Add ±8% multiplicative Gaussian noise to simulate measurement jitter."""
    return arr * (1 + rng.normal(0, noise_frac, len(arr)))


FAULT_PROBE = {
    # Normal: sub-ms RTT, zero loss — but with occasional brief spikes (ARP, GC pause)
    'none': (
        lambda n: np.clip(
            add_normal_spikes(
                np.clip(rng.lognormal(-0.7, 0.55, n), 0.1, 10.0),
                spike_prob=0.025, spike_scale=4.0),
            0.1, 50.0),
        lambda n: np.clip(
            rng.exponential(0.003, n),   # mostly 0, rare tiny loss bursts
            0.0, 0.15),
    ),

    # Congestion: tc-netem 80ms delay + tbf queuing causes 60-110ms RTT.
    # BUT: ~15% of probe cycles catch a brief inter-burst gap → looks near-normal.
    'congestion': (
        lambda n: np.clip(
            add_partial_recovery(
                add_measurement_noise(rng.normal(82.0, 14.0, n)),
                recovery_prob=0.15, recovery_scale=0.03),   # ~3% of normal = ~2ms
            0.5, 250.0),
        lambda n: np.clip(rng.uniform(0.0, 0.06, n), 0.0, 0.12),
    ),

    # Link failure: probe can't reach h2 → timeout (999ms) and 100% loss.
    # ~5% of records: link is mid-transition (up→down or down→up) → partial.
    'link_failure': (
        lambda n: np.where(
            rng.random(n) < 0.05,                          # 5% transitional
            rng.uniform(50.0, 500.0, n),
            np.full(n, 999.0)),
        lambda n: np.where(
            rng.random(n) < 0.05,                          # 5% transitional
            rng.uniform(0.3, 0.9, n),
            np.full(n, 1.0)),
    ),

    # MTU mismatch: MTU=500 → large UDP probe packets fragmented/dropped.
    # Latency only slightly elevated (fragmentation overhead), loss is main signal.
    'mtu_mismatch': (
        lambda n: np.clip(
            add_measurement_noise(rng.lognormal(0.9, 0.6, n)),
            0.5, 25.0),
        lambda n: np.clip(
            add_partial_recovery(
                rng.beta(2.5, 7, n),                       # centre ~0.26
                recovery_prob=0.12, recovery_scale=0.1),
            0.0, 0.85),
    ),

    # Packet loss: 40% tc-netem loss on l1-eth2. Latency unchanged (no delay).
    # Probe sees ~35-45% loss (some variance due to UDP burst timing).
    'packet_loss': (
        lambda n: np.clip(
            add_normal_spikes(
                np.clip(rng.lognormal(-0.4, 0.5, n), 0.2, 8.0),
                spike_prob=0.03, spike_scale=2.0),
            0.2, 15.0),
        lambda n: np.clip(
            add_partial_recovery(
                rng.beta(7, 10, n),                        # centre ~0.41
                recovery_prob=0.10, recovery_scale=0.05),
            0.02, 0.90),
    ),

    # Interface flap: every 3s the link goes down/up → bursty loss + high latency
    # when down, near-normal when up. Very high variance per row.
    'interface_flap': (
        lambda n: (
            lambda is_down:
                np.clip(
                    np.where(is_down,
                        np.clip(rng.normal(150.0, 60.0, n), 5.0, 600.0),   # down
                        np.clip(rng.lognormal(-0.5, 0.6, n), 0.2, 15.0)),  # up
                    0.2, 999.0)
        )(rng.random(n) < 0.45),   # 45% of rows are in "link-down" phase
        lambda n: (
            lambda is_down:
                np.clip(
                    np.where(is_down,
                        rng.beta(4, 2, n),          # down: high loss ~0.67
                        rng.exponential(0.01, n)),   # up:   near-zero loss
                    0.0, 1.0)
        )(rng.random(n) < 0.45),
    ),
}

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"[*] Loading {IN_CSV} ...")
df = pd.read_csv(IN_CSV, low_memory=False)

# Drop duplicate header rows (Mininet sometimes writes header mid-file)
df = df[df['is_anomaly'] != 'is_anomaly'].copy()

# Cast numeric columns
for col in ['latency_ms', 'packet_loss_rate', 'is_anomaly',
            'ifInOctets', 'ifOutOctets', 'ifInDiscards', 'ifOutDiscards',
            'ifInErrors', 'cpu_utilization', 'delta_ifInOctets',
            'delta_ifOutOctets', 'delta_ifInDiscards']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"    Total records   : {len(df):,}")
print(f"    Anomaly records : {df['is_anomaly'].sum():,.0f} "
      f"({df['is_anomaly'].mean()*100:.1f}%)")
print(f"    Fault types     : {sorted(df['anomaly_type'].unique().tolist())}")

# ── Inject synthetic probe values ─────────────────────────────────────────────
print("\n[*] Injecting realistic probe signals ...")
new_lat  = np.zeros(len(df))
new_loss = np.zeros(len(df))

for fault, (lat_fn, loss_fn) in FAULT_PROBE.items():
    mask = df['anomaly_type'] == fault
    n = mask.sum()
    if n == 0:
        continue
    new_lat[mask]  = lat_fn(n)
    new_loss[mask] = loss_fn(n)
    print(f"    {fault:<20s}  n={n:>6,}  "
          f"lat=[{new_lat[mask].min():.1f}..{new_lat[mask].max():.1f}] ms  "
          f"loss=[{new_loss[mask].min():.3f}..{new_loss[mask].max():.3f}]")

df['latency_ms']       = new_lat.round(3)
df['packet_loss_rate'] = new_loss.round(4)

# ── Sanity check ───────────────────────────────────────────────────────────────
print("\n[*] Per-fault probe summary:")
print(df.groupby('anomaly_type')[['latency_ms','packet_loss_rate']]
        .agg(['mean','std']).round(3).to_string())

# ── Save ───────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\n[OK] Saved {len(df):,} records to {OUT_CSV}")
print(f"     File size: {os.path.getsize(OUT_CSV)/1e6:.1f} MB")
