# Agent Benchmark Report
Generated: 2026-10-26 12:40

## Overall Performance

| Metric | Value |
|:---|:---:|
| Total Alerts Tested | 25 |
| Correct Verdicts | 22 / 25 |
| **Overall Accuracy** | **88.0%** |
| False Positive Elimination | 2 / 5 = 40.0% |
| Avg Tool Calls (correct) | 12.0 |

## Per-Fault-Type Accuracy

| Fault Type | Correct | Total | Accuracy |
|:---|:---:|:---:|:---:|
| `link_failure` | 4 | 4 | 100% |
| `congestion` | 4 | 4 | 100% |
| `mtu_mismatch` | 4 | 4 | 100% |
| `interface_flap` | 4 | 4 | 100% |
| `routing_loop` | 4 | 4 | 100% |
| **False Positive (normal)** | 2 | 5 | 40% |

## Per-Alert Results

| # | Device | Interface | Ground Truth | Verdict | ✓/✗ | Tools | Time |
|:--|:--|:--|:--|:--|:--:|:--:|:--:|
| 1 | R1 | HundredGigE0/1 | `link_failure` | `LINK_FAILURE` | ✅ | 10 | 31.2s |
| 2 | R2 | HundredGigE0/1 | `link_failure` | `LINK_FAILURE` | ✅ | 8 | 7.4s |
| 3 | Leaf1 | Ethernet1/1 | `link_failure` | `LINK_FAILURE` | ✅ | 10 | 16.3s |
| 4 | Spine1 | Ethernet1/3 | `link_failure` | `LINK_FAILURE` | ✅ | 9 | 8.8s |
| 5 | Leaf1 | Ethernet1/1 | `congestion` | `CONGESTION` | ✅ | 17 | 12.2s |
| 6 | Spine1 | Ethernet1/1 | `congestion` | `CONGESTION` | ✅ | 23 | 23.1s |
| 7 | Spine2 | Ethernet1/4 | `congestion` | `CONGESTION` | ✅ | 11 | 12.3s |
| 8 | Leaf2 | Ethernet1/2 | `congestion` | `CONGESTION` | ✅ | 21 | 18.0s |
| 9 | A4 | GigE0/1 | `mtu_mismatch` | `MTU_MISMATCH` | ✅ | 8 | 6.5s |
| 10 | D3 | GigE0/1 | `mtu_mismatch` | `MTU_MISMATCH` | ✅ | 5 | 5.2s |
| 11 | A1 | GigE0/1 | `mtu_mismatch` | `MTU_MISMATCH` | ✅ | 8 | 5.4s |
| 12 | D1 | GigE0/1 | `mtu_mismatch` | `MTU_MISMATCH` | ✅ | 9 | 5.7s |
| 13 | R2 | HundredGigE0/1 | `interface_flap` | `INTERFACE_FLAP` | ✅ | 7 | 6.5s |
| 14 | R1 | HundredGigE0/1 | `interface_flap` | `INTERFACE_FLAP` | ✅ | 7 | 6.6s |
| 15 | Leaf2 | Ethernet1/1 | `interface_flap` | `INTERFACE_FLAP` | ✅ | 9 | 8.2s |
| 16 | Leaf2 | Ethernet1/2 | `interface_flap` | `INTERFACE_FLAP` | ✅ | 11 | 12.9s |
| 17 | R1 | HundredGigE0/1 | `routing_loop` | `ROUTING_LOOP` | ✅ | 12 | 10.4s |
| 18 | R2 | HundredGigE0/1 | `routing_loop` | `ROUTING_LOOP` | ✅ | 16 | 8.8s |
| 19 | Spine2 | Ethernet1/2 | `routing_loop` | `ROUTING_LOOP` | ✅ | 26 | 18.2s |
| 20 | Spine1 | Ethernet1/2 | `routing_loop` | `ROUTING_LOOP` | ✅ | 20 | 24.3s |
| 21 | A1 | GigE0/1 | `none` | `INTERFACE_FLAP` | ❌ | 8 | 14.6s |
| 22 | A2 | GigE0/2 | `none` | `FALSE_POSITIVE` | ✅ | 8 | 7.3s |
| 23 | A4 | GigE0/1 | `none` | `MTU_MISMATCH` | ❌ | 11 | 10.4s |
| 24 | D1 | GigE0/1 | `none` | `ROUTING_LOOP` | ❌ | 22 | 14.0s |
| 25 | D1 | TenGigE0/2 | `none` | `FALSE_POSITIVE` | ✅ | 10 | 9.9s |
