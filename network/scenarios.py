"""
Network Failure Scenarios
=========================
Defines the FailureScenario dataclass and the scenario bank used by the
benchmark runner and the triage pipeline.

build_all_scenarios() accepts optional topology/time arguments so it can be
called with or without pre-built topologies (both call sites work):
  - build_all_scenarios()                          — creates default topologies
  - build_all_scenarios(sl_topo, wan_topo, t0)     — uses supplied topologies
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd

from network.anomaly_injector import AnomalyInjector
from network.topology import NetworkTopology


class FailureScenario:
    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str,
        topology_type: str,
        anomaly_configs: List[Dict[str, Any]],
        ground_truth_fault_class: str,
        ground_truth_entity: str,
        ground_truth_interface: str,
        ground_truth_peer: str,
        ground_truth_remediation_keywords: List[str],
        optimal_tool_sequence: List[str],
        difficulty: str,
        alert_message: str,
    ):
        self.id                  = scenario_id
        self.name                = name
        self.description         = description
        self.topology_type       = topology_type
        self.anomaly_configs     = anomaly_configs
        self.ground_truth = {
            "fault_class": ground_truth_fault_class,
            "entity":      ground_truth_entity,
            "interface":   ground_truth_interface,
            "peer":        ground_truth_peer,
            "remediation_keywords": ground_truth_remediation_keywords,
        }
        self.optimal_tool_sequence = optimal_tool_sequence
        self.difficulty            = difficulty
        self.alert_message         = alert_message


def build_all_scenarios(
    spine_leaf_topo: Optional[NetworkTopology] = None,
    wan_topo:        Optional[NetworkTopology] = None,
    base_time:       Optional[datetime]        = None,
) -> List[FailureScenario]:
    """
    Build the full scenario bank.

    All three arguments are optional — if not supplied, fresh default
    topologies are created internally so callers that omit them
    (e.g. cmd_demo in main.py) still work correctly.
    """
    # ── Defaults ───────────────────────────────────────────────────────────────
    if spine_leaf_topo is None:
        spine_leaf_topo = NetworkTopology()
        spine_leaf_topo.build_spine_leaf()

    if wan_topo is None:
        wan_topo = NetworkTopology()
        wan_topo.build_hierarchical_wan()

    if base_time is None:
        base_time = datetime(2026, 9, 18, 0, 0, 0, tzinfo=timezone.utc)

    # Convert to pandas Timestamp once so timedelta arithmetic is consistent
    if not isinstance(base_time, pd.Timestamp):
        base_time = pd.Timestamp(base_time)

    sl  = AnomalyInjector(spine_leaf_topo)
    wan = AnomalyInjector(wan_topo)

    scenarios: List[FailureScenario] = []

    # ── SCEN-01  Spine-Leaf Link Failure (easy) ────────────────────────────────
    cfg_01 = sl.create_link_failure(("Spine1", "Leaf1"), base_time, 30.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-01",
        name="Spine-Leaf Link Failure",
        description="Hard link failure between Spine1 and Leaf1.",
        topology_type="spine_leaf",
        anomaly_configs=[cfg_01],
        ground_truth_fault_class="LINK_FAILURE",
        ground_truth_entity="Spine1",
        ground_truth_interface="HundredGigE1/0/1",
        ground_truth_peer="Leaf1",
        ground_truth_remediation_keywords=["re-enable", "standby", "redundant"],
        optimal_tool_sequence=["inspect_topology", "get_metrics", "trace_path"],
        difficulty="easy",
        alert_message="Alert: ifOperStatus DOWN on Spine1 HundredGigE1/0/1. Investigate.",
    ))

    # ── SCEN-02  WAN Core Link Failure (easy) ─────────────────────────────────
    cfg_02 = wan.create_link_failure(("R1", "R2"), base_time, 30.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-02",
        name="WAN Core Link Failure",
        description="Hard link failure between WAN core routers R1 and R2.",
        topology_type="hierarchical_wan",
        anomaly_configs=[cfg_02],
        ground_truth_fault_class="LINK_FAILURE",
        ground_truth_entity="R1",
        ground_truth_interface="TenGigE0/0/0",
        ground_truth_peer="R2",
        ground_truth_remediation_keywords=["re-enable", "backup path", "BGP"],
        optimal_tool_sequence=["inspect_topology", "trace_path", "query_syslogs"],
        difficulty="easy",
        alert_message="Alert: Core link R1-R2 DOWN. Traffic blackhole suspected.",
    ))

    # ── SCEN-03  Leaf Uplink Congestion (medium) ───────────────────────────────
    cfg_03 = sl.create_congestion(("Spine2", "Leaf2"), base_time, 45.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-03",
        name="Leaf Uplink Congestion",
        description="Sustained high utilisation on Leaf2 uplink to Spine2.",
        topology_type="spine_leaf",
        anomaly_configs=[cfg_03],
        ground_truth_fault_class="CONGESTION",
        ground_truth_entity="Leaf2",
        ground_truth_interface="HundredGigE1/0/2",
        ground_truth_peer="Spine2",
        ground_truth_remediation_keywords=["QoS", "ECMP", "bandwidth"],
        optimal_tool_sequence=["get_metrics", "inspect_topology", "check_routes"],
        difficulty="medium",
        alert_message="Alert: High output drops on Leaf2 HundredGigE1/0/2.",
    ))

    # ── SCEN-04  Interface Flap (medium) ──────────────────────────────────────
    cfg_04 = sl.create_interface_flap(("Spine1", "Leaf3"), base_time, 20.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-04",
        name="Spine-Leaf Interface Flap",
        description="Repeated UP/DOWN cycling on Spine1 → Leaf3 link.",
        topology_type="spine_leaf",
        anomaly_configs=[cfg_04],
        ground_truth_fault_class="INTERFACE_FLAP",
        ground_truth_entity="Spine1",
        ground_truth_interface="HundredGigE1/0/3",
        ground_truth_peer="Leaf3",
        ground_truth_remediation_keywords=["dampening", "SFP", "cable"],
        optimal_tool_sequence=["query_syslogs", "get_metrics", "inspect_topology"],
        difficulty="medium",
        alert_message="Alert: Repeated LINEPROTO flaps on Spine1 HundredGigE1/0/3.",
    ))

    # ── SCEN-05  MTU Mismatch (medium) ────────────────────────────────────────
    cfg_05 = sl.create_mtu_mismatch(("Spine2", "Leaf4"), base_time, 25.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-05",
        name="MTU Mismatch on Leaf4 Uplink",
        description="Fragmentation errors due to MTU mismatch between Spine2 and Leaf4.",
        topology_type="spine_leaf",
        anomaly_configs=[cfg_05],
        ground_truth_fault_class="MTU_MISMATCH",
        ground_truth_entity="Leaf4",
        ground_truth_interface="HundredGigE1/0/4",
        ground_truth_peer="Spine2",
        ground_truth_remediation_keywords=["MTU", "1500", "jumbo"],
        optimal_tool_sequence=["get_metrics", "query_syslogs", "check_routes"],
        difficulty="medium",
        alert_message="Alert: Elevated ifInErrors and fragmentation on Leaf4.",
    ))

    # ── SCEN-06  Routing Loop (hard) ──────────────────────────────────────────
    cfg_06 = sl.create_routing_loop(("Spine1", "Leaf2"), base_time, 30.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-06",
        name="Routing Loop — Spine-Leaf",
        description="Circular OSPF route causing TTL exhaustion on Spine1-Leaf2 segment.",
        topology_type="spine_leaf",
        anomaly_configs=[cfg_06],
        ground_truth_fault_class="ROUTING_LOOP",
        ground_truth_entity="Spine1",
        ground_truth_interface="HundredGigE1/0/2",
        ground_truth_peer="Leaf2",
        ground_truth_remediation_keywords=["route filter", "OSPF", "TTL"],
        optimal_tool_sequence=["check_routes", "query_syslogs", "trace_path"],
        difficulty="hard",
        alert_message="Alert: TTL_EXPIRED messages and rising CPU on Spine1.",
    ))

    # ── SCEN-07  WAN Congestion (medium) ──────────────────────────────────────
    cfg_07 = wan.create_congestion(("R1", "D1"), base_time, 40.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-07",
        name="WAN Aggregation Congestion",
        description="High utilisation on R1 → D1 distribution link.",
        topology_type="hierarchical_wan",
        anomaly_configs=[cfg_07],
        ground_truth_fault_class="CONGESTION",
        ground_truth_entity="R1",
        ground_truth_interface="TenGigE0/0/1",
        ground_truth_peer="D1",
        ground_truth_remediation_keywords=["QoS", "policy", "redistribute"],
        optimal_tool_sequence=["get_metrics", "check_routes", "inspect_topology"],
        difficulty="medium",
        alert_message="Alert: Output queue drops on R1 TenGigE0/0/1 to D1.",
    ))

    # ── SCEN-08  WAN MTU Mismatch (hard) ──────────────────────────────────────
    cfg_08 = wan.create_mtu_mismatch(("D3", "A4"), base_time, 25.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-08",
        name="WAN MTU Mismatch",
        description="MTU mismatch between distribution D3 and access A4.",
        topology_type="hierarchical_wan",
        anomaly_configs=[cfg_08],
        ground_truth_fault_class="MTU_MISMATCH",
        ground_truth_entity="D3",
        ground_truth_interface="GigabitEthernet0/0",
        ground_truth_peer="A4",
        ground_truth_remediation_keywords=["MTU", "1500", "ICMP unreachable"],
        optimal_tool_sequence=["get_metrics", "query_syslogs", "inspect_topology"],
        difficulty="hard",
        alert_message="Alert: Fragmentation errors between D3 and A4.",
    ))

    # ── SCEN-09  WAN Interface Flap (medium) ──────────────────────────────────
    cfg_09 = wan.create_interface_flap(("R2", "D2"), base_time, 20.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-09",
        name="WAN Interface Flap",
        description="Repeated interface flapping on R2 → D2.",
        topology_type="hierarchical_wan",
        anomaly_configs=[cfg_09],
        ground_truth_fault_class="INTERFACE_FLAP",
        ground_truth_entity="R2",
        ground_truth_interface="TenGigE0/0/2",
        ground_truth_peer="D2",
        ground_truth_remediation_keywords=["dampening", "SFP", "NIC driver"],
        optimal_tool_sequence=["query_syslogs", "get_metrics", "trace_path"],
        difficulty="medium",
        alert_message="Alert: Repeated LINEPROTO transitions on R2 TenGigE0/0/2.",
    ))

    # ── SCEN-10  WAN Routing Loop (hard) ──────────────────────────────────────
    cfg_10 = wan.create_routing_loop(("R1", "R2"), base_time, 30.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-10",
        name="WAN Core Routing Loop",
        description="BGP route loop between core WAN routers R1 and R2.",
        topology_type="hierarchical_wan",
        anomaly_configs=[cfg_10],
        ground_truth_fault_class="ROUTING_LOOP",
        ground_truth_entity="R1",
        ground_truth_interface="TenGigE0/0/0",
        ground_truth_peer="R2",
        ground_truth_remediation_keywords=["route filter", "BGP", "TTL", "OSPF"],
        optimal_tool_sequence=["check_routes", "trace_path", "query_syslogs"],
        difficulty="hard",
        alert_message="Alert: TTL_EXPIRED flood and CPU spike on R1. Possible routing loop.",
    ))

    return scenarios
