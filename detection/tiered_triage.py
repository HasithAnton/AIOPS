import pandas as pd
import joblib
import os
import logging
from typing import Dict, Any

from agent.prompts import RCA_SYSTEM_PROMPT
from detection.feature_engineering import NetworkFeaturePipeline

logger = logging.getLogger(__name__)


class TieredTriagePipeline:
    def __init__(self, llm_provider: str = "gemini"):
        self.rf_model_path   = "data/models/random_forest.pkl"
        self.lstm_model_path = "data/models/lstm_autoencoder.pth"
        self.llm_provider    = llm_provider
        self.feature_pipeline = NetworkFeaturePipeline()

    def process_telemetry_window(self, df: pd.DataFrame, network_state: Dict) -> Dict[str, Any]:
        """
        Process a window of telemetry data through the 3-Tier Pipeline:
          Tier 1: Supervised Random Forest  (Known Faults)
          Tier 2: Deep Learning LSTM-AE     (Zero-Day Faults)
          Tier 3: Agentic AI / LangGraph    (Root Cause Investigation)
        """
        result = {
            "tier_1_status": "Passed",
            "tier_2_status": "Passed",
            "tier_3_status": "Skipped",
            "final_rca":     None,
            "anomaly_detected": False,
        }

        if not os.path.exists(self.rf_model_path) or not os.path.exists(self.lstm_model_path):
            raise FileNotFoundError(
                "ML models not found. Run 'python3 main.py detect' first."
            )

        # ── Load models ────────────────────────────────────────────────────────
        import torch
        rf_model   = joblib.load(self.rf_model_path)
        lstm_model = torch.load(
            self.lstm_model_path, map_location="cpu", weights_only=False
        )
        lstm_model.device = torch.device("cpu")
        if hasattr(lstm_model, "model"):
            lstm_model.model.to("cpu")

        # ── Tier 1: Supervised RF ──────────────────────────────────────────────
        rf_preds, rf_scores = rf_model.predict(df)

        # High-confidence: RF flags anomaly with >85% class probability
        high_conf_indices = [
            i for i, (p, s) in enumerate(zip(rf_preds, rf_scores))
            if p == 1 and s > 0.85
        ]

        device, interface, mse_score, tier_num = "Unknown", "Unknown", 0.0, 2
        idx = -1  # will be set in whichever tier fires

        if high_conf_indices:
            result["anomaly_detected"] = True
            result["tier_1_status"]    = "Known Fault Detected (Passed to Tier 3 for RCA)"
            result["tier_2_status"]    = "Skipped"

            idx       = high_conf_indices[0]
            device    = df.iloc[idx]["device_id"]
            interface = df.iloc[idx]["interface_id"]
            mse_score = float(rf_scores[idx])
            tier_num  = 1

        else:
            # ── Tier 2: Unsupervised LSTM-AE ──────────────────────────────────
            lstm_preds, lstm_scores = lstm_model.predict(df)
            anomaly_indices = [i for i, p in enumerate(lstm_preds) if p == 1]

            if not anomaly_indices:
                result["tier_2_status"] = "No Anomalies Detected"
                result["final_rca"]     = "System Normal."
                return result

            result["anomaly_detected"] = True
            result["tier_2_status"]    = (
                "Zero-Day Anomaly Detected (Passed to Tier 3 for RCA)"
            )

            idx       = anomaly_indices[0]
            device    = df.iloc[idx]["device_id"]
            interface = df.iloc[idx]["interface_id"]
            mse_score = float(lstm_scores[idx])
            tier_num  = 2

        # ── Tier 3: LangGraph ReAct Agent ─────────────────────────────────────
        result["tier_3_status"] = "Agent Deployed"

        # Inject live topology/metrics/syslogs into tool state
        from agent.tools.diagnostic_tools import inject_network_state, ALL_TOOLS
        inject_network_state(
            topology = network_state.get("topology", {}),
            metrics  = network_state.get("metrics",  {}),
            syslogs  = network_state.get("syslogs",  []),
            scores   = network_state.get("scores",   {}),
        )

        # Build structured alert dict
        alert = {
            "device_id":    device,
            "interface_id": interface,
            "anomaly_type": df.iloc[idx].get("anomaly_type", "unknown")
                            if "anomaly_type" in df.columns else "unknown",
            "mse_score":    f"{mse_score:.6f}",
            "tier":         tier_num,
            "timestamp":    str(df.iloc[idx].get("timestamp", "unknown")),
        }

        # Get LLM and run LangGraph investigation
        from agent.run_agent import get_llm
        from agent.react_agent import run_rca_investigation

        llm          = get_llm()
        agent_result = run_rca_investigation(
            alert         = alert,
            tools         = ALL_TOOLS,
            llm           = llm,
            system_prompt = RCA_SYSTEM_PROMPT,
            verbose       = False,
        )

        result["final_rca"] = agent_result.get("final_report", "")
        return result
