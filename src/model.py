"""MLflow pyfunc wrapper for TimesFM 2.5 + LoRA adapter.

Keeps the registered artefact small: only the LoRA weights live in the
artefact store, the 200 MB base checkpoint is fetched from the HF Hub
cache on the serving environment.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch


class TimesFMInverterModel(mlflow.pyfunc.PythonModel):
    """Predict next-N AC power per inverter using TimesFM 2.5 + LoRA.

    Input  schema: DataFrame with a `context` column holding list[float]
                   of length >= context_len.
    Output schema: DataFrame with a `forecast` column holding list[float]
                   of length horizon_len.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        # Lazy imports: peft + transformers are heavy and only present in
        # the model-serving environment, never in the local test env.
        from peft import PeftModel  # ty: ignore[unresolved-import]
        from transformers import TimesFm2_5ModelForPrediction  # ty: ignore[unresolved-import]

        base_id = self._read_text(context.artifacts["base_model_id"]).strip()
        hf_cache = context.artifacts.get("hf_cache")
        if hf_cache:
            os.environ.setdefault("HF_HOME", hf_cache)

        base = TimesFm2_5ModelForPrediction.from_pretrained(base_id, torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, context.artifacts["lora_dir"])
        self.model.eval()
        self.horizon_len = int(self._read_text(context.artifacts["horizon_len"]))
        self.context_len = int(self._read_text(context.artifacts["context_len"]))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path) as f:
            return f.read()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        if "context" not in model_input.columns:
            raise ValueError("model_input must have a 'context' column")

        ctx_arrays = []
        for c in model_input["context"]:
            arr = np.asarray(c, dtype=np.float32)
            if len(arr) < self.context_len:
                raise ValueError(
                    f"Each context must have at least {self.context_len} points; got {len(arr)}."
                )
            ctx_arrays.append(arr[-self.context_len :])

        batch = torch.tensor(np.stack(ctx_arrays), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(past_values=batch)
        preds = out.mean_predictions[:, : self.horizon_len].float().cpu().numpy()
        return pd.DataFrame({"forecast": [row.tolist() for row in preds]})
