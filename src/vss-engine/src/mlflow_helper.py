######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
######################################################################################################
"""
MLflow observability helper for the VSS pipeline.

Provides:
  - init_mlflow()              — call once at ViaStreamHandler startup
  - start_request_run(req)     — call before _get_aggregated_summary() so the
                                 llama3-8b autolog trace is captured inside the run
  - end_request_run(req, chunks) — log all metrics/artifacts and end the run

Tracing strategy
----------------
* mlflow.openai.autolog() instruments every openai SDK call at the Python level.
  - cosmos VLM: runs in spawned subprocesses → autolog does NOT carry over to
    child processes. cosmos traces are not captured (multiprocessing limitation).
  - llama3-8b summarization: called via vss-ctx-rag in the MAIN process, inside
    _get_aggregated_summary(). If an MLflow run is active at that point, autolog
    captures the prompt/completion/tokens as a trace linked to the run.

* start_request_run() opens the MLflow run BEFORE _get_aggregated_summary() so
  the llama3-8b OpenAI call fires with an active run → the autolog trace gets the
  mlflow.runId tag → it appears in the Evaluations tab of the run in the UI.

* end_request_run() logs aggregated per-request metrics and artifacts then ends
  the run. Token counts in model-metrics come from the autolog trace.
"""

import logging
import os

logger = logging.getLogger(__name__)

_enabled: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patch_mlflow_http_client(workspace: str) -> None:
    """
    Inject X-MLFLOW-WORKSPACE into every MLflow HTTP call targeting the RHOAI
    MLflow server.  Two patches are required:

    1. mlflow.utils.rest_utils.http_request — tracking API calls
    2. requests.Session.request — artifact PUT/GET uploads
    """
    import mlflow.utils.rest_utils as rest_utils

    if not getattr(rest_utils, "_rhoai_workspace_patched", False):
        _orig_http_request = rest_utils.http_request

        def _patched_http_request(host_creds, endpoint, method, *args, extra_headers=None, **kwargs):
            merged = {"X-MLFLOW-WORKSPACE": workspace}
            if extra_headers:
                merged.update(extra_headers)
            return _orig_http_request(
                host_creds, endpoint, method, *args, extra_headers=merged, **kwargs
            )

        rest_utils.http_request = _patched_http_request
        rest_utils._rhoai_workspace_patched = True

    try:
        import requests
        if not getattr(requests.Session, "_rhoai_workspace_patched", False):
            _tracking_host = os.environ.get("MLFLOW_TRACKING_URI", "")
            _orig_session_request = requests.Session.request

            def _patched_session_request(self, method, url, *args, headers=None, **kwargs):
                if _tracking_host and url.startswith(_tracking_host):
                    headers = dict(headers or {})
                    headers.setdefault("X-MLFLOW-WORKSPACE", workspace)
                return _orig_session_request(self, method, url, *args, headers=headers, **kwargs)

            requests.Session.request = _patched_session_request
            requests.Session._rhoai_workspace_patched = True
    except Exception:
        pass

    logger.debug("MLflow HTTP client patched with X-MLFLOW-WORKSPACE: %s", workspace)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_mlflow() -> None:
    """
    Initialise MLflow tracking and enable OpenAI SDK auto-tracing.

    Environment variables (all optional):
      MLFLOW_TRACKING_URI      — tracking server URL; absent = disabled silently
      MLFLOW_WORKSPACE         — RHOAI workspace name (default: "default")
      MLFLOW_EXPERIMENT_NAME   — experiment name (default: "vss-pipeline")
      MLFLOW_TRACKING_INSECURE_TLS — set "true" to skip TLS verification
    """
    global _enabled

    if _enabled:
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        logger.info(
            "MLFLOW_TRACKING_URI not set — MLflow tracing disabled."
        )
        return

    try:
        import mlflow
        import mlflow.openai  # noqa: F401 — registers the autolog patch

        # ── Auth for RHOAI MLflow server ────────────────────────────────────
        _sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if os.path.isfile(_sa_token_path) and not os.environ.get("MLFLOW_TRACKING_TOKEN"):
            with open(_sa_token_path) as fh:
                os.environ["MLFLOW_TRACKING_TOKEN"] = fh.read().strip()

        if "MLFLOW_TRACKING_INSECURE_TLS" not in os.environ:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

        mlflow.set_tracking_uri(tracking_uri)

        # ── Inject RHOAI workspace header ────────────────────────────────────
        workspace = os.environ.get("MLFLOW_WORKSPACE", "default")
        _patch_mlflow_http_client(workspace)

        # ── Experiment setup ─────────────────────────────────────────────────
        exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "vss-pipeline")
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)

        # ── Enable LLM auto-tracing ──────────────────────────────────────────
        # Captures every openai SDK call (chat.completions.create) in the MAIN
        # process. cosmos VLM calls run in spawned subprocesses and are not
        # captured. llama3-8b summarization runs in-process and IS captured when
        # start_request_run() opens a run before _get_aggregated_summary().
        _autolog_kwargs = dict(log_traces=True, log_models=False, silent=True)
        try:
            mlflow.openai.autolog(**_autolog_kwargs)
        except TypeError:
            try:
                mlflow.openai.autolog(log_traces=True)
            except TypeError:
                mlflow.openai.autolog()

        _enabled = True
        logger.info("MLflow tracing enabled — tracking server: %s", tracking_uri)

    except Exception as exc:
        logger.warning("MLflow initialisation failed; tracing will be disabled: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Per-request run lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start_request_run(req_info) -> None:
    """
    Open an MLflow run for this video-processing request.

    Called from _process_output() BEFORE _get_aggregated_summary() so that the
    llama3-8b OpenAI call (which happens inside _get_aggregated_summary via the
    CA-RAG library) fires with an active run. mlflow.openai.autolog() then tags
    the resulting trace with mlflow.runId, making it appear in the Evaluations
    tab and populating the token-count model-metrics in the RHOAI MLflow UI.

    The run is stored as req_info._mlflow_run_id so end_request_run() can
    find it later (important for multi-threaded requests where thread-local
    active-run state may differ by the time end_request_run is called).
    """
    if not _enabled:
        return
    try:
        import mlflow

        request_id = str(getattr(req_info, "request_id", "unknown"))
        video_file = str(getattr(req_info, "file", ""))
        video_basename = os.path.basename(video_file)[:40] if video_file else "unknown"
        run_name = f"vss-{video_basename}-{request_id[:8]}"

        run = mlflow.start_run(run_name=run_name)
        # Stash the run ID on req_info so end_request_run can re-enter it even
        # if the thread-local active run has changed by then.
        req_info._mlflow_run_id = run.info.run_id
        logger.debug("MLflow run started: %s", run.info.run_id)
    except Exception as exc:
        logger.debug("MLflow start_request_run failed (non-fatal): %s", exc)


def end_request_run(req_info, chunk_responses) -> None:
    """
    Log all per-request metrics and artifacts, then end the MLflow run.

    Called from _process_output() after req_info.status = SUCCESSFUL so that
    end_time is already set. Re-enters the run using the stored run ID in case
    the thread-local active run context has changed.
    """
    if not _enabled:
        return
    try:
        import mlflow

        run_id = getattr(req_info, "_mlflow_run_id", None)
        if not run_id:
            return

        # ── Timing ──────────────────────────────────────────────────────────
        start = getattr(req_info, "start_time", 0) or 0
        end   = getattr(req_info, "end_time",   0) or 0
        e2e_s = max(0.0, end - start)
        ca_rag_s = max(0.0, getattr(req_info, "_ca_rag_latency", 0) or 0)

        # ── Aggregate VLM chunk stats ────────────────────────────────────────
        chunks = chunk_responses or []
        vlm_latencies_ms = [
            (cr.vlm_end_time - cr.vlm_start_time) * 1000
            for cr in chunks
            if getattr(cr, "vlm_start_time", 0) and getattr(cr, "vlm_end_time", 0)
               and cr.vlm_end_time > cr.vlm_start_time
        ]
        avg_vlm_ms = (sum(vlm_latencies_ms) / len(vlm_latencies_ms)) if vlm_latencies_ms else 0.0
        max_vlm_ms = max(vlm_latencies_ms) if vlm_latencies_ms else 0.0

        total_vlm_in  = sum(
            cr.vlm_stats.get("input_tokens",  0)
            for cr in chunks if getattr(cr, "vlm_stats", None)
        )
        total_vlm_out = sum(
            cr.vlm_stats.get("output_tokens", 0)
            for cr in chunks if getattr(cr, "vlm_stats", None)
        )

        # ── Artifact text ────────────────────────────────────────────────────
        captions = "\n\n".join(
            f"--- Chunk {cr.chunk.chunkIdx} "
            f"[{getattr(cr.chunk, 'start_pts', 0) / 1e9:.1f}s – "
            f"{getattr(cr.chunk, 'end_pts',   0) / 1e9:.1f}s] ---\n"
            f"{cr.vlm_response or '(no response)'}"
            for cr in sorted(chunks, key=lambda c: c.chunk.chunkIdx)
            if cr.vlm_response
        )
        final_summary = "\n".join(
            r.response
            for r in (getattr(req_info, "response", None) or [])
            if hasattr(r, "response") and r.response
        )

        # ── Link the llama3-8b autolog trace to this run ────────────────────
        # mlflow.openai.autolog() captures the llama3-8b call as a Trace, but
        # mlflow 3.0 PyPI doesn't auto-set mlflow.runId. Use the link-to-run
        # REST endpoint so the trace appears in the Evaluations tab.
        # The autolog trace is created asynchronously, so we allow up to 3s
        # for the upload to complete before calling link-to-run.
        nim_trace_id = mlflow.get_last_active_trace_id()
        if nim_trace_id:
            try:
                import time as _time, requests, json as _json, warnings as _w
                _w.filterwarnings("ignore")
                tracking_uri = mlflow.get_tracking_uri()
                token    = os.environ.get("MLFLOW_TRACKING_TOKEN", "")
                workspace = os.environ.get("MLFLOW_WORKSPACE", "default")
                hdrs = {
                    "Authorization": f"Bearer {token}",
                    "X-MLFLOW-WORKSPACE": workspace,
                    "Content-Type": "application/json",
                }

                # Wait for async trace upload (up to 3 s, 3 attempts)
                linked = False
                for attempt in range(3):
                    resp = requests.post(
                        f"{tracking_uri}/api/2.0/mlflow/traces/link-to-run",
                        json={"trace_ids": [nim_trace_id], "run_id": run_id},
                        headers=hdrs, verify=False, timeout=8,
                    )
                    if resp.status_code == 200:
                        linked = True
                        logger.info("llama3-8b trace %s linked to run %s",
                                    nim_trace_id[:8], run_id[:8])
                        break
                    _time.sleep(1)

                if linked:
                    # Extract token counts from the autolog span.
                    # mlflow.openai.autolog stores the full completion response
                    # (including usage) as JSON in the mlflow.spanOutputs attribute.
                    try:
                        trace = mlflow.get_trace(nim_trace_id)
                        for span in (trace.data.spans or []):
                            attrs = getattr(span, "attributes", {}) or {}
                            raw = attrs.get("mlflow.spanOutputs", "{}")
                            outputs = _json.loads(raw) if isinstance(raw, str) else raw
                            usage = outputs.get("usage", {}) if isinstance(outputs, dict) else {}
                            if usage.get("prompt_tokens"):
                                req_info._mlflow_nim_tokens_in  = int(usage["prompt_tokens"])
                            if usage.get("completion_tokens"):
                                req_info._mlflow_nim_tokens_out = int(usage["completion_tokens"])
                    except Exception:
                        pass
                else:
                    logger.debug("link-to-run: gave up after 3 attempts for trace %s",
                                 nim_trace_id[:8])
            except Exception as link_exc:
                logger.debug("trace link-to-run failed (non-fatal): %s", link_exc)

        # ── Re-enter the run and log everything ─────────────────────────────
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params({
                "request_id":  str(getattr(req_info, "request_id", ""))[:250],
                "stream_id":   str(getattr(req_info, "stream_id",  ""))[:250],
                "video_file":  str(getattr(req_info, "file", ""))[:250],
                "chunk_count": len(chunks),
                "enable_chat": str(getattr(req_info, "enable_chat", False)),
                "is_live":     str(getattr(req_info, "is_live",     False)),
            })
            metrics = {
                "e2e_latency_s":            round(e2e_s,      3),
                "ca_rag_latency_s":         round(ca_rag_s,   3),
                "avg_vlm_chunk_latency_ms": round(avg_vlm_ms, 1),
                "max_vlm_chunk_latency_ms": round(max_vlm_ms, 1),
                "vlm_total_input_tokens":   float(total_vlm_in),
                "vlm_total_output_tokens":  float(total_vlm_out),
                "chunk_count":              float(len(chunks)),
            }
            # llama3-8b token counts extracted from autolog trace (if linked)
            nim_in  = getattr(req_info, "_mlflow_nim_tokens_in",  0)
            nim_out = getattr(req_info, "_mlflow_nim_tokens_out", 0)
            if nim_in or nim_out:
                metrics["nim_llm2_input_tokens"]  = float(nim_in)
                metrics["nim_llm2_output_tokens"] = float(nim_out)
            mlflow.log_metrics(metrics)
            try:
                if captions:
                    mlflow.log_text(captions, "chunk_captions.txt")
                if final_summary:
                    mlflow.log_text(final_summary, "final_summary.txt")
            except Exception as art_exc:
                logger.debug("MLflow artifact logging skipped: %s", art_exc)

        logger.debug(
            "MLflow run ended: %s (e2e=%.1fs, chunks=%d)",
            run_id, e2e_s, len(chunks),
        )

    except Exception as exc:
        logger.debug("MLflow end_request_run failed (non-fatal): %s", exc)


# Keep old name as alias so existing patched via_stream_handler.py files
# continue to work without a re-patch cycle.
log_request = end_request_run
