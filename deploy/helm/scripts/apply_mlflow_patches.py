#!/usr/bin/env python3
"""
apply_mlflow_patches.py — idempotent MLflow instrumentation patcher.

Applies four patches to /tmp/via/via-engine/via_stream_handler.py:

  A  Import line: init_mlflow + start_request_run + end_request_run
  B  Call init_mlflow() in ViaStreamHandler.__init__
  C  Call start_request_run(req_info) BEFORE _get_aggregated_summary()
     so the llama3-8b autolog trace fires inside an active run and gets
     tagged with mlflow.runId (makes it appear in the Evaluations tab).
  D  Call end_request_run(req_info, chunk_responses) after SUCCESSFUL
     to log all metrics/artifacts and close the run.

The script is idempotent: safe to run on every pod restart.
"""

import os
import shutil
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
PATCHES_DIR = "/opt/mlflow-patches"
VIA_ENGINE  = "/tmp/via/via-engine"
HANDLER_SRC = os.path.join(VIA_ENGINE,  "via_stream_handler.py")
HANDLER_DST = HANDLER_SRC          # patch in-place in the writable copy
HELPER_SRC  = os.path.join(PATCHES_DIR, "mlflow_helper.py")
HELPER_DST  = os.path.join(VIA_ENGINE,  "mlflow_helper.py")

# ── Guard ──────────────────────────────────────────────────────────────────
for path, label in [(VIA_ENGINE, "writable via-engine copy"), (HELPER_SRC, "mlflow_helper.py")]:
    if not os.path.exists(path):
        print(f"[mlflow-patch] {label} not found at {path} — skipping", flush=True)
        sys.exit(0)

# ── 1. Copy mlflow_helper.py ───────────────────────────────────────────────
shutil.copy2(HELPER_SRC, HELPER_DST)
print(f"[mlflow-patch] Copied mlflow_helper.py -> {HELPER_DST}", flush=True)

# ── 2. Patch via_stream_handler.py ────────────────────────────────────────
if not os.path.isfile(HANDLER_SRC):
    print(f"[mlflow-patch] {HANDLER_SRC} not found — skipping handler patch", flush=True)
    sys.exit(0)

with open(HANDLER_SRC, "r") as fh:
    src = fh.read()

changed = False

# ── Patch A: import ────────────────────────────────────────────────────────
IMPORT_ANCHOR = "from via_logger import TimeMeasure, logger\n"
IMPORT_PATCH  = (
    "from mlflow_helper import (\n"
    "    init_mlflow, start_request_run as mlflow_start_request_run,\n"
    "    end_request_run as mlflow_end_request_run,\n"
    "    log_request as mlflow_log_request,\n"
    ")\n"
)
if "mlflow_start_request_run" not in src:
    # Remove any old single-line import first
    old_import = "from mlflow_helper import init_mlflow, log_request as mlflow_log_request\n"
    src = src.replace(old_import, "")
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_PATCH, 1)
    changed = True
    print("[mlflow-patch] Applied patch A: updated mlflow imports", flush=True)
else:
    print("[mlflow-patch] Patch A already present", flush=True)

# ── Patch B: init_mlflow() in __init__ ────────────────────────────────────
INIT_ANCHOR = "        self._start_ca_rag_alert_handler()\n"
INIT_PATCH  = "        init_mlflow()\n"
if INIT_PATCH not in src:
    src = src.replace(INIT_ANCHOR, INIT_ANCHOR + INIT_PATCH, 1)
    changed = True
    print("[mlflow-patch] Applied patch B: init_mlflow() call", flush=True)
else:
    print("[mlflow-patch] Patch B already present", flush=True)

# ── Patch C: start run BEFORE _get_aggregated_summary() ───────────────────
# This is the KEY fix: opening the run here means the llama3-8b OpenAI call
# (inside _get_aggregated_summary via CA-RAG) fires with an active run, so
# mlflow.openai.autolog() tags the trace with mlflow.runId.
# IMPORTANT: use the full 16-space indentation — a shorter anchor is a
# substring of the 16-space line and causes str.replace to mis-indent the code.
SUMMARY_ANCHOR = "                new_response = self._get_aggregated_summary(req_info, chunk_responses)\n"
SUMMARY_PATCH  = (
    "                mlflow_start_request_run(req_info)\n"
    "                new_response = self._get_aggregated_summary(req_info, chunk_responses)\n"
)
if "mlflow_start_request_run(req_info)" not in src:
    if SUMMARY_ANCHOR in src:
        src = src.replace(SUMMARY_ANCHOR, SUMMARY_PATCH, 1)
        changed = True
        print("[mlflow-patch] Applied patch C: mlflow_start_request_run() before summarization", flush=True)
    else:
        print("[mlflow-patch] Patch C anchor not found — skipping", flush=True)
else:
    print("[mlflow-patch] Patch C already present", flush=True)

# ── Patch D: end run after SUCCESSFUL status ───────────────────────────────
# Replace the old mlflow_log_request call (if present from a previous patch
# cycle) with the new mlflow_end_request_run, and add it if missing.
END_ANCHOR = (
    "                req_info.status = RequestInfo.Status.SUCCESSFUL\n"
    "                cuda.bindings.runtime.cudaProfilerStop()\n"
)
END_PATCH = (
    "                req_info.status = RequestInfo.Status.SUCCESSFUL\n"
    "                mlflow_end_request_run(req_info, chunk_responses)\n"
    "                cuda.bindings.runtime.cudaProfilerStop()\n"
)
OLD_LOG_LINE = "                mlflow_log_request(req_info, chunk_responses)\n"

# Upgrade old single-call patch to new split pair if present
if OLD_LOG_LINE in src and "mlflow_end_request_run" not in src:
    src = src.replace(
        "                req_info.status = RequestInfo.Status.SUCCESSFUL\n"
        + OLD_LOG_LINE,
        "                req_info.status = RequestInfo.Status.SUCCESSFUL\n"
        "                mlflow_end_request_run(req_info, chunk_responses)\n",
        1,
    )
    changed = True
    print("[mlflow-patch] Applied patch D: upgraded to mlflow_end_request_run()", flush=True)
elif "mlflow_end_request_run(req_info, chunk_responses)" not in src:
    if END_ANCHOR in src:
        src = src.replace(END_ANCHOR, END_PATCH, 1)
        changed = True
        print("[mlflow-patch] Applied patch D: mlflow_end_request_run() after SUCCESSFUL", flush=True)
    else:
        print("[mlflow-patch] Patch D anchor not found — skipping", flush=True)
else:
    print("[mlflow-patch] Patch D already present", flush=True)

with open(HANDLER_DST, "w") as fh:
    fh.write(src)

action = "patched in-place" if changed else "already up-to-date"
print(f"[mlflow-patch] via_stream_handler.py {action} at {HANDLER_DST}", flush=True)
print("[mlflow-patch] Done", flush=True)
