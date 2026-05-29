# Video Search and Summarization on OpenShift AI

Deploy NVIDIA's video search and summarization AI blueprint on Red Hat OpenShift AI, with GPU MIG scheduling and MLflow observability.

> **Project home**
>
> This repository is part of the [Red Hat AI Quickstarts](https://www.redhat.com/en/blog/introducing-ai-quickstarts) initiative. It extends the upstream [NVIDIA AI Blueprint: Video Search and Summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) with OpenShift AI deployment support and MLflow observability. See [Fork Customizations](#fork-customizations) for details on what this quickstart adds over upstream.

## Table of Contents

- [Fork Customizations](#fork-customizations)
- [Detailed description](#detailed-description)
  - [Architecture diagrams](#architecture-diagrams)
- [Requirements](#requirements)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
  - [Required user permissions](#required-user-permissions)
- [Deploy](#deploy)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Validating the deployment](#validating-the-deployment)
  - [Delete](#delete)
- [Repository structure](#repository-structure)
- [References](#references)
- [Other deployment options](#other-deployment-options)
- [MLflow observability](#mlflow-observability)
- [Known CVEs](#known-cves)
- [Tags](#tags)

## Fork Customizations

This quickstart extends the upstream [NVIDIA VSS Blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) as part of the [Red Hat AI Quickstarts](https://www.redhat.com/en/blog/introducing-ai-quickstarts) initiative, adding:

- **OpenShift AI deployment** — A full OpenShift overlay strategy using KServe `InferenceService` resources, OpenShift Security Context Constraints, and a single Helm command. All four GPU models (Cosmos VLM, Llama LLM, Embedqa, Reranker) run as KServe `InferenceService` objects managed by RHOAI. See the [OpenShift Deployment Guide](deploy/helm/openshift-deployment.md).
- **MIG GPU scheduling** — Validated MIG configuration for running all four GPU workloads on two physical H100 SXM5 96GB GPUs. Includes MIG setup commands and a `values-openshift.yaml` that sets `nvidia.com/gpu: 0` and explicit MIG slice resources to prevent the RHOAI hardware profile controller from injecting conflicting GPU requests.
- **MLflow observability** — Per-request pipeline telemetry logged to the RHOAI MLflow tracking server without rebuilding the container image. A pod-startup patcher injects call sites into the VSS engine and an MLflow helper logs latency, token counts, VLM captions, and full LLM traces. See the [MLflow Observability](#mlflow-observability) section.
- **LLM model size optimization** — Default overrides in `values-openshift.yaml` switch the upstream 70B LLM to `meta/llama-3.1-8b-instruct` (1 GPU) to fit GPU-constrained environments, with documented steps to switch back to 70B.
- **Red Hat UI rebrand** — Logo, colors, and fonts replaced in the Gradio web UI without rebuilding the container image. The rebranded assets are injected at pod startup via a ConfigMap (`vss-ui-rebrand-cm`) and the same runtime-patching mechanism used for MLflow. Primary color: `#EE0000`. Font: Red Hat Text / Red Hat Display. Source files: `src/vss-engine/src/client/assets/`.

> **See also:** [Introducing Red Hat AI Quickstarts](https://www.redhat.com/en/blog/introducing-ai-quickstarts) for the broader context on how Red Hat is making AI blueprints accessible on OpenShift.

## Detailed description

The NVIDIA AI Blueprint for Video Search and Summarization addresses the challenge of efficiently analyzing and summarizing large volumes of video data. Operations teams across manufacturing, logistics, and facilities management generate continuous video streams — from warehouse floors and production lines to loading docks and secure spaces — that are too voluminous to review manually. This quickstart enables teams to query that footage in natural language and receive accurate, cited summaries automatically.

The blueprint deploys a full AI pipeline: a Vision Language Model (Cosmos-Reason2-8B) captions each video frame, a retrieval-augmented generation (CA-RAG) module backed by vector and graph databases answers natural language queries, and an LLM (Llama-3.1-8B) generates summaries and handles chat. This quickstart adapts the upstream NVIDIA blueprint for Red Hat OpenShift AI using KServe `InferenceService` resources, OpenShift-compatible security contexts, and a single Helm command deployment.

MLflow observability is included out of the box. Every video summarization request produces a logged MLflow run in the RHOAI MLflow UI with end-to-end latency, token counts, per-chunk VLM captions, and the full LLM trace — giving operations teams and AI engineers visibility into pipeline performance without modifying the container image.

### Architecture diagrams

![VSS architecture showing video ingestion, VLM captioning, CA-RAG retrieval, and LLM summarization](docs/images/vss_architecture.png)

Video is decoded into chunks by the VSS pipeline pod. Each chunk is captioned by the Cosmos VLM and indexed into Milvus (vector search), ArangoDB, and Neo4j (graph databases). User queries flow through embedding → vector search → reranking → LLM to return cited summaries. Alerts fire when captions match user-defined keywords.

## Requirements

### Hardware requirements

This quickstart was validated on **2x NVIDIA H100 SXM5 96GB** GPUs with Multi-Instance GPU (MIG) enabled. All four GPU workloads share the two physical GPUs via MIG slices:

**GPU 0 — VLM only:**

| MIG slice | VRAM | Workload |
|-----------|------|----------|
| `mig-7g.94gb` | 94 GB | Cosmos-Reason2-8B VLM (`cosmos` InferenceService) |

**GPU 1 — LLM + embedding + reranking + VSS pipeline:**

| MIG slice | VRAM | Workload |
|-----------|------|----------|
| `mig-3g.47gb` | 47 GB | Llama-3.1-8B LLM (`llama3-8b` InferenceService) |
| `mig-2g.24gb` | 24 GB | VSS pipeline pod (NVDEC video frame decoding) |
| `mig-1g.12gb` | 12 GB | Embedqa (`embedqa` InferenceService) |
| `mig-1g.12gb` | 12 GB | Llama-NemoTron-Rerank-1B (`llama-rerank` InferenceService) |

**Minimum requirements for reproduction:**

- 2x H100 SXM5 96GB (or equivalent GPU with ≥94 GB VRAM per VLM slice). The Cosmos-Reason2-8B VLM requires ~85 GB VRAM; a `mig-7g.94gb` slice is the minimum that fits it.
- Alternatively, 4 separate GPUs each with ≥48 GB VRAM (e.g. A100 80GB) using `resources` overrides in `values-openshift.yaml`.
- NVIDIA A10G (22 GB) is **not sufficient** for the VLM.
- ~64 GiB RAM and ~32 vCPU across worker nodes for non-GPU pods (Elasticsearch alone requests 16 GiB).

For other validated GPU topologies (non-OpenShift), see the NVIDIA [supported platforms](https://docs.nvidia.com/vss/latest/content/supported_platforms.html#supported-platforms) page.

### Software requirements

- Red Hat OpenShift 4.12 or later
- Red Hat OpenShift AI (RHOAI) 2.x with KServe model serving configured
- NVIDIA GPU Operator with MIG support enabled
- Helm 3.x
- OpenShift CLI (`oc`) 4.12 or later

### Required user permissions

Deploying this quickstart requires **cluster-admin** access. The Helm chart creates a `ServiceAccount`, a `RoleBinding` granting the `anyuid` Security Context Constraint, and an OpenShift `Route` — all of which require elevated permissions.

## Deploy

### Prerequisites

Before deploying, ensure you have:

- Access to a Red Hat OpenShift cluster with RHOAI and the NVIDIA GPU Operator installed
- `oc` CLI installed and authenticated as a cluster-admin user
- `helm` 3.x installed
- An NGC API key from [NGC](https://org.ngc.nvidia.com/setup/api-keys) or [build.nvidia.com](https://build.nvidia.com/) (requires NVIDIA AI Enterprise license)
- A HuggingFace token with the [Cosmos-Reason2-8B model license](https://huggingface.co/nvidia/Cosmos-Reason2-8B) accepted
- MIG configured on your GPU nodes (see [deploy/helm/openshift-deployment.md](deploy/helm/openshift-deployment.md#tested-hardware) for setup commands)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/rh-ai-quickstart/vss-quickstart.git
cd vss-quickstart
```

2. Create the deployment namespace:

```bash
oc new-project vss
```

3. Export your credentials:

```bash
export NGC_API_KEY="<your NGC API key>"
export HF_TOKEN="<your HuggingFace token>"
```

4. Verify your GPU node tolerations match the defaults in `values-openshift.yaml` (default taint key is `nvidia.com/gpu`):

```bash
oc get nodes -l nvidia.com/gpu.present=true -o name | \
  xargs -I{} oc describe {} | grep -A1 Taints
```

5. Install using Helm:

```bash
helm upgrade --install vss deploy/helm/nvidia-blueprint-vss-2.4.1.tgz \
  -n vss \
  -f deploy/helm/values-openshift.yaml \
  --set nvcf.dockerRegSecrets[0].password="$NGC_API_KEY" \
  --set nvcf.additionalSecrets[0].stringData.value="$NGC_API_KEY" \
  --set nvcf.additionalSecrets[1].stringData.value="$HF_TOKEN"
```

The chart will create the service account, SCC role binding, secrets, route, and all 11 pods. GPU pods (`vss`, `llama3-8b`, `embedqa`, `llama-rerank`) may take 20–30 minutes on first deploy while model weights download.

> **After every `helm upgrade`:** The chart resets the `vss-scripts-cm` ConfigMap to the upstream NVIDIA default, removing the MLflow and UI rebranding startup steps. Re-apply the patched `start.sh` immediately after any upgrade:
> ```bash
> START_SH_CONTENT=$(cat /tmp/opencode/start.sh) && \
> oc patch cm vss-scripts-cm -n vss --type=merge \
>   -p "{\"data\":{\"start.sh\":$(echo "$START_SH_CONTENT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}}" && \
> oc delete pod -n vss -l app.kubernetes.io/name=vss
> ```
> The patched `start.sh` is stored at `/tmp/opencode/start.sh` on the host that ran the deployment. For a persistent reference, see [deploy/helm/openshift-deployment.md](deploy/helm/openshift-deployment.md).

For full configuration options (Llama 70B, custom tolerations, manual secrets), see [deploy/helm/openshift-deployment.md](deploy/helm/openshift-deployment.md).

### Validating the deployment

1. Check all pods are running:

```bash
oc get pods -n vss
```

All pods should reach `Running 1/1`. Expected pods: `vss-vss-deployment`, `nim-llm-predictor-default-*`, `nemo-embedding-predictor-default-*`, `nemo-rerank-predictor-default-*`, and 7 infrastructure pods (Milvus, MinIO, etcd, Elasticsearch, ArangoDB, Neo4j, MinIO for object storage).

2. Get the UI URL:

```bash
oc get route vss-ui -n vss -o jsonpath='{.spec.host}'
```

3. Open `https://<route-host>` in a browser. Upload a video file or enter an RTSP stream URL to begin.

### Delete

To completely remove the deployment:

1. Uninstall the Helm release and delete all PVCs:

```bash
helm uninstall vss -n vss
oc delete pvc --all -n vss
```

2. Delete the project:

```bash
oc delete project vss
```

## Repository structure

```
.
├── deploy/
│   ├── helm/
│   │   ├── nvidia-blueprint-vss-2.4.1.tgz   # Helm chart (use this for deployment)
│   │   ├── values-openshift.yaml             # OpenShift-specific Helm value overrides
│   │   ├── is-sr.yaml                        # Reference InferenceService + ServingRuntime definitions
│   │   ├── job-pvc.yaml                      # PVC and model-download Job definitions
│   │   ├── mlflow.yaml                       # RHOAI MLflow CR (apply once per cluster)
│   │   ├── mlflow-standalone.yaml            # Standalone MLflow for development
│   │   ├── openshift-deployment.md           # Full deployment runbook
│   │   └── scripts/
│   │       └── apply_mlflow_patches.py       # Pod-startup patcher for MLflow instrumentation
│   └── docker/                               # Docker Compose deployment configs
├── src/
│   └── vss-engine/src/
│       ├── mlflow_helper.py                  # MLflow helper: init, run lifecycle, trace linking
│       ├── via_demo_client.py                # Gradio app entry point (title rebranded)
│       └── client/assets/
│           ├── app_bar.html                  # Header bar — Red Hat logo + Red Hat Text font
│           ├── kaizen-theme.css              # Gradio CSS overrides — Red Hat brand colors + fonts
│           └── kaizen-theme.json             # Gradio theme tokens — Red Hat red palette
├── examples/                                 # Usage notebooks and example configs
└── docs/
    └── images/                               # Architecture diagrams and screenshots
```

## References

- [NVIDIA VSS Documentation](https://docs.nvidia.com/vss/latest/index.html)
- [NVIDIA AI Blueprint: Video Search and Summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
- [Red Hat OpenShift AI Documentation](https://docs.redhat.com/en/openshift-ai)
- [NVIDIA GPU Operator MIG documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html)
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [NVIDIA Supported Platforms for VSS](https://docs.nvidia.com/vss/latest/content/supported_platforms.html)

## Other deployment options

This quickstart focuses on the OpenShift AI deployment. The upstream NVIDIA blueprint also supports:

- **Docker Compose** (development/testing): see [deploy/docker/README.md](deploy/docker/README.md)
- **Vanilla Kubernetes Helm** (production non-OpenShift): see the [NVIDIA VSS Helm documentation](https://docs.nvidia.com/vss/latest/content/vss_dep_helm.html)
- **Brev Launchable** (cloud, zero hardware setup): see [deploy/1_Deploy_VSS_docker_Crusoe.ipynb](deploy/1_Deploy_VSS_docker_Crusoe.ipynb)

## MLflow observability

VSS on OpenShift is instrumented with [MLflow](https://mlflow.org/) to log per-request pipeline telemetry to the RHOAI MLflow tracking server. Every video summarization produces one MLflow run in the `vss-pipeline` experiment.

### What is logged per request

| Category | Data |
|----------|------|
| **Parameters** | `request_id`, `video_file`, `stream_id`, `chunk_count`, `enable_chat`, `is_live` |
| **Metrics** | `e2e_latency_s`, `ca_rag_latency_s`, `avg_vlm_chunk_latency_ms`, `max_vlm_chunk_latency_ms`, `vlm_total_input_tokens`, `vlm_total_output_tokens`, `llm_input_tokens`, `llm_output_tokens` |
| **Artifacts** | `chunk_captions.txt` (per-chunk VLM captions), `final_summary.txt` (aggregated summary) |
| **Evaluations** | Full LLM trace (prompt, completion, token counts) via `mlflow.openai.autolog()` |

### Key files

| File | Purpose |
|------|---------|
| `src/vss-engine/src/mlflow_helper.py` | MLflow helper: init, workspace auth, run lifecycle, trace linking, token extraction |
| `deploy/helm/scripts/apply_mlflow_patches.py` | Pod-startup patcher — idempotent, runs on every pod restart |
| `deploy/helm/mlflow.yaml` | RHOAI MLflow CR — apply once per cluster in `redhat-ods-applications` |
| `deploy/helm/mlflow-standalone.yaml` | Optional standalone MLflow for development (no auth, port 5000) |

For full setup instructions see [deploy/helm/openshift-deployment.md § MLflow Observability](deploy/helm/openshift-deployment.md#mlflow-observability).

> **Note:** Cosmos VLM calls run in spawned subprocesses and cannot be traced by `mlflow.openai.autolog()`. Only the in-process LLM call is traced.

## Known CVEs

VSS Engine 2.4.1 container has the following known CVEs:

| CVE | Description |
|-----|-------------|
| [GHSA-58pv-8j8x-9vj2](https://github.com/jaraco/jaraco.context/security/advisories/GHSA-58pv-8j8x-9vj2) | Impacts jaraco.context < 6.1.0. Does not affect VSS — it does not install user-provided Python packages. |
| [CVE-2025-69223](https://github.com/advisories/GHSA-6mq8-rvhq-8wgg) | Impacts aiohttp < 3.13.3. Does not affect VSS — aiohttp is included only as a private ray dependency and ray is not used by VSS. |
| [GHSA-f83h-ghpp-7wcc](https://github.com/advisories/GHSA-f83h-ghpp-7wcc) | Impacts pdfminer.six < 20251230. Does not affect VSS — it does not implement PDF parsing. |
| [CVE-2025-68973](https://nvd.nist.gov/vuln/detail/CVE-2025-68973) | Impacts gnupg < 2.4.8. Does not affect VSS — it does not implement GPG encryption. |
| [GHSA-mcmc-2m55-j8jj](https://github.com/advisories/GHSA-mcmc-2m55-j8jj) [GHSA-mrw7-hf4f-83pf](https://github.com/advisories/GHSA-mrw7-hf4f-83pf) [CVE-2025-62372](https://github.com/advisories/GHSA-pmqf-x6x8-p7qw) | Impacts vLLM < 0.11.1. Does not affect VSS — it does not support user-provided embeddings. |
| [CVE-2026-21441](https://github.com/advisories/GHSA-38jv-5279-wg99) | Impacts urllib3 < 2.6.3. Does not affect VSS — it does not access user-provided URLs at runtime. |
| [CVE-2025-3887](https://ubuntu.com/security/CVE-2025-3887) | Impacts GStreamer H.265 codec parser. Malformed streams can cause a stack overflow. Users must ensure malicious H.265 streams are not added to VSS. Can be remedied by building and installing the GStreamer 1.24.2 codec parser with the [patch from gstreamer.freedesktop.org](https://gstreamer.freedesktop.org/security/sa-2025-0001.html). |
| [GHSA-rcfx-77hg-w2wv](https://github.com/advisories/GHSA-rcfx-77hg-w2wv) | Impacts fastmcp < 2.14.0. Does not affect VSS — it already uses an updated version of the MCP SDK. |

## Tags

**Title:** Video Search and Summarization on OpenShift AI  
**Description:** Deploy NVIDIA's video search and summarization AI blueprint on Red Hat OpenShift AI, with GPU MIG scheduling and MLflow observability.  
**Industry:** Manufacturing  
**Product:** OpenShift AI  
**Use case:** Video analytics, observability  
**Partner:** NVIDIA  
**Contributor org:** Red Hat
