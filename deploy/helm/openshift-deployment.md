# Deploying NVIDIA VSS on Red Hat OpenShift AI

This guide covers deploying the [NVIDIA Video Search and Summarization (VSS) v2.4.1](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) blueprint on Red Hat OpenShift AI (RHOAI) using a single Helm command. All OpenShift-specific adaptations are applied at install time - no post-deploy patching is required.

## Table of Contents

- [What We're Deploying](#what-were-deploying)
- [OpenShift Overlay Strategy](#openshift-overlay-strategy)
- [Tested Hardware](#tested-hardware)
- [Prerequisites](#prerequisites)
- [Configuration Reference](#configuration-reference)
- [Deployment](#deployment)
- [Model Size Optimization](#model-size-optimization)
- [MLflow Observability](#mlflow-observability)
- [OpenShift-Specific Challenges and Solutions](#openshift-specific-challenges-and-solutions)
- [Deployment Files](#deployment-files)

---

## What We're Deploying

VSS is a video analytics platform that ingests video (file upload or RTSP live stream), captions individual frames using a Vision Language Model, and makes the content searchable via natural language. It combines:

- **Vision Language Model** (Cosmos-Reason2-8B) for frame-by-frame video captioning
- **RAG pipeline** with vector search (Milvus), embedding, and reranking for natural language queries
- **LLM inference** for chat, summaries, and notifications
- **Event alerting** triggered when captions match user-defined keywords
- **Graph and document databases** (ArangoDB, Neo4j, Elasticsearch) for metadata and knowledge

**Data flow:**

- **Upload:** video → vss (Cosmos captions each frame) → nemo-embedding → Milvus
- **Search:** user query → nemo-embedding → Milvus (vector search) → nemo-rerank → nim-llm → response
- **Alerts:** vss monitors captions for user-defined event keywords

The Helm chart deploys 11 pods. Four require GPUs (`vss`, `nim-llm`, `nemo-embedding`, `nemo-rerank`); the rest are infrastructure services (Milvus, MinIO, etcd, Elasticsearch, ArangoDB, Neo4j).

---

## OpenShift Overlay Strategy

1. We introduced an `openshift.enabled` flag in the pre-existing [`values.yaml`](nvidia-blueprint-vss-2.4.1.tgz) file. This is the only non-additive change we made on top of the NVIDIA upstream codebase.
2. We added OpenShift AI model serving resources to run GPU models as KServe `InferenceService` instead of basic pods. These are enabled via `openshift.ai.enabled` in `values-openshift.yaml`.
3. All the values and resources required for the OpenShift deployment are placed in two dedicated files: [`values-openshift.yaml`](values-openshift.yaml) and [`templates/openshift.yaml`](nvidia-blueprint-vss-2.4.1.tgz) (inside the chart), as well as [`templates/openshift-ai.yaml`](nvidia-blueprint-vss-2.4.1.tgz) for OpenShift AI resources.
4. NGC and HuggingFace secrets are created via NVIDIA's built-in `nvcf` mechanism (`templates/nvcf_secrets.yaml`), configured in `values-openshift.yaml`. Service credential secrets (ArangoDB, MinIO, Neo4j) are created by `templates/openshift.yaml` when `openshift.secrets.create` is `true`, or can be created manually before install. ServiceAccount, SCC RoleBinding, and Route are also created by `templates/openshift.yaml`, gated by the `openshift.enabled` flag.
5. The deployment requires only two files and a single Helm command — no external scripts or separate OpenShift directory. See the [Deployment](#deployment) section below.

---

## Tested Hardware

This deployment was validated on the following cluster configuration:

**Cluster:** OpenShift 4.18 on NVIDIA LaunchPad (on-premises bare metal)

### GPU node

| GPU | VRAM | Count | MIG enabled |
|-----|------|-------|-------------|
| NVIDIA H100 SXM5 96GB | 96 GB each | 2 | Yes |

All four GPU workloads (Cosmos VLM, llama3-8b LLM, embedqa, llama-rerank) run on two physical H100 SXM5 96GB GPUs using **Multi-Instance GPU (MIG)**. MIG partitions each physical GPU into isolated slices, each with dedicated memory and compute, so multiple model serving containers can share the same GPU without memory contention.

### MIG configuration

The two H100 GPUs are configured as follows:

**GPU 0 — VLM only:**

| MIG slice | VRAM | Workload |
|-----------|------|----------|
| `1x mig-7g.94gb` | 94 GB | Cosmos-Reason2-8B VLM (`cosmos` InferenceService) |

**GPU 1 — LLM + embedding + rerank + VSS pipeline:**

| MIG slice | VRAM | Workload |
|-----------|------|----------|
| `1x mig-3g.47gb` | 47 GB | Llama-3.1-8B LLM (`llama3-8b` InferenceService) |
| `1x mig-2g.24gb` | 24 GB | VSS pipeline pod (NVDEC video frame decoding) |
| `1x mig-1g.12gb` | 12 GB | Embedqa (`embedqa` InferenceService) |
| `1x mig-1g.12gb` | 12 GB | Llama-NemoTron-Rerank-1B (`llama-rerank` InferenceService) |

> **Note:** The `mig-7g.94gb` slice occupies all 7 compute instances of one GPU — no other workload can share it. The second GPU's slices (3+2+1+1 = 7 compute instances) also fill that GPU completely. This leaves zero spare GPU capacity; adding workloads requires additional GPU hardware.

### Configuring MIG on H100 SXM5

MIG must be enabled before deploying VSS. On the node running the H100s:

```bash
# Enable MIG mode on both GPUs (requires root; node reboot may be needed)
nvidia-smi -i 0 --mig-mode=Enabled
nvidia-smi -i 1 --mig-mode=Enabled

# GPU 0: one full 7g.94gb instance
nvidia-smi mig -i 0 -cgi 0 -C   # creates 1x 7g.94gb + 1x compute instance

# GPU 1: 3g.47gb + 2g.24gb + 2x 1g.12gb
nvidia-smi mig -i 1 -cgi 9,15,19,19 -C
# Profile IDs: 9=3g.47gb, 15=2g.24gb, 19=1g.12gb (verify with: nvidia-smi mig -lgip)
```

If deploying on OpenShift with the NVIDIA GPU Operator, configure MIG via the `ClusterPolicy`:

```yaml
# In the GPU Operator ClusterPolicy
mig:
  strategy: mixed   # allows different slice sizes on the same node
```

Then label nodes with the desired MIG profile per device:

```bash
oc label node <gpu-node> nvidia.com/mig.config=all-disabled   # reset first
oc label node <gpu-node> nvidia.com/mig.config=custom          # then apply custom
```

For more detail see the [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) and the [GPU Operator MIG documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html).

### Worker nodes (non-GPU)

| vCPU | RAM | Count | Role in VSS |
|------|-----|-------|-------------|
| 32+ | 64+ GiB | 3+ | Milvus, MinIO, etcd, Elasticsearch, ArangoDB, Neo4j |

### Minimum hardware for reproduction

Any cluster with the following should work:

- **2x H100 SXM5 96GB** (or equivalent ≥96 GB GPU) with MIG configured exactly as above. The Cosmos-Reason2-8B VLM alone requires ~85 GB VRAM at `gpu-memory-utilization=0.93`; a 7g.94gb MIG slice (94 GB) is the minimum that fits it.
- As an alternative without MIG, **4 separate GPUs** each with ≥48 GB VRAM (e.g. L40S 46 GB is marginal for the VLM) can work with appropriate `resources` overrides in `values-openshift.yaml`.
- NVIDIA A10G (22 GB) is **not sufficient** for the VLM.
- **~64 GiB RAM** and **~32 vCPU** across worker nodes for non-GPU pods (Elasticsearch alone requests 16 GiB).
- To run **Llama 70B** instead of 8B, the `llama3-8b` InferenceService requires a `3g.47gb` slice at minimum but will be slow; a dedicated 4-GPU node with A100 80GB or higher is recommended. Edit `values-openshift.yaml`: update `llama3-8b` InferenceService resources and `global.ucfGlobalEnv[0].value`. See NVIDIA's [supported platforms](https://docs.nvidia.com/vss/latest/content/supported_platforms.html#supported-platforms) for validated GPU topologies.

---

## Prerequisites

- OpenShift CLI (`oc`) 4.12+ installed and authenticated with cluster-admin privileges
- Helm 3.x installed
- Red Hat OpenShift AI (RHOAI) operator installed on the cluster and configured for model serving (KServe)
- NVIDIA GPU Operator installed on the cluster and `nvidia.com/gpu` resource is allocatable
- NGC API key from [NGC](https://org.ngc.nvidia.com/setup/api-keys) or [build.nvidia.com](https://build.nvidia.com/) (requires NVIDIA AI Enterprise license)
- HuggingFace token with the [Cosmos-Reason2-8B license](https://huggingface.co/nvidia/Cosmos-Reason2-8B) accepted
- GPU nodes are ready: `oc get nodes -l nvidia.com/gpu`
- GPU node taint keys identified: `oc describe node <gpu-node> | grep -A5 Taints`
- Helm chart `deploy/helm/nvidia-blueprint-vss-2.4.1.tgz` present in this repo

---

## Configuration Reference

All configuration is defined in [`values-openshift.yaml`](values-openshift.yaml). Only API keys are passed via `--set` at install time.

### Required (passed via `--set`)

| Parameter | Description |
|-----------|-------------|
| `nvcf.dockerRegSecrets[0].password` | NGC API key for container image pulls |
| `nvcf.additionalSecrets[0].stringData.value` | NGC API key for NIM runtime authentication |
| `nvcf.additionalSecrets[1].stringData.value` | HuggingFace token for the gated Cosmos-Reason2-8B model |

### Configurable in `values-openshift.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nim-llm.model.name` | `meta/llama-3.1-8b-instruct` | NIM LLM model name |
| `nim-llm.image.repository` | `nvcr.io/nim/meta/llama-3.1-8b-instruct` | NIM LLM container image |
| `nim-llm.image.tag` | `latest` | NIM LLM image tag |
| `nim-llm.resources.limits.nvidia.com/gpu` | `1` | GPUs allocated to nim-llm (4 for 70B) |
| `vss.resources.limits.nvidia.com/gpu` | `1` | GPUs allocated to the vss VLM (2 for 70B) |
| `global.ucfGlobalEnv[0].value` | `meta/llama-3.1-8b-instruct` | LLM_MODEL env var propagated to all pods |
| `global.ucfGlobalEnv[1].value` | `true` | DISABLE_GUARDRAILS (recommended `true` for 8B) |
| Tolerations (vss, nim-llm, nemo-*) | `nvidia.com/gpu` | GPU node taint keys — update for your cluster |

---

## Deployment

### 1. Prepare your environment

```bash
# Verify OpenShift connectivity
oc login --token=$OPENSHIFT_TOKEN --server=$OPENSHIFT_CLUSTER_URL
oc whoami
oc cluster-info

# Verify Helm is installed
helm version

# Create and switch to the deployment namespace
oc new-project vss
```

### 2. Configure secrets

All secrets can be created either by the Helm chart or manually before install.

Export your API keys:

```bash
export NGC_API_KEY="<your NGC key>"
export HF_TOKEN="<your HuggingFace token>"
```

**Option A: Let Helm create all secrets (default)**

NGC and HuggingFace secrets are created via NVIDIA's built-in `nvcf` mechanism (`templates/nvcf_secrets.yaml`), pre-configured in `values-openshift.yaml` — just pass API keys via `--set` at install time. Service credential secrets (ArangoDB, MinIO, Neo4j) are created by `templates/openshift.yaml` when `openshift.secrets.create: true` (the default). No extra steps needed.

> **Key rotation:** `nvcf` secrets use `helm.sh/hook: pre-install` and won't update on `helm upgrade`. To rotate keys, delete the secret first (`oc delete secret <name> -n vss`) then re-run `helm upgrade`.

**Option B: Create all secrets manually (recommended for production)**

Clear `nvcf.dockerRegSecrets` and `nvcf.additionalSecrets` (set to `[]`) and set `openshift.secrets.create=false` in your values, then create all secrets manually:

> **Important:** Secret names are hardcoded in the chart's sub-charts. They must match exactly as shown below — regardless of the Helm release name.

```bash
# NGC container registry pull secret
oc create secret docker-registry ngc-docker-reg-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  -n vss

# NGC API key
oc create secret generic ngc-api-key-secret \
  --from-literal=NGC_API_KEY="$NGC_API_KEY" \
  -n vss

# HuggingFace token
oc create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n vss

# ArangoDB credentials
oc create secret generic arango-db-creds-secret \
  --from-literal=username=root \
  --from-literal=password=password \
  -n vss

# MinIO credentials
oc create secret generic minio-creds-secret \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=minioadmin \
  -n vss

# Neo4j credentials
oc create secret generic graph-db-creds-secret \
  --from-literal=username=neo4j \
  --from-literal=password=password \
  -n vss
```

### 3. Configure deployment

Export the LLM model. All other configuration (GPU counts, tolerations, security contexts) is defined in [`values-openshift.yaml`](values-openshift.yaml).

```bash
# LLM model — must match nim-llm.model.name in values-openshift.yaml
export LLM_MODEL="meta/llama-3.1-8b-instruct"
```

> **Important:** GPU pods will stay `Pending` if your cluster's GPU node taints don't match the tolerations in `values-openshift.yaml`. The default key is `nvidia.com/gpu`. Find your taint keys and update the `gpuTolerations` anchor in `values-openshift.yaml` before installing:
> ```bash
> oc get nodes -l nvidia.com/gpu.present=true -o name | \
>   xargs -I{} oc describe {} | grep -A1 Taints
> ```

> **Llama 70B:** Set `LLM_MODEL=meta/llama-3.1-70b-instruct` and edit `values-openshift.yaml`: `nim-llm.model.name`, `nim-llm.image.repository`, `global.ucfGlobalEnv[0].value`, `nim-llm.resources.limits.nvidia.com/gpu=4`, `vss.resources.limits.nvidia.com/gpu=2`.

### 4. Install the Helm chart

```bash
helm upgrade --install vss deploy/helm/nvidia-blueprint-vss-2.4.1.tgz \
  -n vss \
  -f deploy/helm/values-openshift.yaml \
  --set nvcf.dockerRegSecrets[0].password="$NGC_API_KEY" \
  --set nvcf.additionalSecrets[0].stringData.value="$NGC_API_KEY" \
  --set nvcf.additionalSecrets[1].stringData.value="$HF_TOKEN"
```

> **If guardrails are enabled** (`DISABLE_GUARDRAILS: "false"` in `values-openshift.yaml`), add the following `--set` flags to override the guardrails model (chart defaults to 70B):
> ```bash
> --set "vss.configs.guardrails_config\.yaml.models[0].engine=nim" \
> --set-string "vss.configs.guardrails_config\.yaml.models[0].model=$LLM_MODEL" \
> --set "vss.configs.guardrails_config\.yaml.models[0].parameters.base_url=http://llm-nim-svc:8000/v1" \
> --set "vss.configs.guardrails_config\.yaml.models[0].type=main"
> ```

The Helm chart will:

1. Create the `vss-sa` service account
2. Create a RoleBinding granting the `anyuid` SCC to `vss-sa`
3. Create all required secrets (NGC registry, NGC API key, HF token, ArangoDB, MinIO, Neo4j)
4. Create an OpenShift Route for external UI access
5. Deploy all 11 pods with OpenShift-compatible security contexts, tolerations, and storage

### 5. Verify the deployment

Check that all pods are running:

```bash
oc get pods -n vss
```

All pods should reach `Running 1/1`. GPU pods (`nim-llm`, `nemo-embedding`, `nemo-rerank`, `vss`) may take 20-30 minutes on first deploy while model weights are downloaded.

**Expected pods:**

| Pod | Purpose | GPU |
|-----|---------|-----|
| `vss-vss-deployment` | Core pipeline + VLM (Cosmos-Reason2-8B) | Yes |
| `nim-llm-predictor-default-*` | LLM inference (Llama 8B/70B) via OpenShift AI | Yes |
| `nemo-embedding-predictor-default-*` | Vector embedding generation via OpenShift AI | Yes |
| `nemo-rerank-predictor-default-*` | Document reranking via OpenShift AI | Yes |
| `milvus-milvus-deployment` | Vector database | No |
| `milvus-minio-*` | Object storage for Milvus | No |
| `etcd-*` | Milvus metadata store | No |
| `elasticsearch-*` | Search engine | No |
| `arango-db-*` | Graph database | No |
| `neo-4-j-*` | Graph database | No |
| `minio-*` | Object storage | No |

Verify the OpenShift resources were created by the Helm template:

```bash
# Route for external access — the UI URL is in the HOST/PORT column
oc get route vss-ui -n vss

# RoleBinding granting anyuid SCC to vss-sa
oc get rolebinding vss-anyuid-scc -n vss

# ServiceAccount
oc get serviceaccount vss-sa -n vss

# Secrets
oc get secrets -n vss | grep -E "ngc-|hf-|arango-|minio-|graph-"
```

To follow progress on specific pods:

```bash
oc logs -f deployment/vss-vss-deployment -n vss
oc logs -f statefulset/nim-llm -n vss
```

### 6. Access the UI

The Helm chart creates an OpenShift Route with TLS edge termination. Get the URL:

```bash
oc get route vss-ui -n vss -o jsonpath='{.spec.host}'
```

Open `https://<route-host>` in a browser.

### 7. Uninstall

```bash
helm uninstall vss -n vss
oc delete pvc --all -n vss
oc delete project vss
```

---

## Model Size Optimization

In GPU-constrained environments, the upstream chart's 70B LLM (4 GPUs) and 2-GPU VLM defaults leave multiple pods `Pending`. The `values-openshift.yaml` overrides these to `llama-3.1-8b-instruct` (1 GPU) and 1 GPU for the VLM respectively.

Configuration in `values-openshift.yaml`:

1. **LLM model and GPU count** — `nim-llm.model.name`, `nim-llm.image.repository`, `nim-llm.resources`, and `global.ucfGlobalEnv[0].value` (LLM_MODEL). Defaults to `meta/llama-3.1-8b-instruct` with 1 GPU.
2. **VLM GPU count** — `vss.resources.limits.nvidia.com/gpu` overrides the default 2-GPU request. The quantized `int4_awq` model fits on a single GPU.

Changing the LLM model also requires updating the model name in multiple locations - see [Challenge 10](#10-llm-model-name-consistency) for details.

---

## MLflow Observability

VSS is instrumented with MLflow to record per-request pipeline telemetry in the RHOAI MLflow UI. Every video summarization request produces one MLflow run containing metrics, artifacts, and a linked LLM trace.

### Architecture

```
VSS pod (vss-vss-deployment)
├── mlflow_helper.py          ← helper injected at startup by apply_mlflow_patches.py
├── via_stream_handler.py     ← patched in-place at pod startup to call the helper
│   ├── start_request_run()   ← opens MLflow run BEFORE _get_aggregated_summary()
│   └── end_request_run()     ← logs metrics/artifacts, links llama3-8b trace, ends run
└── mlflow.openai.autolog()   ← auto-traces llama3-8b summarization calls
                                 (cosmos VLM runs in spawned subprocesses — not traceable)

RHOAI MLflow server (redhat-ods-applications namespace)
└── Workspace: default  →  Experiment: vss-pipeline
    └── Per-request Run
        ├── Parameters: request_id, video_file, chunk_count, enable_chat, is_live
        ├── Metrics:    e2e_latency_s, ca_rag_latency_s,
        │               avg/max_vlm_chunk_latency_ms,
        │               vlm_total_input/output_tokens,
        │               nim_llm2_input/output_tokens
        ├── Artifacts:  chunk_captions.txt, final_summary.txt
        └── Evaluations tab: llama3-8b trace (full prompt + completion + token counts)
```

### How it works

1. **Startup patch** — `start.sh` (in `vss-scripts-cm`) copies the read-only `/opt/nvidia/via/via-engine` to a writable `/tmp/via/via-engine`, installs `mlflow>=3.0,<3.1`, then runs `apply_mlflow_patches.py`. The patcher injects four call sites into `via_stream_handler.py` and copies `mlflow_helper.py` alongside it — idempotent on every pod restart.

2. **Run lifecycle** — `start_request_run()` is called *before* `_get_aggregated_summary()` so the llama3-8b OpenAI call fires inside an active MLflow run. `mlflow.openai.autolog()` captures that call as a Trace. `end_request_run()` then links the trace to the run via the `POST /api/2.0/mlflow/traces/link-to-run` REST endpoint and extracts token counts from the trace's `mlflow.spanOutputs` span attribute.

3. **RHOAI workspace auth** — The RHOAI MLflow server requires an `X-MLFLOW-WORKSPACE` header on every request. `mlflow_helper.py` monkey-patches both `mlflow.utils.rest_utils.http_request` and `requests.Session.request` to inject this header automatically. The Kubernetes service account token (`/var/run/secrets/kubernetes.io/serviceaccount/token`) is used as the Bearer token.

4. **Server-side fixes** — The RHOAI MLflow server has two bugs in the validated version that require patches applied via a `sitecustomize.py` on the MLflow PVC:
   - **Missing trace REST endpoints** — `PATH_AUTHORIZATION_RULES` was missing all `/api/2.0/mlflow/traces/*` paths, blocking autolog trace uploads. Fixed by extending the authorization rules map.
   - **False-positive experiment conflict** — `_validate_artifact_isolation_constraints` raised `MlflowException` for workspace-owned experiments. Fixed by adding `.filter(workspace IS NULL)` to exclude them from the collision check. The patch must be applied to **both** `SqlAlchemyStore` and its subclass `WorkspaceAwareSqlAlchemyStore` since the subclass overrides the method.

### Files

| File | Purpose |
|------|---------|
| `src/vss-engine/src/mlflow_helper.py` | Core helper: init, HTTP patch, run lifecycle, trace linking, token extraction |
| `deploy/helm/scripts/apply_mlflow_patches.py` | Startup patcher: copies helper, patches `via_stream_handler.py` in-place (4 patches, idempotent) |
| `deploy/helm/mlflow.yaml` | RHOAI MLflow CR (`redhat-ods-applications`, 20 Gi PVC, `PYTHONPATH=/mlflow/python-patches`) |
| `deploy/helm/mlflow-standalone.yaml` | Optional standalone MLflow (port 5000, no auth) for development/testing |

### Deploying the MLflow server

Apply the RHOAI MLflow CR once per cluster. This requires cluster-admin or the RHOAI operator to be installed:

```bash
oc apply -f deploy/helm/mlflow.yaml
```

The operator creates the MLflow deployment in `redhat-ods-applications`. Apply the server-side bug fixes to the PVC (required for trace upload to work):

```bash
# Copy sitecustomize.py to the MLflow PVC via an ephemeral pod
# See the RHOAI MLflow server-side fix details above for the full patch content
oc exec -n redhat-ods-applications deployment/mlflow -- \
  bash -c "cat > /mlflow/python-patches/sitecustomize.py" < sitecustomize.py

# Restart the MLflow pod to pick up PYTHONPATH=/mlflow/python-patches
oc rollout restart deployment/mlflow -n redhat-ods-applications
```

Grant the `vss-sa` service account access to read MLflow experiments in the `default` workspace:

```bash
oc apply -f - <<'EOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlflow-experiments-role
  namespace: default
rules:
- apiGroups: ["mlflow.kubeflow.org"]
  resources: ["experiments"]
  verbs: ["get", "list", "create", "update", "delete", "patch"]
EOF
oc create rolebinding vss-sa-mlflow -n default \
  --role=mlflow-experiments-role --serviceaccount=vss:vss-sa
```

The `vss-mlflow-patches-cm` ConfigMap (containing `mlflow_helper.py` and `apply_mlflow_patches.py`) must exist in the `vss` namespace before installing the Helm chart:

```bash
oc create configmap vss-mlflow-patches-cm -n vss \
  --from-file=mlflow_helper.py=src/vss-engine/src/mlflow_helper.py \
  --from-file=apply_mlflow_patches.py=deploy/helm/scripts/apply_mlflow_patches.py
```

> **Cosmos VLM traces are not captured.** Cosmos runs in spawned subprocesses (`multiprocessing.get_context("spawn")`). Python's `mlflow.openai.autolog()` monkey-patch does not carry across `spawn` boundaries. Only the llama3-8b summarization call (which runs in the main process) is traced.

---

## OpenShift-Specific Challenges and Solutions

The upstream VSS Helm chart targets vanilla Kubernetes. Running it on OpenShift requires addressing incompatibilities across security contexts, storage permissions, secrets, GPU scheduling, and service configuration. All fixes are applied at install time via `values-openshift.yaml` and `templates/openshift.yaml` (inside the chart) - no post-deploy patching is required.

---

### 1. Storage Permissions

OpenShift assigns a random UID (e.g. `1000660000`) to containers rather than the UID defined in the image. Because this UID does not own the container's data directories, both services fail on startup with permission errors.

**Affected Services:**

- **milvus-minio** - Object storage for Milvus (`/minio_data`)
- **milvus** - Vector database persistence (`/var/lib/milvus`)

**Solution:** Mount an `emptyDir` volume over each problematic path. OpenShift automatically sets GID 0 with group-write permissions on `emptyDir` volumes, making them writable by any assigned UID.

```yaml
milvus-minio:
  extraPodVolumes:
  - name: data-volume
    emptyDir: {}
  extraPodVolumeMounts:
  - name: data-volume
    mountPath: /minio_data

milvus:
  extraPodVolumes:
  - name: data-volume
    emptyDir: {}
  extraPodVolumeMounts:
  - name: data-volume
    mountPath: /var/lib/milvus
```

> **Note:** `emptyDir` data is lost on pod restart. Replace with `PersistentVolumeClaims` for production.

---

### 2. Security Context Constraints

OpenShift's default `restricted-v2` SCC requires containers to run as a UID within the namespace-assigned range. Several sub-charts hardcode specific UIDs that fall outside this range, causing pods to fail admission with `unable to validate against any security context constraint: provider "anyuid": Forbidden`.

**Affected Services:**

- **arango-db** - Graph database (image-defined UID)
- **neo4j** - Graph database (`runAsUser: 7474`)
- **vss** - Core pipeline service (`runAsUser: 1000`)

**Solution:** The Helm chart creates a dedicated `vss-sa` service account and a RoleBinding granting the `anyuid` SCC exclusively to it (`templates/openshift.yaml`), scoping the elevated permission to a single named identity rather than the namespace-wide `default` service account.

```yaml
arango-db:
  serviceAccount:
    create: false
    name: vss-sa

neo4j:
  serviceAccount:
    create: false
    name: vss-sa

vss:
  serviceAccount:
    create: false
    name: vss-sa
```

---

### 3. Security Context Removal

The GPU containers (NIM and NeMo) are pre-configured with specific user/group IDs (`runAsUser: 1000`) that conflict with OpenShift's random UID allocation. Unlike the services in [Challenge 2](#2-security-context-constraints) that require their hardcoded UIDs, these containers work fine under any UID.

**GPU-Dependent Services:**

- **nim-llm** - `podSecurityContext.runAsUser: 1000`
- **nemo-embedding** - `securityContext.runAsUser: 1000`
- **nemo-rerank** - `securityContext.runAsUser: 1000`

**Solution:** Nullify the hardcoded security contexts in `values-openshift.yaml`, allowing OpenShift to assign its own UID via the `restricted-v2` SCC:

```yaml
nim-llm:
  podSecurityContext:
    runAsUser: null
    runAsGroup: null
    fsGroup: null

nemo-embedding:
  applicationSpecs:
    embedding-deployment:
      securityContext:
        runAsUser: null
        runAsGroup: null
        fsGroup: null

nemo-rerank:
  applicationSpecs:
    ranking-deployment:
      securityContext:
        runAsUser: null
        runAsGroup: null
```

---

### 4. GPU Scheduling

GPU nodes carry custom `NoSchedule` taints. Without matching tolerations, the scheduler cannot place GPU workloads on those nodes and the pods stay `Pending`.

**GPU-Dependent Services:**

- **nim-llm** - LLM inference
- **nemo-embedding** - Embedding model
- **nemo-rerank** - Reranking model
- **vss** - Core pipeline and VLM

**Solution:** Tolerations are defined in `values-openshift.yaml` for all four GPU services. The default key is `nvidia.com/gpu` — update to match your cluster's GPU node taint keys.

---

### 5. Missing Secrets

The chart references multiple secrets that must exist prior to installation but provides no mechanism to create them. Without them, pods fail with `secret not found` on volume mounts or image pulls.

**Required Secrets:**

- **ngc-docker-reg-secret** - Image pull secret for `nvcr.io`
- **ngc-api-key-secret** - Runtime NGC authentication for nemo-embedding and nemo-rerank. This is separate from the pull secret because image pull secrets (`kubernetes.io/dockerconfigjson`) cannot be referenced as `secretKeyRef` env vars
- **arango-db-creds-secret** - ArangoDB credentials
- **minio-creds-secret** - MinIO access credentials
- **graph-db-creds-secret** - Neo4j credentials, mounted as files by the parent chart.
- **hf-token-secret** - HuggingFace token for gated model downloads (see [Challenge 7](#7-hf_token-for-gated-model))

**Solution:** NGC and HuggingFace secrets are created via NVIDIA's built-in `nvcf` mechanism (`templates/nvcf_secrets.yaml`), configured in `values-openshift.yaml` and passed via `--set` at install time. Service credential secrets (ArangoDB, MinIO, Neo4j) are created by `templates/openshift.yaml` when `openshift.secrets.create` is `true`, or can be created manually before install.

---

### 6. Shared Memory Limit

Both sub-charts run NVIDIA Triton Inference Server with a Python BLS backend, which relies on POSIX shared memory (`/dev/shm`) for IPC between the server process and Python stub processes. OpenShift's default 64 MB `/dev/shm` limit is insufficient under concurrent inference load, resulting in `Failed to initialize Python stub: No space left on device` and pod crashes under load (exit code 137).

**Affected Services:**

- **nemo-embedding** - Vector embedding generation
- **nemo-rerank** - Document reranking

**Solution:** Mount a `Memory`-backed `emptyDir` at `/dev/shm`:

```yaml
nemo-embedding:
  extraPodVolumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 2Gi
  extraPodVolumeMounts:
  - name: dshm
    mountPath: /dev/shm

nemo-rerank:
  extraPodVolumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 2Gi
  extraPodVolumeMounts:
  - name: dshm
    mountPath: /dev/shm
```

---

### 7. HF_TOKEN for Gated Model

The vss container downloads `nvidia/Cosmos-Reason2-8B` from HuggingFace at startup. This model is gated - users must accept NVIDIA's license and authenticate with an HF token. Without `HF_TOKEN`, the download fails silently and the server never opens port 8000, so the pod stays `Running` but the readiness probe never passes.

**Solution:** The Helm chart creates `hf-token-secret` via NVIDIA's `nvcf.additionalSecrets` mechanism, configured in `values-openshift.yaml` and passed via `--set nvcf.additionalSecrets[1].stringData.value=$HF_TOKEN`. The chart already references `hf-token-secret` as an optional `secretKeyRef` — the secret being absent is what caused the silent failure.

---

### 8. Tokenizer Thread Pool Burst

Both services run Triton with a Python BLS backend. Triton spawns 16 stub processes simultaneously at startup, each invoking the HuggingFace fast tokenizer's `encode()` during initialization. The tokenizer is Rust-backed and uses the Rayon thread pool library, which initializes lazily and defaults to one thread per CPU. On a high-CPU node, this produces thousands of simultaneous `pthread_create()` calls. The Linux kernel returns `EAGAIN` to some of them, causing Rayon to panic rather than retry, and the pod enters a crash loop with 200+ restarts.

**Affected Services:**

- **nemo-embedding** - Embedding model serving
- **nemo-rerank** - Reranking model serving

**Solution:** Set `TOKENIZERS_PARALLELISM=false` on both containers to disable the tokenizer's internal parallelism:

```yaml
nemo-embedding:
  applicationSpecs:
    embedding-deployment:
      containers:
        embedding-container:
          env:
          - name: TOKENIZERS_PARALLELISM
            value: "false"

nemo-rerank:
  applicationSpecs:
    ranking-deployment:
      containers:
        ranking-container:
          env:
          - name: TOKENIZERS_PARALLELISM
            value: "false"
```

---

### 9. Guardrails False Positive on Image Input with 8B LLM

When using `llama-3.1-8b-instruct` as the guardrails LLM, image summarization requests are incorrectly blocked as unsafe.

**Solution:** `DISABLE_GUARDRAILS` is set to `"true"` by default in `values-openshift.yaml` for the 8B configuration. This does not affect core search, summarization, or alert functionality.

```yaml
global:
  ucfGlobalEnv:
  - name: LLM_MODEL
    value: meta/llama-3.1-8b-instruct
  - name: DISABLE_GUARDRAILS
    value: "true"
```

---

### 10. LLM Model Name Consistency

The LLM model name is hardcoded in three independent locations within the chart. Switching the LLM (e.g. from 70B to 8B) without updating all three causes the vss context manager to return 404 errors and guardrails to fall back to NVIDIA's cloud API with 401 Unauthorized.

**Affected Locations:**

- `nim-llm.model.name` - the model identity used by the NIM server itself
- `LLM_MODEL` env var in vss - used by the context manager for chat, summarization, and notifications. Set via `global.ucfGlobalEnv` in `values-openshift.yaml`
- `guardrails_config.yaml` `models[0]` - used by NeMo Guardrails for its startup validation test (only relevant when guardrails are enabled)

**Solution:** All three are configured in `values-openshift.yaml`: `nim-llm.model.name` and `nim-llm.image.repository` for the NIM server, `global.ucfGlobalEnv[0].value` for the LLM_MODEL env var. When guardrails are enabled, the guardrails config model defaults to `meta/llama-3.1-70b-instruct` from the chart — update it if using a different model.

---

## Deployment Files

All OpenShift customizations are codified in the `deploy/helm/` directory alongside the upstream chart:

| File | Description |
|------|-------------|
| `nvidia-blueprint-vss-2.4.1.tgz` | Packaged Helm chart with all OpenShift customizations baked in. This is what `helm upgrade --install` uses. |
| `nvidia-blueprint-vss/` | Unpacked chart source (edit here, then repack with `tar -czf nvidia-blueprint-vss-2.4.1.tgz nvidia-blueprint-vss`). |
| `values-openshift.yaml` | Helm values overlay — all OpenShift-specific overrides: security contexts, MIG GPU resources, tolerations, secrets, model endpoints, MLflow env vars, and OpenShift AI. |
| `nvidia-blueprint-vss/templates/openshift.yaml` | Helm template (gated by `openshift.enabled`). Creates ServiceAccount, RoleBinding (anyuid SCC), Route, and service credential secrets. |
| `nvidia-blueprint-vss/templates/openshift-ai.yaml` | Helm template (gated by `openshift.ai.enabled`). Creates PVCs, model download Jobs, `InferenceService`, and `ServingRuntime` resources for all four GPU models. |
| `is-sr.yaml` | Reference YAML: standalone `InferenceService` + `ServingRuntime` definitions for all four models. Useful for manually applying or inspecting individual model serving configs. |
| `job-pvc.yaml` | Reference YAML: standalone PVC + Job definitions for model weight downloads. Useful for pre-populating PVCs outside of Helm. |
| `mlflow.yaml` | RHOAI MLflow CR (`mlflow.opendatahub.io/v1`) applied in `redhat-ods-applications`. Creates the MLflow tracking server with 20 Gi PVC. |
| `mlflow-standalone.yaml` | Optional standalone MLflow server in the `vss` namespace (port 5000, no auth). For development and testing without RHOAI. |
| `scripts/apply_mlflow_patches.py` | Startup patcher run by `start.sh` on every pod restart. Injects MLflow instrumentation into `via_stream_handler.py`. |
| `scripts/override_remote_endpoints.sh` | Upstream utility script for generating a values override when using remote NIM endpoints instead of local KServe InferenceServices. |
