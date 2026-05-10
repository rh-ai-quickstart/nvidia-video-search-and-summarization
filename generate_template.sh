echo "{{- if .Values.openshift.ai.enabled }}" > deploy/helm/nvidia-blueprint-vss/templates/openshift-ai.yaml
cat deploy/helm/job-pvc.yaml >> deploy/helm/nvidia-blueprint-vss/templates/openshift-ai.yaml
echo "---" >> deploy/helm/nvidia-blueprint-vss/templates/openshift-ai.yaml
cat deploy/helm/is-sr.yaml >> deploy/helm/nvidia-blueprint-vss/templates/openshift-ai.yaml
echo "{{- end }}" >> deploy/helm/nvidia-blueprint-vss/templates/openshift-ai.yaml
