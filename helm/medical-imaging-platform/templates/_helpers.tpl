{{- define "medical-imaging-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "medical-imaging-platform.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "medical-imaging-platform.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "medical-imaging-platform.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "medical-imaging-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
medical-imaging-platform.openai.com/scope: research-engineering-demonstrator
{{- end -}}

{{- define "medical-imaging-platform.serviceAccountName" -}}
{{- if .Values.global.serviceAccount.create -}}
{{- default (include "medical-imaging-platform.fullname" .) .Values.global.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.global.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "medical-imaging-platform.image" -}}
{{- $globalTag := .root.Values.global.imageTag -}}
{{- $tag := default $globalTag .image.tag -}}
{{- $digest := default .root.Values.global.imageDigest .image.digest -}}
{{- if $digest -}}
{{- printf "%s@%s" .image.repository $digest -}}
{{- else -}}
{{- printf "%s:%s" .image.repository $tag -}}
{{- end -}}
{{- end -}}
