# Deploying NVIDIA Nemo Retriever Embedding Microservice

## NVIDIA NIM for NV-EmbedQA-E5-V5

**Setup Environment**

First create your namespace and your secrets

```
NAMESPACE=nvidia-nims

DOCKER_CONFIG='{"auths":{"nvcr.io":{"username":"$oauthtoken", "password":"'${NGC_API_KEY}'" }}}'

echo -n $DOCKER_CONFIG | base64 -w0

NGC_REGISTRY_PASSWORD=$(echo -n $DOCKER_CONFIG | base64 -w0 )

kubectl create namespace ${NAMESPACE}

kubectl apply -n ${NAMESPACE} -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: nvcrimagepullsecret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: ${NGC_REGISTRY_PASSWORD}
EOF
kubectl create -n ${NAMESPACE} secret generic ngc-api --from-literal=NGC_API_KEY=${NGC_API_KEY}


secret/nvcrimagepullsecret created
secret/ngc-api created
```

## Install the chart

```
helm upgrade \
    --install \
    --username '$oauthtoken' \
    --password "${NGC_API_KEY}" \
    -n ${NAMESPACE} \
    --set persistence.class="local-nfs" \
    text-embedding-nim \
    https://helm.ngc.nvidia.com/nim/nvidia/charts/text-embedding-nim-1.2.0.tgz
```




https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/helm-charts/text-embedding-nim