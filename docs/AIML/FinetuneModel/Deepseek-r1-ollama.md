# Deepseek-R1 ollama

## Model Dockerhub link
[Model Dockerhub link](https://hub.docker.com/r/mdelapenya/deepseek-r1)

# Deploy Deepseek-R1 ollama in local AWS EKS with ALB Ingress

## Deployment

```
deepseek-r1-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek-r1
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek-r1
  template:
    metadata:
      labels:
        app: deepseek-r1
    spec:
      containers:
      - name: deepseek-r1
        image: mdelapenya/deepseek-r1:0.5.4-7b
        ports:
        - containerPort: 11434
```

## Service

```
deepseek-r1-service.yaml

apiVersion: v1
kind: Service
metadata:
  name: deepseek-r1-service
spec:
  selector:
    app: deepseek-r1
  ports:
  - protocol: TCP
    port: 11434
    targetPort: 11434
  type: ClusterIP
```

## Ingress

```
deepseek-ingress.yaml

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: deepseek-r1-service
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:ap-south-1:777203855866:certificate/b7856d98-d602-4a77-afdf-98d0b00706ff
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/healthcheck-path: /
    alb.ingress.kubernetes.io/load-balancer-attributes: idle_timeout.timeout_seconds=900
spec:
  ingressClassName: alb
  tls:
   - hosts:
       - deepseek-r1.visionaryai.aimledu.com
  rules:
    - host: deepseek-r1.visionaryai.aimledu.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: deepseek-r1-service
                port:
                  number: 11434
```

## Deployment

## Create namespace deepseek

```
kubectl create namespace deepseek
```

## Deploy deepseek, service, ingress

```
kubectl apply -f deepseek-r1-deployment.yaml -n deepseek

kubectl apply -f deepseek-r1-service.yaml -n deepseek

kubectl apply -f deepseek-ingress.yaml -n deepseek
```

```
kubectl -n deepseek logs -f pod/deepseek-r1-6b58ff58bc-7nznv
2025/05/21 12:43:58 routes.go:1259: INFO server config env="map[CUDA_VISIBLE_DEVICES: GPU_DEVICE_ORDINAL: HIP_VISIBLE_DEVICES: HSA_OVERRIDE_GFX_VERSION: HTTPS_PROXY: HTTP_PROXY: NO_PROXY: OLLAMA_DEBUG:false OLLAMA_FLASH_ATTENTION:false OLLAMA_GPU_OVERHEAD:0 OLLAMA_HOST:http://0.0.0.0:11434 OLLAMA_INTEL_GPU:false OLLAMA_KEEP_ALIVE:5m0s OLLAMA_KV_CACHE_TYPE: OLLAMA_LLM_LIBRARY: OLLAMA_LOAD_TIMEOUT:5m0s OLLAMA_MAX_LOADED_MODELS:0 OLLAMA_MAX_QUEUE:512 OLLAMA_MODELS:/root/.ollama/models OLLAMA_MULTIUSER_CACHE:false OLLAMA_NOHISTORY:false OLLAMA_NOPRUNE:false OLLAMA_NUM_PARALLEL:0 OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:* app://* file://* tauri://* vscode-webview://*] OLLAMA_SCHED_SPREAD:false ROCR_VISIBLE_DEVICES: http_proxy: https_proxy: no_proxy:]"
time=2025-05-21T12:43:58.101Z level=INFO source=images.go:757 msg="total blobs: 5"
time=2025-05-21T12:43:58.101Z level=INFO source=images.go:764 msg="total unused blobs removed: 0"
[GIN-debug] [WARNING] Creating an Engine instance with the Logger and Recovery middleware already attached.

[GIN-debug] [WARNING] Running in "debug" mode. Switch to "release" mode in production.
 - using env:	export GIN_MODE=release
 - using code:	gin.SetMode(gin.ReleaseMode)

[GIN-debug] POST   /api/pull                 --> github.com/ollama/ollama/server.(*Server).PullHandler-fm (5 handlers)
[GIN-debug] POST   /api/generate             --> github.com/ollama/ollama/server.(*Server).GenerateHandler-fm (5 handlers)
[GIN-debug] POST   /api/chat                 --> github.com/ollama/ollama/server.(*Server).ChatHandler-fm (5 handlers)
[GIN-debug] POST   /api/embed                --> github.com/ollama/ollama/server.(*Server).EmbedHandler-fm (5 handlers)
[GIN-debug] POST   /api/embeddings           --> github.com/ollama/ollama/server.(*Server).EmbeddingsHandler-fm (5 handlers)
[GIN-debug] POST   /api/create               --> github.com/ollama/ollama/server.(*Server).CreateHandler-fm (5 handlers)
[GIN-debug] POST   /api/push                 --> github.com/ollama/ollama/server.(*Server).PushHandler-fm (5 handlers)
[GIN-debug] POST   /api/copy                 --> github.com/ollama/ollama/server.(*Server).CopyHandler-fm (5 handlers)
[GIN-debug] DELETE /api/delete               --> github.com/ollama/ollama/server.(*Server).DeleteHandler-fm (5 handlers)
[GIN-debug] POST   /api/show                 --> github.com/ollama/ollama/server.(*Server).ShowHandler-fm (5 handlers)
[GIN-debug] POST   /api/blobs/:digest        --> github.com/ollama/ollama/server.(*Server).CreateBlobHandler-fm (5 handlers)
[GIN-debug] HEAD   /api/blobs/:digest        --> github.com/ollama/ollama/server.(*Server).HeadBlobHandler-fm (5 handlers)
[GIN-debug] GET    /api/ps                   --> github.com/ollama/ollama/server.(*Server).PsHandler-fm (5 handlers)
[GIN-debug] POST   /v1/chat/completions      --> github.com/ollama/ollama/server.(*Server).ChatHandler-fm (6 handlers)
[GIN-debug] POST   /v1/completions           --> github.com/ollama/ollama/server.(*Server).GenerateHandler-fm (6 handlers)
[GIN-debug] POST   /v1/embeddings            --> github.com/ollama/ollama/server.(*Server).EmbedHandler-fm (6 handlers)
[GIN-debug] GET    /v1/models                --> github.com/ollama/ollama/server.(*Server).ListHandler-fm (6 handlers)
[GIN-debug] GET    /v1/models/:model         --> github.com/ollama/ollama/server.(*Server).ShowHandler-fm (6 handlers)
[GIN-debug] GET    /                         --> github.com/ollama/ollama/server.(*Server).GenerateRoutes.func1 (5 handlers)
[GIN-debug] GET    /api/tags                 --> github.com/ollama/ollama/server.(*Server).ListHandler-fm (5 handlers)
[GIN-debug] GET    /api/version              --> github.com/ollama/ollama/server.(*Server).GenerateRoutes.func2 (5 handlers)
[GIN-debug] HEAD   /                         --> github.com/ollama/ollama/server.(*Server).GenerateRoutes.func1 (5 handlers)
[GIN-debug] HEAD   /api/tags                 --> github.com/ollama/ollama/server.(*Server).ListHandler-fm (5 handlers)
[GIN-debug] HEAD   /api/version              --> github.com/ollama/ollama/server.(*Server).GenerateRoutes.func2 (5 handlers)
time=2025-05-21T12:43:58.102Z level=INFO source=routes.go:1310 msg="Listening on [::]:11434 (version 0.5.4-0-g2ddc32d-dirty)"
time=2025-05-21T12:43:58.102Z level=INFO source=routes.go:1339 msg="Dynamic LLM libraries" runners="[cuda_v12_avx cpu cpu_avx cpu_avx2 cuda_v11_avx]"
time=2025-05-21T12:43:58.102Z level=INFO source=gpu.go:226 msg="looking for compatible GPUs"
time=2025-05-21T12:43:58.501Z level=INFO source=types.go:131 msg="inference compute" id=GPU-ec991403-1199-e627-c275-11191969eefd library=cuda variant=v12 compute=8.6 driver=12.8 name="NVIDIA A10G" total="22.3 GiB" available="8.0 GiB"
```

## Post successfull deployment how to test & use 

```
curl -X POST https://deepseek-r1.visionaryai.aimledu.com/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "deepseek-coder:6.7b"}'
```

```
curl -X POST https://deepseek-r1.visionaryai.aimledu.com/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder:6.7b",
    "prompt": "Explain what a Kubernetes ingress controller does.",
    "stream": false
  }'
```

```
curl -X POST https://deepseek-r1.visionaryai.aimledu.com/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder:6.7b",
    "prompt": "Explain what a Kubernetes ingress controller does.",
    "stream": false
  }'
{"model":"deepseek-coder:6.7b","created_at":"2025-05-21T13:04:43.677107124Z","response":"A Kubernetes Ingress Controller is a dedicated component that acts as an API frontend for services in your cluster, directing HTTP(S) traffic to the appropriate service based on rules defined by you or operators. \n\nIn other words, it's like a load balancer but specifically designed to work with Kubernetes and integrate with the ingress resources. Ingress Controllers are responsible for routing external traffic into services within your cluster, which means directing HTTP(S) requests based on the request host or path to specific services.\n\nThere are several types of Ingress controllers available:\n\n1. NGINX: A popular choice, it's known for its high performance and stability, especially with heavy traffic loads.\n2. Traefik: An open-source project that offers a dynamic reverse proxy solution. \n3. HAProxy: Also known for its speed and robustness in handling requests, especially when dealing with SSL offloading.\n4. Amazon ALB Ingress Controller: For AWS users, it integrates with the Application Load Balancer (ALB) to manage external access to services within an EKS cluster.\n5. Contour: A lightweight Ingress controller using Envoy as its data plane.\n6. GCE or GKE Ingress: These are specifically for Google Cloud Platform's products, and integrate with their load balancers.\n\nThe main responsibility of the ingress controller is to provide a layer 7 routing mechanism that can distribute traffic between different services within your Kubernetes cluster based on HTTP routes. This means you define rules about how external users should access your services, without those details being exposed to them directly.\n","done":true,"done_reason":"stop","context":[2042,417,274,20926,14244,20391,11,26696,254,20676,30742,339,8589,2008,11,6908,457,20676,30742,7958,11,285,340,885,3495,4301,4512,276,4531,8214,13,1487,4636,2223,13143,4301,11,5411,285,13936,4447,11,285,746,2159,12,13517,250,8214,4301,11,340,540,20857,276,3495,13,185,13518,3649,3475,25,185,1488,20667,852,245,716,31055,9350,6208,698,8888,1214,13,185,13518,21289,25,185,32,716,31055,9350,680,3524,18173,317,245,10653,5785,344,11773,372,274,8690,3853,408,327,3235,279,518,9654,11,1706,272,18125,7,50,8,9186,276,254,6854,2408,2842,331,6544,4212,457,340,409,10715,13,207,185,185,769,746,3061,11,359,6,82,833,245,3299,4862,12774,545,10184,5392,276,826,365,716,31055,9350,285,24729,365,254,6208,698,6177,13,680,3524,3458,20029,417,8874,327,27462,6659,9186,878,3235,2372,518,9654,11,585,2445,1706,272,18125,7,50,8,12443,2842,331,254,3092,3686,409,3076,276,3041,3235,13,185,185,2948,417,2961,4997,280,680,3524,630,20029,2315,25,185,185,16,13,461,16161,55,25,338,4493,4850,11,359,6,82,3174,327,891,1453,3779,285,13699,11,4386,365,6751,9186,18127,13,185,17,13,6726,811,1913,25,1633,1714,12,1905,2299,344,5157,245,10999,13322,15072,3402,13,207,185,18,13,414,32,18131,25,6067,3174,327,891,4575,285,13130,1457,279,14326,12443,11,4386,750,14029,365,25811,838,20711,13,185,19,13,11183,8855,33,680,3524,18173,25,1487,29182,4728,11,359,3834,980,365,254,15838,15748,9817,12774,207,7,1743,33,8,276,8800,6659,2451,276,3235,2372,274,426,17607,9654,13,185,20,13,3458,415,25,338,27395,680,3524,8888,1242,2344,85,1143,372,891,1189,9633,13,185,21,13,452,4402,409,452,7577,680,3524,25,3394,417,10184,327,5594,15948,27782,6,82,3888,11,285,24729,365,699,3299,4862,29664,13,185,185,546,1959,12374,280,254,6208,698,8888,317,276,2764,245,6271,207,22,27462,12379,344,482,27898,9186,1433,1442,3235,2372,518,716,31055,9350,9654,2842,331,18125,22168,13,997,2445,340,5928,6544,782,940,6659,4728,1020,2451,518,3235,11,1666,1454,4283,1430,14660,276,763,4712,13,185],"total_duration":3740760469,"load_duration":8736319,"prompt_eval_count":81,"prompt_eval_duration":4000000,"eval_count":353,"eval_duration":3726000000}%
```



