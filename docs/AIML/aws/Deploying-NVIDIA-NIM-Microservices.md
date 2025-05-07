# Deploying NVIDIA NIM Microservices
## Deploying NVIDIA NIM for LLMs

(Default flow deploys meta/llama3-8b-instruct)

- Follow the steps from nim-deploy repository to deploy NIM LLM microservice with meta/llama3-8b-instruct as default LLM model.

```
https://github.com/NVIDIA/nim-deploy/tree/main/helm
```

## Using the NVIDIA NIM for LLMs helm chart

The NIM Helm chart requires a Kubernetes cluster with appropriate GPU nodes and the GPU Operator installed.

## Setting up the environment

Set the NGC_API_KEY environment variable to your NGC API key, as shown in the following example.

export NGC_API_KEY="key from ngc"

- Clone this repository and change directories into nim-deploy/helm. The following commands must be run from that directory.

```
git clone git@github.com:NVIDIA/nim-deploy.git
cd nim-deploy/helm
```

- Select a NIM to use in your helm release

Each NIM contains an AI model, application, or workflow. All files necessary to run the NIM are encapsulated in the container that is available on NGC. The NVIDIA API Catalog provides a sandbox to experiment with NIM APIs prior to container and model download.

## Setting up your helm values

All available helm values can be discoved by running the helm command after downloading the repo.

```
helm show values nim-llm/
```

## Create a namespace

```
kubectl create namespace nim
```

## Launching a NIM with a minimal configuration

You can launch llama3-8b-instruct using a default configuration while only setting the NGC API key and persistence in one line with no extra files. Set persistence.enabled to true to ensure that permissions are set correctly and the container runtime filesystem isn't filled by downloading models.

```
helm --namespace nim install my-nim nim-llm/ --set model.ngcAPIKey=$NGC_API_KEY --set persistence.enabled=true
NAME: my-nim
LAST DEPLOYED: Wed May  7 12:29:46 2025
NAMESPACE: nim
STATUS: deployed
REVISION: 1
NOTES:
Thank you for installing nim-llm.

**************************************************
| It may take some time for pods to become ready |
| while model files download                     |
**************************************************

Your NIM version is: 1.0.3
```

## Running inference

If you are operating on a fresh persistent volume or similar, you may have to wait a little while for the model to download. You can check the status of your deployment by running

```
kubectl get pods -n nim
NAME       READY   STATUS    RESTARTS   AGE
my-nim-0   0/1     Running   0          4m26s
```

And check that the pods have become "Ready".

Once that is true, you can try something like:

[Deploying NVIDIA NIM Microservices](https://github.com/NVIDIA/nim-deploy/tree/main/helm)

