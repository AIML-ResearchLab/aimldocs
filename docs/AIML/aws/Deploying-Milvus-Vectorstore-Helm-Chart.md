## Deploying Milvus Vectorstore Helm Chart

1. Create a new nanespace for vectorstore

```
kubectl create namespace vectorstore
```

2. Add the milvus repository

```
helm repo add milvus https://zilliztech.github.io/milvus-helm/
```

3. Update the helm repository

```
helm repo update
```

4. Create a file named custom_value.yaml with below content to utilize GPU's

```
standalone:
  resources:
    requests:
      nvidia.com/gpu: "1"
    limits:
      nvidia.com/gpu: "1"
```

5. Install the helm chart and point to the above created file using -f argument as shown below.

```
helm install milvus milvus/milvus --set cluster.enabled=false --set etcd.replicaCount=1 --set minio.mode=standalone --set pulsar.enabled=false -f custom_value.yaml -n vectorstore


NAME: milvus
LAST DEPLOYED: Wed May  7 13:00:44 2025
NAMESPACE: vectorstore
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

6. Check status of the pods

```
kubectl get pods -n vectorstore
NAME                                 READY   STATUS      RESTARTS   AGE
milvus-etcd-0                        1/1     Running     0          117s
milvus-minio-cd798dd6f-zszjv         1/1     Running     0          117s
milvus-pulsarv3-bookie-0             1/1     Running     0          117s
milvus-pulsarv3-bookie-1             1/1     Running     0          117s
milvus-pulsarv3-bookie-2             1/1     Running     0          116s
milvus-pulsarv3-bookie-init-6q6ts    0/1     Completed   0          117s
milvus-pulsarv3-broker-0             1/1     Running     0          117s
milvus-pulsarv3-broker-1             1/1     Running     0          117s
milvus-pulsarv3-proxy-0              0/1     Running     0          117s
milvus-pulsarv3-proxy-1              0/1     Running     0          117s
milvus-pulsarv3-pulsar-init-66kcq    0/1     Completed   0          117s
milvus-pulsarv3-recovery-0           1/1     Running     0          117s
milvus-pulsarv3-zookeeper-0          1/1     Running     0          117s
milvus-pulsarv3-zookeeper-1          1/1     Running     0          116s
milvus-pulsarv3-zookeeper-2          1/1     Running     0          116s
milvus-standalone-7bf84684d4-bt9tv   0/1     Running     0          117s
```


## Configuring Examples

You can configure various parameters such as prompts and vectorstore using environment variables. Modify the environment variables in the env section of the query service in the values.yaml file of the respective examples.

**Configuring Prompts**

```
---
depth: 2
local: true
backlinks: none
---
```


Each example utilizes a ```prompt.yaml``` file that defines prompts for different contexts. These prompts guide the RAG model in generating appropriate responses. You can tailor these prompts to fit your specific needs and achieve desired responses from the models.

## Accessing Prompts

The prompts are loaded as a Python dictionary within the application. To access this dictionary, you can use the ```get_prompts()``` function provided by the ```utils``` module. This function retrieves the complete dictionary of prompts.


Consider we have following ```prompt.yaml``` file which is under ```files``` directory for all the helm charts


You can access it's chat_template using following code in you chain server


```
from RAG.src.chain_server.utils import get_prompts

prompts = get_prompts()

chat_template = prompts.get("chat_template", "")
```

- Once you have updated the prompt you can update the deployment for any of the examples by using the command below.

```
helm upgrade <rag-example-name> <rag-example-helm-chart-path> -n <rag-example-namespace> --set imagePullSecret.password=$NGC_CLI_API_KEY
```

## Configuring VectorStore

The vector store can be modified from environment variables. You can update:

1. ```APP_VECTORSTORE_NAME:``` This is the vector store name. Currently, we support milvus and pgvector Note: This only specifies the vector store name. The vector store container needs to be started separately.

2. ```APP_VECTORSTORE_URL:``` The host machine URL where the vector store is running.

## Additional Resources

Learn more about how to use NVIDIA NIM microservices for RAG through our Deep Learning Institute. Access the course here.

## Security considerations

The RAG applications are shared as reference architectures and are provided “as is”. The security of them in production environments is the responsibility of the end users deploying it. When deploying in a production environment, please have security experts review any potential risks and threats (including direct and indirect prompt injection); define the trust boundaries, secure the communication channels, integrate AuthN & AuthZ with appropriate access controls, keep the deployment including the containers up to date, ensure the containers are secure and free of vulnerabilities.






