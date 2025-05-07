# Configure GPU time-slicing if you have fewer than three GPUs.

1. Create a file, ```time-slicing-config-all.yaml```, with the following content:

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config-all
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 3
```

- The sample configuration creates three replicas from each GPU on the node.Replicas can be increase and decrease.

- Add the config map to the Operator namespace:
```
kubectl create -n gpu-operator -f time-slicing-config-all.yaml
```

- Configure the device plugin with the config map and set the default time-slicing configuration:

```
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config-all", "default": "any"}}}}'
```

- Reset the Configure the device plugin if any new changes in the replica

```
kubectl patch clusterpolicy cluster-policy \
  -n gpu-operator \
  --type=json \
  -p='[{"op": "remove", "path": "/spec/devicePlugin/config"}]'
```

- Verify that at least 3 GPUs are allocatable:

```
kubectl get nodes -l nvidia.com/gpu.present -o json | jq '.items[0].status.allocatable | with_entries(select(.key | startswith("nvidia.com/"))) | with_entries(select(.value != "0"))'
```

