1. create Cluster
```bash
eksctl create cluster \
  --name reasonforge \
  --region us-east-1 \
  --version 1.34 \
  --without-nodegroup
```

2. create cpu nodegroup
```bash
eksctl create nodegroup \
  --cluster reasonforge \
  --region us-east-1 \
  --name cpu-nodes \
  --node-type t4g.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 5 \
  --node-volume-size 50 \
  --node-labels "role=cpu,workload-type=general" \
  --managed
```

3. create gpu nodegroup
```bash
eksctl create nodegroup \
  --cluster reasonforge \
  --region us-east-1 \
  --name gpu-nodes \
  --node-type g5.xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --node-volume-size 100 \
  --node-labels "role=gpu,workload-type=ml" \
  --node-ami-family AmazonLinux2023 \
  --managed
```

4. manage with eks by adding to config
```bash
aws eks --region us-east-1 update-kubeconfig --name reasonforge
```

5. Create S3 for data and model storage
```bash

```

6.  