---
title: 支持 GPU 调度的 kubernetes 部署方案(CentOS)
date: 2019-04-16 15:12:47
tags: [kubernetes, linux, 运维]
---
## docker installation

- optional: clean old version if needed
```
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine \
                  docker-ce \
                  docker-ce-cli \
                  containerd.io
```

- install yum utils
```
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
```

- add docker-ce repo
```
sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
```

<!-- more -->

- install docker-ce
```
sudo yum install docker-ce docker-ce-cli containerd.io
```

- optional: setup docker `data-root`

`dockerd` store `images/caches/volumes ...` data in `/var/lib/docker` by default, and the `kuberntes` will GC docker
image NOT CURRENT IN USING, change the `data-root` to a large disk portion.

```
sudo vi /usr/lib/systemd/system/docker.service

> append --data-root <a large disk portion> behind dockerd Exec
```

## nvidia-docker | nvidia-container-runtime installation

- add nvidia-docker repo
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
```

- install nvidia-docker
```
sudo yum install nvidia-docker2
sudo pkill -SIGHUP dockerd
```

- modify `/etc/docker/daemon.json` to enable `nvidia` as default docker runtime

- optional: setup your own `shadowsocks server & client & privoxy`

- modify `/usr/lib/systemd/system/docker.server` to enable docker image pull access to `gcr.io`

```
Environment="HTTP_PROXY=x.x.x.x:xx;HTTPS_PROXY=x.x.x.x:xx;NO_PROXY=x.x.x.x:xx"
```

## kubernetes stack installation (local kubelet)

- optional: remove outdated kubeadm, kubelet, kubectl

```
sudo yum remove -y kubeadm kubelet kubectl
```

- `kubelet`, `kubectl`, `kubeadm` install
    >follow [here](https://kubernetes.io/docs/setup/independent/install-kubeadm/)

- using `kubeadm` to install `HA` cluster
    >follow [here](https://kubernetes.io/docs/setup/independent/setup-ha-etcd-with-kubeadm/)

## kubernetes stack installation (rke -> stack in docker)

- install [rke](https://github.com/rancher/rke)

- rke up

```yaml

nodes:
    - address: 192.168.1.14
      user: jinyi
      role:
        - controlplane
        - etcd
        - worker
    - address: 192.168.1.15
      user: jinyi
      role:
        - controlplane
        - etcd
        - worker
    - address: 192.168.1.16
      user: jinyi
      role:
        - controlplane
        - etcd
        - worker

```

- more config [here](https://rancher.com/docs/rke/latest/en/)

## apply services & conf to cluster

### kubernetes-dashboard

- apply stable `kubernetes-dashboard`

```bash
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v1.10.1/src/deploy/recommended/kubernetes-dashboard.yaml
```

- create `admin role binding` (local only for security)

```bash
echo `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kube-system
` | kubectl apply -f -
```

- get `dashboard login token` & login to dashboard

```bash
    kubectl -n kube-system describe secrets admin-user | grep token:
    
    # copy the output token to clipboard
    
    # start local proxy
    kubectl proxy
    
    # open in bro
```

- open in browser [kubernetes-dashboard](http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/)

- enter `token` you copy before & login
