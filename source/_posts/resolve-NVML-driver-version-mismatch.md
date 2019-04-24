---
title: '解决 "Failed to initialize NVML: Driver/library version mismatch"'
date: 2019-04-24 15:25:47
tags: [linux, 运维, nvidia]
---

服务器更新 `nvidia driver` 后遇到以下问题:

`Failed to initialize NVML: Driver/library version mismatch`

## 一句话解决方案:

```bash
    # su 权限
    lsmod | grep -i ^nvidia | awk '{print $1}' | rmmod && nvidia-smi
```

或者

```bash
    # 雾
    sudo reboot
```

## 原因分析:

驱动更新后 linux 内核对应驱动的 kernel module 并没有重置, 外部相关进程引用了旧版本驱动相关的 mod, 需要手动卸载, 重新执行 `nvidia-smi`
会自动加载新版本 mod 到内核

## 注意

卸载过程可能会因为相关进程引用或者内核 mod 引用顺序导致卸载失败, 这时需要按照提示顺序卸载.

<!-- more -->

比如:

```bash
    rmmod nvidia
    > rmmod: ERROR: Module nvidia is in use by: nvidia_modeset nvidia_uvm
```

这时就需要先卸载`nvidia_modeset` 和 `nvidia_uvm`

一些相关的 `kernel mod` 命令

* 查看进程引用 `mod`

```bash
    lsof -n -w /dev/nvidia
```

* kernel mod 卸载

```bash
    rmmod <module_name> | modprobe -r <module_name>
```

* kernel mod 加载

```bash
    modprobe
```
 

## 参考资料

- [stackoverflow](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch)
- [Comzyh的博客](https://comzyh.com/blog/archives/967/)
- [archlinux](https://wiki.archlinux.org/index.php/Kernel_module)
