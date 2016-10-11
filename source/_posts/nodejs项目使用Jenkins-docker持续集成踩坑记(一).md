---
title: nodejs项目使用Jenkins+docker持续集成踩坑记<一>
date: 2016-10-11 13:53:04
tags: [运维, nodejs, Jenkins, CI]
---

之前手头某node项目在运行过程中遇到了一个坑爹的事情. 本身一直以来我对node项目上线的习惯都是直接`git checkout -b release-XXX && pm2 start process.json`, 尽管慢了一些, 但是还是在可忍受范围之内.

但是该项目是使用shell脚本重新打包目录结构重新release, 导致每次上线都要进行 **运行脚本**, **迁移配置**, **项目启动** 的过程. 这个低效率的过程实在是不能忍, 于是 **痛下决心** 调研 `CI` 工具链.

最初的设想是直接使用 [`travis-ci.org`](http://travis-ci.org) , 然而公司项目 & 私活项目 都不能完美解决, 还是自己部署一下 `Jenkins`, 踩坑之旅走起......

------

## 工具链条

* [Jenkins](https://jenkins.io): 基于java的持续集成工具
* [docker](https://www.docker.com): 容器服务, 使用容器部署`Jenkins` <del>因为我特么是个node程序猿, 不会部署java整套环境</del>
* [coding.net](https://coding.net): 代码托管, 有富余服务器的直接出门左转`gitlab`, 主要是使用`webHook` *用csdn的你加油...*
* [阿里ecs](https://www.aliyun.com/product/ecs): `centOS 7.0 64位` 1M小水管服务器...

------

## 环境搭建

* **docker安装**
    docker原本是由 `LXC(linux 内核虚拟化技术)` 而来, 所以早期的时候在不同平台上的实现方式不太一样, 在 `1.0` 版本之前, windows平台和
    mac平台上的docker都是使用`virtual box`虚拟机运行一个linux环境, 然后通过ssh进入linux环境实现的虚拟, 本质上类似于 `vagrant` + `docker
     for linux`.

    在 `1.0` 版本之后, docker 和 操作系统厂商合作, mac 上终于有了基于原生CPU虚拟技术实现的 [`HyperKit`](http://https://github.com/docker/HyperKit/), 其底层是基于[`Hypervisor.framework`](https://developer.apple.com/library/mac/documentation/DriversKernelHardware/Reference/Hypervisor/index.html). windows平台没有做过多的调研, 不太清楚, 不过应该也不是单纯的一个虚拟机那么简单了.技术的进步还是来自于各大厂商的合作啊.

    docker是基于linux内核虚拟化, 所以对内核版本和系统版本有要求, 所以需要确保你的系统为 `linux 64位` 系统并且 `内核版本` **大于** `3.10`. 使用 `uname -r` 查看内核版本, 确保你的系统符合需求.

    ```
    [~]$ uname -r
        3.10.0-327.36.1.el7.x86_64
    [~]$
    ```

    docker安装一般可以通过 `<pkg manage>` + `repo source` 或者 `install script`, 具体可以在[官网](https://docs.docker.com/engine/installation/)查看. 个人更推荐使用 `install script`, 一来安装起来方便, 二来, 使用`yum`等包管理工具安装, 以后升级的时候需要先进行卸载操作, 否则会出现问题. 使用 `install script` 则可以在任何情况下无缝进行升级和迁移.

    这里以`centOS`为例, 具体安装过程如下:

    ```
    [~]$ sudo yum update

    [~]$ curl -fsSL https://get.docker.com/ | sh
    ```

    因为我本地已经安装过, 所以会有如下提示, 这就是为什么我说用脚本安装的方式更好一些.

    ```
    Warning: the "docker" command appears to already exist on this system.

    If you already have Docker installed, this script can cause trouble, which is
    why we're displaying this warning and provide the opportunity to cancel the
    installation.

    If you installed the current Docker package using this script and are using it
    again to update Docker, you can safely ignore this message.

    You may press Ctrl+C now to abort this script.
    + sleep 20
    ```

    **注意**, 安装完成之后docker engine 会自动添加一个 `docker` 的用户组, 请务必根据相关提示添加当前用户到docker用户组中, 否则创建的容器将无法正常使用.

    ```
    [~]$ sudo usermod -aG docker your_username
    ```

    安装完成之后可以启动系统服务并后台常驻docker服务

    ```
    [~]$ sudo systemctl enable docker.service

    [~]$ sudo systemctl start docker
    ```

* **Jenkins 容器安装**
    下班回家更新......
