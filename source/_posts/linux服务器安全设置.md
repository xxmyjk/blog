---
title: linux服务器安全设置
date: 2016-03-18 14:25:23
tags: [linux, 运维, 安全]
---

简明的linux服务器安全设置指南, 包括: 公钥登录, 禁止密码登录, 禁用 root 账户等.

公司的阿里云主机常年被 ssh 外加 http 各种扫, 除了一方面写出更加安全, 健壮的代码之外, 另一方面服务器的安全设置也不容忽视.
下面是我自己常用的服务器端相关配置. 阿里云主机, centOS 6.x.

## 账户设置

添加公共账户, 避免直接使用 root 账户.
阿里云的主机默认只提供了一个 root 账户, 我们需要添加一个工作账户, 并赋予 root 权限, 避免直接使用 root.

```sh
    useradd devops         //添加 devops 账户
    passwd devops          //修改 devops 账户密码
    useradd -G root devops //添加 devops 到 root 用户组
```

这样我们就拥有了一个 root 权限的账户, 接下来就是禁止 root 账户的 shell 登录和使用.

<!-- more -->
由于我们以后不会再使用密码登录, 并且要禁止 root 的 shell 登录. 所以, 在禁用之前, 需要先配置好公钥文件, 防止无法正常登录服务器.

1. 生成密钥(ssh-keygen)
2. 复制公钥到服务器(ssh-copy-id)
3. 修改 ssh server 配置文件, 允许公钥认证, sudo vi /etc/ssh/sshd_config
```
    RSAAuthentication yes       //开启RSA 及公钥认证
    PubkeyAuthentication yes
```

4. 修改服务器端文件夹的拥有者及权限, 权限设置是必须的, 否则不能正常识别公钥
```sh
    chown -R devops:devops .ssh         //修改.ssh 文件夹的拥有者
    chmod 700 .ssh                      //修改文件夹权限为700,必须
    chmod 600 .ssh/authorized_keys      //修改文件权限为600,必须
    sudo services sshd restart          //重启 ssh 服务
```

接下来进行 ssh 登录测试, 如果正常登录且未提示输入密码, 证明我们的公钥配置已经生效, 这个时候就可以大胆的关闭 root 账户登录和账户登录的密码验证了.
sudo vi /etc/ssh/sshd_config

```sh
    PermitRootLogin no          //禁止 root 用户登录
    PasswordAuthentication no   //禁用密码验证
```

## 端口设置及 iptables

除了做到以上的还不能确保足够的安全, 我们需要对服务器的端口进行限定开放.

我司服务器目前对外开放端口只有80, 443, 22三个端口, 即除了 ssh, http, https 之外, 不对外部开放任何端口.有需要可以修改 ssh 默认端口号, sudo vi /etc/ssh/sshd_config
```
    port xx
```

这个配置可以在阿里云的控制面板内进行设置, 当然本地进行设置也是一样的道理.

由于阿里云的机房内不是一台机器, 也就是说尽管我们的主机没有暴露在外网环境下, 但是阿里云的内网内部还是可以扫描到我们的服务器的.所以,
我们还需要使用 iptables 对内网 ip 进行限制.

我司在阿里有两台服务器, 分布在同一个内网环境, 所以除了两者之间互访之外, 屏蔽所有其他的内网互访. 这个在阿里云主机的控制面板也是可以设置的, iptables 同理.

## 其他

防火墙保持常开, 定时备份, 磁盘加密, 及时查 ssh 和 http 的相关 log, 发现可疑情况及时处理.
另, 对于 http 层可以在 nginx 接入层设置 ip 黑名单进行屏蔽.
