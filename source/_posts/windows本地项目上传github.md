---
title: Windows本地项目同步到Github
tags: [Github]
categories: Git
---
本文记录了如何在windows下将本地项目同步到Github<!-- more -->
### 创建[Github](https://github.com/join?source=header-home)账户
### 安装[cmder](https://scnico.github.io/2017/12/24/Git%E7%AC%94%E8%AE%B0/#cmder)

### 创建本地版本库
1. 进入项目根目录
2. 右键**cmder here**
3. 在cmder中输入`git init`来初始化本地仓库
4. 在cmder中输入`git add .`把该目录下所有文件添加到仓库
5. 在cmder中输入`git commit -m "init commit"`将项目提交到仓库


### 创建秘钥并关联远程仓库
1. 在cmder中输入 `git config --global user.name "username"` 和 `git config --global user.email your@email.com` 配置
2. 在cmder中输入`ssh-keygen -t rsa -C "youremail@example.com"`生成秘钥
3. 在**C:\Users\yourusername\\.ssh**在目录下找到**id_rsa.pub**文件，可以用记事本或者其他文本工具打开该文件并全选复制里面的公钥
4. 登录[Github](https://github.com)找到**Settings**，找到其中**SSH and GPG keys**功能，**New SSH key**，其中的**Title**可以随便命名，然后将复制的公钥粘贴到**Key**中，最后点击**Add SSH key**
5. 在Github上**Create a new repository**，Repository name与项目名相同，不选择创建README，点击**Create repository**
6. 继续在cmder中输入以下命令关联远程仓库
   `git remote add origin https://github.com/username/repositoryname.git`

### 将本地仓库内容推送到远程仓库
在cmder中输入`git push -u origin master`，由于新建的远程仓库是空的，所以要加上-u这个参数。在以后的使用中你可以只需要做以下的操作即可将本地内容同步到Github中

```cmd
git add .
git commit -m "commit description"
git push
```
