---
title: Git笔记
tags: git
categories: tools
---


### Git简介
Git是目前世界上最好的分布式版本控制系统，通过Git可以将你的文件目录下所有的文件管理起来，每个文件的修改、删除，Git都能跟踪到，以便在任何时刻都可以追踪历史或者还原。<!-- more -->注：以下操作均在windows环境下。由于本文章是在看了廖雪峰老师的教程以后的笔记，大家也可以学习廖雪峰老师的[详细课程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)。

### cmder
[cmder](http://cmder.net/)是windows的命令行工具，它不仅可以使用windows的所有命令，更能使用linux和shell的相关命令，并且`Full`版本内建`Git for Windows`，最重要的是颜值巨高。所以以下操作均在`cmder`中操作。

#### 安装cmder
在[cmder](http://cmder.net/)中点击下载`Full`版本，直接解压便可以使用啦
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/9a6LCi1mGI.png?imageslim" width="70%" height="70%" />

#### 让cmder便于使用
1. 将`cmder`添加到右键菜单中：在环境变量中添加`cmder.exe`的路径，然后在命令行中执行`Cmder.exe /REGISTER ALL`就可以在右键中出现`Cmder Here`了

2. 将λ符号修改成`$`符号：`cmder`默认的命令提示符是`λ`，我们可以将其改成常见的`$`，在`cmder/vendor/clink.lua`文件中修改其中第46和第48行的将`λ`为`$`



<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/Ie74gDlLKG.png?imageslim" width="80%" height="80%" />

然后我们就发现变成下面这样了
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/iHK7A8eleK.png?imageslim" width="70%" height="70%" />

### 仓库
版本库，又名仓库，在这个仓库中可以通过Git管理该仓库的历史版本。通过在windows任意目录在命令行中输入`git init`就可以把该目录变成Git管理的仓库。假设我们现在在目录`GitLearn`目录下使用了该命令，那么在该目录下就会出现`.git`文件夹。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/lBCI58Cj1j.png?imageslim" width="80%" height="80%" />

#### 添加文件到版本库
版本控制系统，只能跟踪文本文件的改动，例如`txt`文件、代码。所有要使用版本控制就需要以纯文本方式编写文件。
1. 在`GitLearn`目录下新建文本文档`README.txt`：注意使用notepad以utf-8打开，里面的内容如下：

```txt
Hello Git!
```
2. 将文件添加到仓库：

```git
git add README.txt
```
3. 将文件改动提交到仓库：`-m`为本次提交的说明，最好添加有意义的说明，这样就能方便从历史中找到改动记录。

```git
git commit -m "add readme file"
```
<center><img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/Llj1GKj4kb.png?imageslim" width="80%" height="80%" /></center>

注：`git add`可以添加多个文件，而`git commit`一次可以提交多条改动，例如你可以通过以下语句添加`README1.txt`和`README2.txt`:
```git
git add README1.txt README2.txt
```


#### 修改文件
1. 修改`README.txt`如下：

```txt
Hello Git!
first modify readme.txt
```

2. 查看当前仓库的状态：
   `git status`可以查看当前仓库的状态，可以从图中看到README.txt被修改了，然而还没有准备提交修改
```git
git status
```
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/333F7D50Cf.png?imageslim" width="80%" height="80%" />

3. 查看修改过的地方：
   `git diff`可以查看当前仓库具体修改了![img](http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/333F7D50Cf.png?imageslim)
4. 什么内容，可以从图中看到README.txt后面添加了新行
```git
git diff
```
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/303eCa1CCG.png?imageslim" width="80%" height="80%" />

4. 用之前的方法提交改动到仓库

```git
git add README.txt
git commit -m "modify readme.txt"
```

#### 版本回退
1. 查看版本日志
   `git log`可以查看仓库的版本日志，可以从图中看到一共修改了两次分别是`add readme file`和`modify readme.txt`，其中`commit`后的数字是版本号，这是用`SHA1`计算出来的，这是每个版本的唯一标识
```git
git log
```
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/kD5C9HK9hh.png?imageslim" width="80%" height="80%" />

2. 回退到某个版本：
   在Git中`HEAD`表示当前版本，`HEAD^`表示上个版本，`HEAD^^`表示上上个版本，要是回退`N`个版本可以使用`HEAD~N`，要将当前版本回退回上一个版本可以使用`git reset`命令，而在`cmd`中`^`是一个特殊字符，需要用`""`包括，当然也可以使用`HEAD~1`，此外还可以通过版本号来选择回退回的版本，当然只需要输入版本号的能够让Git区分的版本号前几位就可以了

```git
git reset --hard HEAD"^"
git reset --hard HEAD~1
git reset --hard bdf8646
```
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/j8gF4jhfii.png?imageslim" width="80%" height="80%"/>

3. 回到当前版本：
   存到上图中我们可以看到当前版本已经变成了`add readme file`版本，如果当前的命令行还未关闭的话就可以通过同样的命令返回到最新的版本，否则`add readme file`版本后的所有版本都会消失。

#### 管理修改
在Git中实际上跟踪管理的是修改，而非文件，`git add`命令实际上就是把要提交的所有修改放到暂存区，然后执行`git commit`就可以一次性把暂存区的所有修改提交到分支。
##### 撤销修改
如果我们在文件`README.txt`胡乱输入了一些内容，那我们怎么将其修正呢，通过`git status`我么可以看到我们可以通过`git checkout -- README.txt`命令来丢弃工作区的修改，当然你也可以恢复到上个版本或者直接删掉最后一行。
加入我们`README.txt`中现在如下所示：
```txt
Hello World!
sdfadsfdasfaew
```
通过输入以下命令我们可以将修改丢弃，我们会发现`README.txt`又变成了原样：
```git
git checkout -- README.txt
```
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/iIL56ff98l.png?imageslim" width="60%" height="60%"/>

##### 删除文件
首先我们新建`test.txt`并将其添加到Git中，删除文件一共有两种情况：第一种就是真的想要删除文件，当你在文件管理器中删除了该文件，你可以通过`git rm test.txt`删除版本库中的该文件；第二种是删错了，这时候你可以通过`git checkout -- test.txt`将误删的文件恢复到最新版本

##### 待续...