# 我的LLM学习心得

## 一、本地部署llama
本章主要参考 `https://github.com/facebookresearch/llama.git`
### 1、申请llama许可
申请网站（需要科学上网）： `https://ai.meta.com/resources/models-and-libraries/llama-downloads/`
![申请网站](images/applying-web.jpg)
勾选 Llama 2 & Llama Chat 和底部 I accept the terms and conditions 点击 Accept and Continue  
一段时间后，邮箱会收到一封邮件，以 `https://download.llamameta.net/*` 开头的链接就是下载权重参数文件时需要验证的信息。
![邮件](images/applying-url.jpg)
### 2、下载权重参数文件
首先配置虚拟环境
```
conda create -n llama python=3.10
conda activate llama
```
然后克隆llama的git仓库并安装依赖
```
git clone https://github.com/facebookresearch/llama.git 
cd llama
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```
下载权重参数
```commandline
bash download.sh
```
输入第一步所得URL，然后询问下载哪种参数文件
![开始下载](images/download.jpg)
这里可以选择7B，然后等待下载结束
![正在下载](images/downloading.jpg)
参数文件保存路径为`./llama-2-7b/consolidated.00.pth`
### 3、运行模型
llama文件中提供了两个demo以供运行。一个是句子补全任务`example_text_completion.py`，一个是对话任务 `example_chat_completion.py`(需要看下载参数文件时选择的7B还是7B-chat)  
两个常用脚本
```commandline
torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4
torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 4
```
想体验自己的文本输出，可以在`example_text_completion.py`里修改`prompts`变量。`prompts`是一个列表，里面是所有想要补全的文本。
![修改prompts](images/prompts.jpg)
或者修改`example_chat_completion.py`里的`dialogs`变量。按照openai`system`、`user`、`assistant`的标准格式改写即可
![修改dialogs](images/dialogs.jpg)

## (持续更新中......)