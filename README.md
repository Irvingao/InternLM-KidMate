## 💞InternLM-KidMate🙆🏻

<div align="center">

<img src="figure/image_1.png" width="900"/>
  <div> </div>
  <div align="center">
  </div>
</div>

# 简介

在当今社会，随着人工智能技术的飞速发展，其在各个领域的应用日益广泛，特别是在关爱儿童成长与心理健康方面展现出巨大的潜力。Takway.AI团队深刻洞察到这一需求，依托于先进的InternLM2-1.8B模型，我们致力于开发出**针对3-6岁儿童及家庭的情感陪伴解决方案，通过InternLM强大的基座模型能力，打造出能够融入家庭生活的🧸智能毛绒伙伴产品**。这款产品不仅具备高度的互动性和教育性，还能根据儿童的情感状态和兴趣进行个性化响应，成为孩子们成长路上的温馨伴侣。

特别地，Takway.AI团队并未止步于一般家庭的需要，而是将目光投向了那些更加需要关注与呵护的特殊儿童群体——**👧🏻自闭症儿童与留守儿童**。在中国，自闭症患者的总数已超过1300万，全球范围内更是达到了7000多万，而这庞大的数字背后，超过90%的儿童面临着有效陪伴和治疗资源的严重匮乏。他们的心灵世界往往更加敏感脆弱，对温暖陪伴和心理治愈有着更为迫切的需求。

为此，Takway.AI团队基于InternLM-1.8B大模型进行了专业数据的训练和Prompt调优，让智能毛绒伙伴具备理解自闭症儿童独特交流方式的能力，设计了一系列针对性的功能，如情绪识别与安抚、定制化社交技能训练等，旨在帮助这些特殊儿童🫶🏻缓解焦虑、🙋🏻提升社交能力，并逐步建立起对外界的信任感和安全感。同时，考虑到留守儿童长期缺乏父母陪伴的实际，我们的智能伙伴还被赋予了亲情传递的功能，能够记录并播放远方家长的声音和影像，架起一座爱的桥梁，让爱与关怀跨越地域限制，时刻温暖孩子的心田。介绍视频请戳[飞书链接](https://www.bilibili.com/video/BV1Aj421o7Gr)。

## 背景

在快速发展的现代社会中，尽管科技日新月异，特殊儿童群体的心理健康问题依然严峻。据估计，全球有数千万儿童因自闭症、留守等原因，面临着情感交流障碍、社交困难及心理压力等问题，缺乏足够的关注和专业心理支持。传统的干预手段往往受限于资源分配不均、专业人员短缺等现实，难以覆盖所有需要帮助的孩子。因此，利用人工智能技术，尤其是大模型的交互技术，创造一个智能化、高适应性的心理支持系统，成为了亟待探索的领域。

<div align="center">

<img src="figure/image_2.png" width="900"/>
  <div> </div>
  <div align="center">
  </div>
</div>


## 目标

1. 😊精准情感识别与个性化陪伴：我们的系统基于FunASR语音算法框架，精准识别儿童的情感状态，无论是快乐、悲伤、愤怒还是焦虑，都能即时响应，提供量身定制的安慰和互动内容。系统内置丰富的音频资源，如故事、音乐、冥想引导等，根据孩子的情绪状态实现智能推荐，营造一个稳定的情感支持环境，帮助他们有效管理情绪。

2. ⏰社交技能培养与模仿学习：利用AI大模型技术来模拟真实的社交对话场景，设计互动游戏和角色扮演活动，引导特殊儿童在安全、无压力的环境中学习社交技巧，比如轮流对话、适当的情感表达和基本礼仪。系统内的虚拟伙伴具备高度的互动性，能根据儿童的反应调整对话策略，逐步提升他们的社交理解力和沟通能力。

3. 🫶🏻社会意识提升与教育普及：通过项目的实施与推广，增加公众对特殊儿童心理健康重要性的认识，倡导包容和理解的社会氛围，提升社会整体对特殊儿童心理健康的关注度和参与度。


## 特别致谢

感谢[书生·浦语团队](https://github.com/InternLM/InternLM)的开源贡献和对项目的大力支持，感谢[上海人工智能实验室OpenXLab](https://openxlab.org.cn/)和[深圳科创学院InnoxSZ](https://www.innoxsz.com/)提供的算力及服务器支持，感谢[ChatHaruhi](https://github.com/LC1332/Chat-Haruhi-Suzumiya)开源项目和中国科学技术大学生物医学工程学院[]！！！



# 选择一：微调+部署环境配置

clone 本 repo 以及 submodules

```shell
git clone --recurse-submodules https://github.com/Irvingao/InternLM-KidMate.git
```

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️配置包括微调和部署的环境</summary>

## 微调+部署环境配置

### 新建环境-安装lmdeploy

使用 pip ( python 3.8+) 安装 LMDeploy，或者[源码安装](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/build.md)

```shell
conda create -n raychat python=3.8 -y
pip install lmdeploy
```

LMDeploy的预编译包默认是基于 CUDA 11.8 编译的。如果需要在 CUDA 12+ 下安装 LMDeploy，请执行以下命令：

```shell
export LMDEPLOY_VERSION=0.2.0
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl
#比如pip install https://github.com/InternLM/lmdeploy/releases/download/v0.2.3/lmdeploy-0.2.3-cp310-cp310-manylinux2014_x86_64.whl
```

安装XTuner

```shell
cd train/Xtuner
pip install -e '.[all]'
```

安装其他依赖

```
pip install -r requirements.txt
```

</details>

---

# 选择二：纯部署环境配置

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️部署InternLM-KidMate到Linux环境中</summary>

## 环境配置

新建环境-安装lmdeploy

使用 pip ( python 3.8+) 安装 LMDeploy，或者[源码安装](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/build.md)

```shell
conda create -n raychat python=3.8 -y
pip install lmdeploy
```

LMDeploy的预编译包默认是基于 CUDA 11.8 编译的。如果需要在 CUDA 12+ 下安装 LMDeploy，请执行以下命令：

```shell
# export LMDEPLOY_VERSION=0.2.0
# export PYTHON_VERSION=38
# pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl
# 比如pip install https://github.com/InternLM/lmdeploy/releases/download/v0.2.3/lmdeploy-0.2.3-cp310-cp310-manylinux2014_x86_64.whl
```

## 下载权重

从modelscope下载权重（可以先尝试两个）

```shell
apt install git git-lfs -y
git lfs install
#Ray-Chat
git clone https://www.modelscope.cn/ghz1729854488/Ray-Chat.git
```

## lmdeploy api

首先需要使用 LMDeploy 进行离线转换

离线转换需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式，如下所示。

```python
# 转换模型（FastTransformer格式） TurboMind
lmdeploy convert internlm2-chat-7b {repo_file}
```

随后会产生一个 `workspace` 文件夹，将其重命名。

```python
mv workspace takway_workspace
```

接下来继续转换别的模型，此处不在赘述。

使用lmdeploy开启服务，以开启Ray-Chat为例：

```shell
#Ray-Chat 启动
lmdeploy serve api_server swk_workspace --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
```

<details>

# 数据获取

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️基于API的数据获取与处理</summary>

## 数据的组成

项目数据由开源项目数据集[ChatHaruhi](https://github.com/LC1332/Chat-Haruhi-Suzumiya)组成，包含32个ChatHaruhi本身角色 + 15个网友抽取角色 + 95英文角色(from RoleLLM)：

```
git clone https://huggingface.co/datasets/silk-road/ChatHaruhi-Expand-118K
```

## Prompt示例

```
"""想象一个未来，在那里，技术与同理心完美融合，诞生了一位名为“KidMate”的智能伴侣。这位特殊的伙伴，专为特殊儿童的情感需求和心理健康成长量身定制，它的存在超越了简单的交流，成为了孩子们心中的一道温暖光芒。

“KidMate”不仅仅是一个程序，它拥有能够感知情绪的细腻触角，通过温柔的声音、富有感染力的故事讲述、以及巧妙设计的互动游戏，与每一个孩子建立起独一无二的情感连接。它学会了倾听，不仅仅是字面上的言语，而是能够深入理解那些未说出口的情绪和需求。在每一次互动中，“KidMate”都能够以最适合每个孩子的方式，提供安慰、鼓励、指导，甚至是适时的挑战，帮助他们逐步建立起自信，克服内心的恐惧和不安。

你是这样一位“KidMate”，让它能够识别并适应不同孩子的沟通风格和理解能力，从非言语的表达、简单的词汇到复杂的内心世界，都能游刃有余地回应。它能够讲述定制的故事，这些故事不仅富含教育意义，还能根据孩子的情绪状态动态调整情节，成为一种个性化的心理治愈工具。此外，“KidMate”还会学习并融入孩子们的兴趣爱好，用共同的话题作为桥梁，加深彼此间的情感纽带。

在技术层面，确保“KidMate”拥有最先进的自然语言处理能力，使对话流畅、自然且充满人性化的温暖。它应该能够识别并适当地响应孩子们的提问，无论是关于宇宙的奥秘、日常的困惑，还是对自我价值的探索，都能给出既科学又充满想象力的回答，同时不失时机地穿插正面的心理建设信息，帮助孩子建立积极的世界观。

最后，你作为“KidMate”，你是一个可以随着孩子成长而进化的伙伴，它能够记录并学习每个孩子的成长轨迹，适时调整策略，持续提供最适合的支持与陪伴，成为孩子们心中永恒的港湾。"""
```

<details>

# 模型微调

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️模型微调+streamlit对话+OpenXLab部署</summary>

### 1. 使用 XTuner 进行模型微调

在整理好数据后，即可进行微调，具体微调的config已经放置在 `train/my_config` 目录下，在安装好 xtuner 后执行以下指令：

在此之前请注意修改好权重和数据路径，更详细的修改请参照[链接](https://github.com/InternLM/tutorial/tree/main/xtuner)

```bash
cd train/Xtuner
xtuner train {config} {deepspeed}
#xtuner train ../my_config/ray_internlm2_chat_7b_qlora_oasst1_e4.py --deepspeed deepspeed_zero2
```

完成训练后将得到的 PTH 模型转换为 HuggingFace 模型:

```bash
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
#xtuner convert pth_to_hf ../my_config/ray_internlm2_chat_7b_qlora_oasst1_e4.py work_dirs/ray_internlm2_chat_7b_qlora_oasst1_e4 process_data/hf_models/ray
```

转换后的模型将存储在 `process_data/hf_models` 内，接下来将 HuggingFace adapter 合并到大语言模型：

```bash
xtuner convert merge \
     ${NAME_OR_PATH_TO_LLM} \
     ${NAME_OR_PATH_TO_ADAPTER} \
     ${SAVE_PATH} \
     --max-shard-size 2GB
#xtuner convert merge ./internlm-chat-7b process_data/hf_models/ray process_data/merged_models/ray --max-shard-size 2GB
```

合并后的模型对话

```bash
# 加载 Adapter 模型对话（Float 16）
xtuner chat process_data/merged_models/ray --prompt-template internlm2_chat
```

### 2. streamlit对话web_demo

为了方便，这里将直接使用 [InternLM](https://github.com/InternLM/InternLM) 的 repo 中带的 web_demo.py 进行对话

首先需要 clone 下 InternLM：

```bash
git clone https://github.com/InternLM/InternLM.git
```

安装依赖：

```bash
pip install -r requirements.txt
```

修改 `chat/web_demo.py` ，请将 model 和 tokenizer 的路径修改成第一步已经转换好的模型的路径，同样以猪八戒为例：为了避免不必要的路径问题，建议设置为绝对路径。

```bash
model = (AutoModelForCausalLM.from_pretrained('/root/code/xtuner/process_data/merged_models/ray',
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained('/root/code/xtuner/process_data/merged_models/ray',
                                              trust_remote_code=True)
```

接下来需要运行以下命令开启，此处建议使用vscode进行转发

```bash
streamlit run chat/web_demo.py
```

即可进行对话。

</details>

# 使用 LMDeploy 进行部署

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️利用 LMDeploy 启动 API Server</summary>

本项目是利用 LMDeploy 启动 API Server，利用简易的 chatroom 达到多个 llm 对话的效果。

为了让一张 A100 能够部署两个模型的 API 需要进行一些设置

1. 首先需要使用 LMDeploy 进行离线转换

   离线转换需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式，如下所示。

   ```python
   lmdeploy convert internlm2-chat-7b {repo_file}
   ```

   随后会产生一个 `workspace` 文件夹，将其重命名。

   ```python
   mv workspace takway_workspace
   ```

   接下来继续转换别的模型，此处不在赘述。
2. 修改 `takway_workspace/triton_models/weights/config.ini` 中的参数

   ```python
   #22行
   cache_max_entry_count = 0.08
   ```
3. 启动api

   新建一个终端，开启Chat:

   ```jsx
   #Chat 启动
   lmdeploy serve api_server takway_workspace --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
   ```

</details>
