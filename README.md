# RockArt Agent

面向岩画图像的实例分割与分析 Agent。项目包含一个基于 FastAPI 的检测服务，以及一个基于 LangGraph 的 Agent 层，用于根据用户问题决定是否调用岩画检测工具、读取历史记忆，或直接回答使用说明类问题。

## 项目亮点

- 使用 RTMDet-Ins / MMDetection 对岩画图像进行目标检测与实例分割
- 提供 FastAPI `/predict` 接口，支持 bbox、类别、置信度和可选 COCO RLE mask 输出
- 提供 LangGraph Agent 流程，包含意图判断、工具调用、失败重试、人工复核、记忆读取与结果总结
- 支持 OpenAI-compatible LLM，用于把结构化检测结果生成中文分析说明
- 支持 CLI 演示，适合课程展示、项目答辩和 GitHub 展示

## 仓库结构

```text
rockart/
├── app.py                         # FastAPI model inference service
├── agent.py                       # Backward-compatible detection tool wrapper
├── pyproject.toml                 # Editable package and CLI entrypoint
├── requirements.txt               # Shared lightweight dependencies
├── requirements-api.txt           # FastAPI + MMDetection API dependencies
├── requirements-agent.txt         # LangGraph Agent dependencies
├── configs/
│   └── rtmdet_ins_l_rock_art.py   # MMDetection config
├── checkpoints/
│   └── rockart.pth                # Local model weights, not committed to GitHub
├── rockart_agent/
│   ├── state.py                   # Agent state schema and defaults
│   ├── tools.py                   # Detection API tool
│   ├── graph.py                   # LangGraph workflow
│   ├── memory.py                  # SQLite memory for reviewed cases
│   ├── llm.py                     # Optional OpenAI-compatible LLM client
│   └── cli.py                     # Command-line runner
├── docs/
│   ├── README.md                  # Demo asset guide
│   ├── screenshots/               # Put screenshots here
│   └── videos/                    # Put demo videos/GIFs here
└── tests/                         # Reserved for tests
```

## Demo

你可以把截图放到 `docs/screenshots/`，把录屏、GIF 或演示视频放到 `docs/videos/`。

建议展示顺序：

1. 启动 FastAPI 检测服务
2. 上传或传入一张岩画图片
3. 展示 `/predict` 返回的检测结果
4. 使用 `rockart-agent` 或 `python -m rockart_agent.cli` 展示 Agent 的工具调用轨迹
5. 展示记忆问答或人工复核逻辑

## 环境准备

建议把检测服务和 Agent 层拆成两个 Python 环境：检测服务环境安装 MMDetection、PyTorch 和 CUDA 相关依赖；Agent 环境安装 LangGraph、requests 和可选 LLM 客户端。

创建 API 环境：

```bash
conda create -n rockart-api python=3.10 -y
conda activate rockart-api
```

安装 PyTorch、OpenMMLab 和 API 依赖。CUDA 版本需要根据你的服务器环境选择：

```bash
pip install -U openmim
mim install "mmengine>=0.7.1"
mim install "mmcv>=2.0.0,<2.2.0"
pip install "mmdet>=3.0.0,<3.4.0"
pip install -r requirements-api.txt
```

创建 Agent 环境：

```bash
conda create -n rockart-agent python=3.10 -y
conda activate rockart-agent
pip install -r requirements-agent.txt
```

## 配置

复制 `.env.example`，按你的本地路径和服务配置填写实际值。

```bash
cp .env.example .env
```

常用环境变量：

```bash
MMDET_CONFIG=configs/rtmdet_ins_l_rock_art.py
MMDET_CHECKPOINT=checkpoints/rockart.pth
MMDET_DEVICE=cuda:0
SCORE_THR=0.3
ROCKART_LLM_API_KEY=your_api_key_here
ROCKART_LLM_MODEL=your_model_name_here
ROCKART_LLM_BASE_URL=https://your-provider-base-url
```

注意：`checkpoints/rockart.pth` 属于模型权重文件，通常不建议直接提交到 GitHub。公开仓库可以只保留路径说明，并在 README 中补充权重下载方式。

## 启动检测 API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

调用预测接口：

```python
import requests

with open("test.png", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": f},
        params={"score_thr": 0.3, "include_masks": False},
    )

print(response.json())
```

## 运行 Agent

从项目根目录运行：

```bash
python -m rockart_agent.cli test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像"
```

查看工具调用与路由轨迹：

```bash
python -m rockart_agent.cli test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像" --trace
```

启用 LLM 分析节点：

```bash
python -m rockart_agent.cli test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像" --use-llm --trace
```

安装为可编辑包后，也可以直接使用命令行入口：

```bash
pip install -e .
rockart-agent test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像"
```

## Agent 流程

```text
validate_input
  -> load_memory
  -> decide_intent
      -> detect_instances      image analysis requests call the segmentation tool
      -> answer_from_memory    follow-up/history requests use memory only
      -> direct_answer         usage/capability questions do not call tools
  -> retry_detection
  -> human_review
  -> analyze_detection
  -> save_memory
  -> final_response
```

## 提交前检查

提交到 GitHub 前，建议确认没有敏感信息或不该上传的大文件：

```bash
git status
git diff
git grep -i "api_key"
git grep -i "secret"
git grep -i "password"
```

不要提交：

- `.env`
- 真实 API Key、Token、密码
- `__pycache__/`
- `uploads/`
- 本地 SQLite 记忆文件
- 大体积模型权重文件，例如 `checkpoints/*.pth`

## License

License is currently TBD. Choose and replace `LICENSE` before publishing if you want the project to be open source.
