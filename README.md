# RockArt Agent

RockArt Agent is an AI-assisted rock art image analysis project. It combines an
RTMDet-Ins instance segmentation model, a FastAPI inference service, and a
LangGraph-based agent workflow for tool calling, memory-aware follow-up, human
review, and structured analysis.

## Overview

The system is designed for rock art image detection and analysis. A user can
provide an image and a question, and the Agent decides whether to call the vision
tool, answer from previous memory, or respond directly to a capability or usage
question.

Core capabilities:

- Rock art instance detection with RTMDet-Ins and MMDetection
- FastAPI inference endpoint for image prediction
- Bounding box, class label, confidence score, and optional mask output
- LangGraph workflow for routing, retries, human review, memory, and final response
- Optional OpenAI-compatible LLM analysis layer
- CLI interface for demos and debugging

## Repository Structure

```text
rockart/
+-- app.py                         # FastAPI model inference service
+-- agent.py                       # Backward-compatible detection tool wrapper
+-- pyproject.toml                 # Package metadata and CLI entrypoint
+-- requirements.txt               # Shared lightweight dependencies
+-- requirements-api.txt           # API service dependencies
+-- requirements-agent.txt         # Agent dependencies
+-- configs/
|   +-- rtmdet_ins_l_rock_art.py   # MMDetection config
+-- checkpoints/
|   +-- rockart.pth                # Model checkpoint expected at runtime
+-- rockart_agent/
|   +-- state.py                   # Agent state schema and defaults
|   +-- tools.py                   # Detection API tool
|   +-- graph.py                   # LangGraph workflow
|   +-- memory.py                  # SQLite memory layer
|   +-- llm.py                     # Optional OpenAI-compatible LLM client
|   +-- cli.py                     # Command-line runner
|   +-- __init__.py
+-- docs/
|   +-- screenshots/               # Demo screenshots
|   +-- videos/                    # Demo videos or GIFs
+-- tests/
```

## Architecture

```text
User image / query
  -> LangGraph Agent
      -> validate_input
      -> load_memory
      -> decide_intent
          -> detect_instances      image analysis requests
          -> answer_from_memory    previous-result questions
          -> direct_answer         general capability questions
      -> retry_detection
      -> human_review
      -> analyze_detection
      -> save_memory
      -> final_response
```

For image analysis, the Agent calls the detection tool, which sends the image to
the FastAPI `/predict` endpoint. The API loads the MMDetection model and returns
structured detection results. The Agent can then generate a final response using
deterministic logic or an optional LLM.

## Environment Setup

The API service and Agent layer can run in separate Python environments.

API environment:

```bash
conda create -n rockart-api python=3.10 -y
conda activate rockart-api
pip install -U openmim
mim install "mmengine>=0.7.1"
mim install "mmcv>=2.0.0,<2.2.0"
pip install "mmdet>=3.0.0,<3.4.0"
pip install -r requirements-api.txt
```

Agent environment:

```bash
conda create -n rockart-agent python=3.10 -y
conda activate rockart-agent
pip install -r requirements-agent.txt
```

## Configuration

Create a local `.env` file from the example configuration:

```bash
cp .env.example .env
```

Common environment variables:

```bash
MMDET_CONFIG=configs/rtmdet_ins_l_rock_art.py
MMDET_CHECKPOINT=checkpoints/rockart.pth
MMDET_DEVICE=cuda:0
SCORE_THR=0.3

ROCKART_LLM_API_KEY=your_api_key_here
ROCKART_LLM_MODEL=your_model_name_here
ROCKART_LLM_BASE_URL=https://your-provider-base-url
```

## Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

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

## Run the Agent

Run from the project root:

```bash
python -m rockart_agent.cli test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像"
```

Show the routing and tool-call trace:

```bash
python -m rockart_agent.cli test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像" --trace
```

Use the optional LLM analysis node:

```bash
python -m rockart_agent.cli test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像" --use-llm --trace
```

Install as an editable package:

```bash
pip install -e .
rockart-agent test.png --api-url http://127.0.0.1:8000 --query "分析这张岩画图像"
```

## Demo Assets

Demo screenshots can be placed in `docs/screenshots/`.

Demo videos or GIFs can be placed in `docs/videos/`.

Recommended demo flow:

1. Start the FastAPI service.
2. Run the Agent with a rock art image.
3. Show the returned detection result.
4. Show the Agent trace with `intent`, `tool_calls`, `memory_used`, and `decision_trace`.
5. Ask a follow-up question to demonstrate memory-aware response.

## License

See `LICENSE`.
