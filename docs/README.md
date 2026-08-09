# Demo Assets

Use this folder for GitHub presentation assets.

Recommended layout:

```text
docs/
├── screenshots/
│   └── api-result.png
└── videos/
    └── agent-demo.mp4
```

Suggested screenshots:

1. FastAPI service running or `/docs` page
2. `/predict` response with detected classes, boxes and scores
3. Agent CLI trace showing `intent`, `tool_calls`, `memory_used` and `decision_trace`

Suggested video flow:

1. Start the API server
2. Run the Agent on `test.png`
3. Show the trace output and final answer
4. Ask a follow-up question that uses memory

After adding assets, update the Demo section in the root `README.md` with image or video links.

