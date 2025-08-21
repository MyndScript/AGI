# Environment Setup for AGI Project

## Python Environment
- Recommended: Python 3.10+
- Create a virtual environment:
  ```pwsh
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- Upgrade pip:
  ```pwsh
  python -m pip install --upgrade pip
  ```

## Install Core Dependencies
- GPT-2: `transformers`, `torch`
- FLAN: `transformers`, `torch`
- MCP integration: (add relevant packages as needed)

Install with:
```pwsh
pip install transformers torch
```

Add additional requirements to `requirements.txt` as the project grows.
