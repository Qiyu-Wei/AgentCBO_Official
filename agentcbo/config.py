import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
DEFAULT_LLM_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')

# vLLM config (USE_VLLM=true to enable)
USE_VLLM = os.environ.get('USE_VLLM', 'false').lower() == 'true'
VLLM_BASE_URL = os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')
VLLM_MODEL = os.environ.get('VLLM_MODEL', 'Qwen/Qwen3.5-35B-A3B')
