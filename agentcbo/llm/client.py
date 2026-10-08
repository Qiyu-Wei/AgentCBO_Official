from __future__ import annotations
from typing import Literal

from pydantic import BaseModel

REASONING_MODELS = {'o1', 'o3', 'o3-mini', 'o3-pro', 'o4-mini', 'gpt-5', 'gpt-5-mini'}


class PairwiseJudgment(BaseModel):
    cause: str
    effect: str
    direction: Literal["A->B", "B->A", "none"]
    confidence: float
    reasoning: str


class LLMClient:

    def __init__(self, api_key: str = None, api_base: str = None,
                 model: str = None, temperature: float = 0.0,
                 use_vllm: bool = None):
        import httpx
        from openai import OpenAI
        from agentcbo.config import (OPENAI_API_KEY, OPENAI_API_BASE, DEFAULT_LLM_MODEL,
                                     USE_VLLM, VLLM_BASE_URL, VLLM_MODEL)
        # Bound stalled socket reads so the retry policy can take effect.
        _timeout = httpx.Timeout(timeout=120.0, connect=10.0, read=120.0, write=30.0)
        self._is_vllm = USE_VLLM if use_vllm is None else use_vllm
        if self._is_vllm:
            self.client = OpenAI(api_key='EMPTY', base_url=api_base or VLLM_BASE_URL,
                                 http_client=httpx.Client(timeout=_timeout))
            self.model = model or VLLM_MODEL
        else:
            self.client = OpenAI(api_key=api_key or OPENAI_API_KEY,
                                 base_url=api_base or OPENAI_API_BASE,
                                 http_client=httpx.Client(timeout=_timeout))
            self.model = model or DEFAULT_LLM_MODEL
        self.temperature = temperature
        self.total_calls = 0
        self.total_tokens = 0

    def _is_reasoning_model(self) -> bool:
        return any(self.model.startswith(p) for p in REASONING_MODELS)

    def _build_messages(self, prompt: str, system_prompt: str = None) -> list:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def query(self, prompt: str, system_prompt: str = None) -> str:
        from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        @retry(stop=stop_after_attempt(3),
               wait=wait_exponential(multiplier=1, min=2, max=30),
               retry=retry_if_exception_type((APIConnectionError, APITimeoutError,
                                               RateLimitError, InternalServerError)))
        def _call():
            params = dict(model=self.model,
                          messages=self._build_messages(prompt, system_prompt))
            if not self._is_reasoning_model():
                params['temperature'] = self.temperature
            if self._is_vllm:
                params['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
            response = self.client.chat.completions.create(**params)
            self._track_usage(response)
            if response.choices[0].message.refusal:
                raise ValueError(f"LLM refused: {response.choices[0].message.refusal}")
            return response.choices[0].message.content

        return _call()

    def query_structured(self, prompt: str, response_format: type,
                         system_prompt: str = None):
        from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        @retry(stop=stop_after_attempt(3),
               wait=wait_exponential(multiplier=1, min=2, max=30),
               retry=retry_if_exception_type((APIConnectionError, APITimeoutError,
                                               RateLimitError, InternalServerError)))
        def _call():
            messages = self._build_messages(prompt, system_prompt)
            if self._is_vllm:
                params = dict(model=self.model, messages=messages,
                              response_format={
                                  'type': 'json_schema',
                                  'json_schema': {
                                      'name': response_format.__name__.lower(),
                                      'schema': response_format.model_json_schema(),
                                      'strict': True,
                                  }
                              },
                              extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                if not self._is_reasoning_model():
                    params['temperature'] = self.temperature
                response = self.client.chat.completions.create(**params)
                self._track_usage(response)
                return response_format.model_validate_json(response.choices[0].message.content)
            else:
                params = dict(model=self.model, messages=messages,
                              response_format=response_format)
                if not self._is_reasoning_model():
                    params['temperature'] = self.temperature
                response = self.client.beta.chat.completions.parse(**params)
                self._track_usage(response)
                if response.choices[0].message.refusal:
                    raise ValueError(f"LLM refused: {response.choices[0].message.refusal}")
                return response.choices[0].message.parsed

        return _call()

    def _track_usage(self, response):
        self.total_calls += 1
        if response.usage:
            self.total_tokens += response.usage.total_tokens
