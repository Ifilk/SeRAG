import dataclasses
import logging
from enum import Enum
from http import HTTPStatus
from typing import Any, Mapping

import dashscope
import requests
from dashscope.api_entities.dashscope_response import Message
from langchain_core.language_models.llms import LLM
from pydantic import SkipValidation


@dataclasses.dataclass
class ApiCallException(Exception):
    status_code: int
    message: str
    raw_response: Any

    def __str__(self):
        return f'[code={self.status_code}, message={self.message}]'


class AlibabaModelName(Enum):
    bailian_v1 = 'bailian-v1'
    dolly_12b_v2 = 'dolly-12b-v2'
    qwen_turbo = 'qwen-turbo'
    qwen_plus = 'qwen-plus'
    qwen_max = 'qwen-max'


class AlibabaLLM(LLM):
    model_name: AlibabaModelName
    total_input_tokens: SkipValidation[int] = 0
    total_output_tokens: SkipValidation[int] = 0
    total_usage: SkipValidation[int] = 0

    @property
    def _llm_type(self) -> str:
        return self.model_name.name

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            return self._make_api_call(prompt)
        except Exception as e:
            logging.error(f"Error in LLM _call: {e}", exc_info=True)
            raise

    def _make_api_call(self, prompt: str) -> str:
        messages = [Message(role='system', content='You are a helpful assistant.'),
                    Message(role='user', content=prompt)]

        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            self.total_input_tokens = response.usage.input_tokens
            self.total_output_tokens = response.usage.output_tokens
            self.total_usage = response.usage.total_tokens
            return response.output.choices[0].message.content
        else:
            raise ApiCallException(status_code=response.status_code,
                                   message=response.message,
                                   raw_response=response)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_usage": self.total_usage}


class Local(LLM):

    @property
    def _llm_type(self) -> str:
        return 'local'

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            return self._make_api_call(prompt)
        except Exception as e:
            logging.error(f"Error in LLM _call: {e}", exc_info=True)
            raise

    def _make_api_call(self, prompt: str) -> str:
        payload = {
            "model": "*",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        response = requests.post('http://127.0.0.1:8000/v1/chat/completions', json=payload, headers={
            "Content-Type": "application/json",
        }, stream=True)
        if response.status_code == HTTPStatus.OK:
            # _r = response.json()['choices'][0]['message']['content']
            # logging.debug(_r)
            # return _r
            return response.json()['choices'][0]['message']['content']
        else:
            raise ApiCallException(status_code=response.status_code,
                                   message=response.text,
                                   raw_response=response)
