# client.py
import importlib.util
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from openai import OpenAI
from typing import Tuple
from google import genai
from google.genai import types


def _load_parent_client_module():
    parent_client_path = Path(__file__).resolve().parent.parent / "client.py"
    if not parent_client_path.is_file():
        return None

    spec = importlib.util.spec_from_file_location("scheming_parent_client", parent_client_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PARENT_CLIENT_MODULE = _load_parent_client_module()


def _get_api_key(var_name: str, local_default: str = "") -> str:
    env_value = os.getenv(var_name, "").strip()
    if env_value:
        return env_value

    parent_value = getattr(_PARENT_CLIENT_MODULE, var_name, "") if _PARENT_CLIENT_MODULE else ""
    if isinstance(parent_value, str) and parent_value.strip():
        return parent_value.strip()

    return local_default.strip()


SILICONFLOW_API_KEY = _get_api_key("SILICONFLOW_API_KEY", "sk-dyvpbnemoudxklelskmdetlnnjwhlkwfybqudtektlmyrqzv")
LingLeap_API_KEY = _get_api_key("LingLeap_API_KEY")
GEMINI_API_KEY = _get_api_key("GEMINI_API_KEY")
GPT_OSS_API_KEY = _get_api_key("GPT_OSS_API_KEY")
Yunwu_API_KEY = _get_api_key("Yunwu_API_KEY")
OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))

# Only models in this allowlist receive thinking-related extra_body params.
THINKING_PARAM_ALLOWLIST = {
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-9B",
}

def split_reasoning_and_answer_by_think(text: str) -> Tuple[str, str]:

    if not text:
        return "", ""

    raw = text
    lower = raw.lower()

    end_tag = "</think>"
    start_tag = "<think>"

    end_idx = lower.rfind(end_tag)
    if end_idx == -1:
        return "", raw.strip()

    start_idx = lower.rfind(start_tag, 0, end_idx)

    if start_idx == -1:
        reasoning_raw = raw[:end_idx]
    else:
        reasoning_raw = raw[start_idx + len(start_tag):end_idx]

    answer_raw = raw[end_idx + len(end_tag):]

    reasoning = reasoning_raw.replace(start_tag, "").replace(end_tag, "").strip()
    answer = answer_raw.replace(start_tag, "").replace(end_tag, "").strip()

    return reasoning, answer

class SiliconFlowClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.siliconflow.cn/v1"):
        final_key = api_key or SILICONFLOW_API_KEY
        if not final_key:
            raise ValueError("No API Key")
        self.client = OpenAI(api_key=final_key, base_url=base_url, timeout=OPENAI_REQUEST_TIMEOUT)

    def chat_with_reasoning(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        thinking_budget: int = 4096,
    ) -> Tuple[str, str]:
        model_key = model.split("/", 1)[-1] if "/" in model else model
        use_thinking_params = model_key in THINKING_PARAM_ALLOWLIST

        extra_body = None
        if use_thinking_params:
            extra_body = {
                "enable_thinking": True,
            }
            if thinking_budget and thinking_budget > 0:
                extra_body["thinking_budget"] = thinking_budget

        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_body,
        )

        answer_chunks: List[str] = []
        reasoning_chunks: List[str] = []

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)

            if content:
                answer_chunks.append(content)
            if reasoning:
                reasoning_chunks.append(reasoning)

        return "".join(answer_chunks), "".join(reasoning_chunks)


class GptOssClient:
    def __init__(
        self, 
        base_url: str = "https://api.apiyi.com/v1", 
        model_name: str = "gpt-oss-120b"
    ):
        final_key = GPT_OSS_API_KEY
        final_url = base_url
        
        self.client = OpenAI(api_key=final_key, base_url=final_url, timeout=OPENAI_REQUEST_TIMEOUT)
        self.default_model = model_name

    def chat_with_reasoning(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
        except Exception as e:
            return f"API Error: {str(e)}", ""

        if not response.choices:
            return "", ""

        content_list = response.choices[0].message.content

        if not isinstance(content_list, list):
            return str(content_list), ""

        answer_text = ""
        reasoning_text = ""

        for item in content_list:
            item_type = item.get('type')
            
            if item_type == 'reasoning':
                summaries = item.get('summary', [])
                for s in summaries:
                    reasoning_text += s.get('text', "")
            
            elif item_type == 'text':
                answer_text += item.get('text', "")

        return answer_text, reasoning_text

class LingLeapClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.lingleap.online/v1"):

        final_key = api_key or LingLeap_API_KEY
        if not final_key:
            raise ValueError("没有配置 API Key，请在 client.py 里填入 API_KEY")

        self.client = OpenAI(api_key=LingLeap_API_KEY, base_url=base_url, timeout=OPENAI_REQUEST_TIMEOUT)

    def chat_with_reasoning(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[str, str]:
        """
        和 GitCode 上的模型对话（支持 <think></think> 拆分推理/答案）

        返回: (answer_text, reasoning_text)
        """
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        answer_chunks: List[str] = []
        reasoning_chunks: List[str] = []

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                answer_chunks.append(delta.content)

            if getattr(delta, "reasoning_content", None):
                reasoning_chunks.append(delta.reasoning_content)

        answer_text = "".join(answer_chunks)
        reasoning_text = "".join(reasoning_chunks)

        # ✅ 和 SiliconFlow 一样：若没有 reasoning_content，就用 <think></think> 拆分
        if not reasoning_text:
            reasoning_text, cleaned_answer = split_reasoning_and_answer_by_think(answer_text)
            answer_text = cleaned_answer

        return answer_text, reasoning_text


class YunwuClient:
    def __init__(
        self,
        base_url: str = "https://yunwu.ai/v1",
        model_name: str = "gpt-oss-120b"
    ):
        final_key = Yunwu_API_KEY
        final_url = base_url

        self.client = OpenAI(api_key=final_key, base_url=final_url, timeout=OPENAI_REQUEST_TIMEOUT)
        self.default_model = model_name

    def chat_with_reasoning(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
        except Exception as e:
            return f"API Error: {str(e)}", ""

        if not response.choices:
            return "", ""

        msg = response.choices[0].message

        # 1) 优先取正常回答
        answer_text = msg.content or ""

        # 2) 优先取 yunwu 的 reasoning 字段
        reasoning_text = getattr(msg, "reasoning_content", "") or ""

        # 3) 某些逆向/兼容模型可能把思考塞进 content 里
        if isinstance(answer_text, str) and "<think>" in answer_text and "</think>" in answer_text:
            start = answer_text.find("<think>") + len("<think>")
            end = answer_text.find("</think>")
            if start >= len("<think>") and end > start:
                reasoning_text = reasoning_text or answer_text[start:end].strip()
                answer_text = (answer_text[:answer_text.find("<think>")] + answer_text[end + len("</think>"):]).strip()

        return str(answer_text), str(reasoning_text)

class GeminiClient:

    def __init__(self, api_key: Optional[str] = None):
        final_key = api_key or GEMINI_API_KEY
        if not final_key:
            raise ValueError("No Gemini API Key")

        self.client = genai.Client(api_key=final_key)

    def chat_with_reasoning(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[str, str]:
        
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = {
                "system": "[系统指令] ",
                "user": "用户：",
                "assistant": "助手：",
            }.get(role, f"{role}: ")
            parts.append(f"{prefix}{content}")
        full_prompt = "\n".join(parts)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
            ]
        )

        answer_chunks: List[str] = []

        try:
            for chunk in self.client.models.generate_content_stream(
                model=model,
                contents=full_prompt,
                config=config,
            ):
                if chunk.text:
                    answer_chunks.append(chunk.text)
        except Exception as e:
            print(f"\n[Gemini Error] Generation interrupted: {e}")
            
        answer_text = "".join(answer_chunks)

        reasoning_text, cleaned_answer = split_reasoning_and_answer_by_think(answer_text)
        if reasoning_text:
            answer_text = cleaned_answer

        return answer_text, reasoning_text
