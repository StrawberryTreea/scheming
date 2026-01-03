# client.py
from typing import Optional, List, Dict, Tuple
from openai import OpenAI
from typing import Tuple
from google import genai
from google.genai import types

SILICONFLOW_API_KEY = ""
GEMINI_API_KEY = ""
GPT_OSS_API_KEY = ""

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
        self.client = OpenAI(api_key=final_key, base_url=base_url)

    def chat_with_reasoning(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7, 
        max_tokens: int = 8192,   
        thinking_budget: int = 4096, 
    ) -> Tuple[str, str]:
        
        extra_params = {}
        if thinking_budget > 0:
            extra_params["thinking_budget"] = thinking_budget

        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_params 
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

        if not reasoning_text:
            reasoning_text, cleaned_answer = split_reasoning_and_answer_by_think(answer_text)
            answer_text = cleaned_answer

        return answer_text, reasoning_text


class GptOssClient:
    def __init__(
        self, 
        base_url: str = "https://api.apiyi.com/v1", 
        model_name: str = "gpt-oss-120b"
    ):
        final_key = GPT_OSS_API_KEY
        final_url = base_url
        
        self.client = OpenAI(api_key=final_key, base_url=final_url)
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
