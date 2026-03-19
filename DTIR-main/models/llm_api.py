import time
from openai import OpenAI
import os
import base64
from utils.log import log_info, log_verbose, set_log_level


VOLCENGINE_API_KEY = ""
GPT_API_KEY = ""
QWEN_API_KEY = ""

OPENAI_BASE_URL = ""
VOLCENGINE_BASE_URL = ""
QWEN_BASE_URL = ""

MODEL_CONFIG = {
    "gpt": {
        "api_key": GPT_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "models": {
            "default": "gpt-3.5-turbo",
            "gpt-3.5": "gpt-3.5-turbo",
            "gpt-4": "gpt-4-turbo",
            "gpt-4o": "gpt-4o"
        }
    },
    "deepseek": {
        "api_key": VOLCENGINE_API_KEY,
        "base_url": VOLCENGINE_BASE_URL,
        "models": {
            "default": "deepseek-v3-250324",
            "v3": "deepseek-v3-250324",
            "r1": "deepseek-r1-250120"
        }
    },
    "doubao": {
        "api_key": VOLCENGINE_API_KEY,
        "base_url": VOLCENGINE_BASE_URL,
        "models": {
            "default": "doubao-1-5-lite-32k-250115",
            "lite": "doubao-1-5-lite-32k-250115",
            "pro": "doubao-1-5-pro-32k-250115",
            "vision-lite": "doubao-1.5-vision-lite-250315",
            "vision-pro": "doubao-1-5-vision-pro-32k-250115"
        }
    },
    "qwen": {
        "api_key": QWEN_API_KEY,
        "base_url": QWEN_BASE_URL,
        "models": {
            "default": "qwen-plus",
            "turbo": "qwen-turbo",
            "plus": "qwen-plus-latest",
            "omni-turbo": "qwen-omni-turbo",
            "vl-plus": "qwen-vl-plus",
            "vl-max": "qwen-vl-max",
            "qwen3": "qwen3-235b-a22b"
        }
    }
}

MAX_INPUT_TOKENS = 12800
RETRY_TIMES = 3
RETRY_DELAY = 2  # seconds
class LlmAPIWrapper:
    def __init__(self, provider="deepseek", model=None, api_key=None, base_url=None, sys_prompt="You are a helpful assistant"):
        self.provider = provider.lower()
        
        if self.provider not in MODEL_CONFIG:
            log_info(f"[Warning] Unknown provider '{provider}', defaulting to deepseek")
            self.provider = "deepseek"
        
        config = MODEL_CONFIG[self.provider]
        self.api_key = api_key if api_key else config["api_key"]
        self.base_url = base_url if base_url else config["base_url"]
        if model is None or model not in config["models"]:
            self.model = config["models"]["default"]
        else:
            self.model = config["models"][model]

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.sys_prompt = sys_prompt
        self.clear_history()
        log_info(f"Initialized LLM wrapper with provider: {self.provider}, model: {self.model}")


    def inference(self, prompt, temperature=1.0):
        is_stream = False
        if self.provider == "qwen" and self.model == "qwen-omni-turbo":
            is_stream = True

        message = {"role": "user", "content": prompt}
        self.history_context.append(message)
        # Check if the history context is too long and truncate if necessary
        if len(self.history_context) > 2:  # Keep at least the system prompt and current message
            # Calculate approximate token count (rough estimation)
            total_content = ""
            for msg in self.history_context:
                if isinstance(msg["content"], str):
                    total_content += msg["content"]
                elif isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            total_content += item.get("text", "")
            
            # Rough estimation: 1 token ≈ 4 characters
            estimated_tokens = len(total_content) / 4
            
            # If estimated tokens exceed the limit, truncate history
            if estimated_tokens > MAX_INPUT_TOKENS:
                # Keep system prompt and current message, remove oldest messages
                system_prompt = self.history_context[0]
                current_message = self.history_context[-1]
                # self.history_context = [system_prompt, current_message]
                self.history_context = [current_message]
                log_info(f"[Error] History context truncated due to token limit (est. {int(estimated_tokens)} tokens)")

        output_text = ""
        retry_delay = RETRY_DELAY
        for retry_count in range(RETRY_TIMES):
            try:
                # If not the first attempt, log retry information
                if retry_count > 0:
                    log_info(f"Retrying API call (attempt {retry_count+1}/{RETRY_TIMES})...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history_context,
                    stream=is_stream,
                    temperature=temperature,
                    extra_body={"enable_thinking": False},
                )
                output_text = ""
                if self.provider == "qwen" and self.model == "qwen-omni-turbo":
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            output_text += chunk.choices[0].delta.content
                else:
                    output_text = response.choices[0].message.content

                assistant_message = {"role": "assistant", "content": output_text}
                self.history_context.append(assistant_message)
                break

            except Exception as e:
                log_info(f"[Error] API Exception: {str(e)}. Retrying in {retry_delay} seconds (attempt {retry_count+1}/{RETRY_TIMES})...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 2
            
        return output_text


    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def image_inference(self, prompt, image_path, img_type="jpeg", temperature=1.0):
        try:
            if image_path is not None:
                base64_image = self.encode_image(image_path)
                message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"}, 
                                },
                            ],
                        }
            else:
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
                
            self.history_context.append(message)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history_context,
                temperature=temperature
            )
            output_text = response.choices[0].message.content

            assistant_message = {"role": "assistant", "content": output_text}
            self.history_context.append(assistant_message)
            return output_text
        except Exception as e:
            log_info(f"[Error] API Exception: {str(e)}")
            return ""


    def clear_history(self):
        self.history_context = [{"role": "system", "content": self.sys_prompt}]


    def get_available_providers(self):
        return list(MODEL_CONFIG.keys())


    def get_available_models(self, provider=None):
        if provider is None:
            provider = self.provider
            
        if provider not in MODEL_CONFIG:
            return []
            
        return list(MODEL_CONFIG[provider]["models"].keys())