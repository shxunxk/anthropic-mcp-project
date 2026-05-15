import requests
from typing import List, Dict, Any

class Claude:
    def __init__(self, model: str):
        self.model = model
        self.base_url = "http://localhost:11434"

    def _message_text(self, message):
        if isinstance(message, dict):
            if "response" in message:
                return message["response"]
            if "content" in message:
                content = message["content"]
                return content if isinstance(content, str) else str(content)
            return str(message)

        return message.content if hasattr(message, "content") else str(message)

    def add_user_message(self, messages: list, message):--
        user_message = {
            "role": "user",
            "content": self._message_text(message),
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": self._message_text(message),
        }
        messages.append(assistant_message)

    def text_from_message(self, message):
        if isinstance(message, dict):
            return self._message_text(message)

        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ):
        # Convert messages to Ollama format
        prompt = ""
        if system:
            prompt += f"System: {system}\n"
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += "Assistant:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "stop": stop_sequences,
            }
        }

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Local Ollama request failed ({response.status_code}): {response.text}"
            ) from exc
        result = response.json()
        return {
            "response": result.get("response", ""),
            "stop_reason": result.get("stop_reason"),
        }