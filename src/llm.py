import os

from decouple import config
from openai import OpenAI
from google import genai
from abc import ABC, abstractmethod 

class LLM(ABC):
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model
        self.client = None
    
    @abstractmethod
    def query(self, prompt: str) -> str:
        pass
        
class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o"):
        assert config("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
        super().__init__(host="openai", model=model)
        self.client = OpenAI(api_key=config("OPENAI_API_KEY"))
        
    def query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    

class GoogleLLM(LLM):
    def __init__(self, model: str = "gemini-2.0-flash"):
        assert config("GOOGLE_API_KEY"), "GOOGLE_API_KEY is not set"
        super().__init__(host="google", model=model)
        self.client = genai.Client(api_key=config("GOOGLE_API_KEY"))
        
    def query(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text


def llm_factory(model: str) -> LLM:
    if "gpt" in model:
        return OpenAILLM(model=model)
    elif "gemini" in model:
        return GoogleLLM(model=model)
    else:
        raise ValueError(f"Invalid model: {model}")
