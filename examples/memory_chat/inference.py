import os
from typing import Iterator, List, Dict, Tuple, Any, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class LangChainInference:
    """LangChain-based inference engine using OpenAI."""

    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 api_key: str = None,
                 base_url: str = None,
                 temperature: float = 0.0,
                 max_tokens: int = 2048):
        """Initialize the LangChain OpenAI inference engine."""
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LangChain ChatOpenAI client."""
        kwargs = {
            "model_name": self.model_name,
            "openai_api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": True,
        }

        if self.base_url:
            kwargs["openai_api_base"] = self.base_url

        self.client = ChatOpenAI(**kwargs)

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Any]:
        """Convert dict messages to LangChain message objects."""
        langchain_messages = []
        for message in messages:
            content = message["content"]
            role = message["role"]

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> Iterator[str]:
        """Get completion using LangChain and OpenAI."""
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)

            # Configure the client for this request
            self.client.temperature = temperature
            self.client.max_tokens = max_tokens

            if stream:
                for chunk in self.client.stream(langchain_messages):
                    yield chunk.content
            else:
                # Non-streaming response
                response = self.client.invoke(langchain_messages)
                yield response.content

        except Exception as e:
            yield f"Error: {str(e)}"

    async def achat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> AsyncGenerator[str, None]:
        """Async version of chat completion."""
        try:
            langchain_messages = self._convert_messages(messages)

            self.client.temperature = temperature
            self.client.max_tokens = max_tokens

            if stream:
                async for chunk in self.client.astream(langchain_messages):
                    yield chunk.content
            else:
                response = await self.client.ainvoke(langchain_messages)
                yield response.content

        except Exception as e:
            yield f"Error: {str(e)}"