# ruff: noqa
from langchain.chat_models import init_chat_model

prompt = (
    "When playing a jazz standard, what piano notes to play on a Cm7 chord?"
)

# Ollama
ollama_llm = init_chat_model("ollama:nemotron-3-super", reasoning=True)
ollama_response = ollama_llm.invoke(prompt)
ollama_response.content_blocks
ollama_response.usage_metadata
print(ollama_response.text)

# OpenAI
# llm = init_chat_model("openai:gpt-5-nano", reasoning=True) # fails
openai_llm = init_chat_model(
    "openai:gpt-5-nano", reasoning={"effort": "low", "summary": "auto"}
)
openai_response = openai_llm.invoke(prompt)
openai_response.content_blocks
openai_response.usage_metadata
print(openai_response.text)
