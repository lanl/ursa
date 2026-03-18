from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from ursa.agents.rag_agent import RAGAgent

# Initialize agent
llm = init_chat_model("openai:gpt-5.2")
embedding = init_embeddings("openai:text-embedding-3-large")

agent = RAGAgent(
    llm=llm,
    embedding=embedding,
    return_k=10,  # Number of chunks to retrieve
    workspace="rag_workspace",
    database_path="document_library",
    vectorstore_path="vectorstore",
)

# Ingest documents and retrieve information
result = agent.invoke({
    "context": "What is the syntax for parallel Bayesian optimization in bohydra?"
})

# Access results
print("bohydra Summary:", result["summary"])

result = agent.invoke({"context": "How does the URSA hypothesizer work?"})

print("URSA Summary:", result["summary"])
