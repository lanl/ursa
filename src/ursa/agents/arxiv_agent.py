import base64
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from urllib.parse import quote

import feedparser
import pymupdf
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from PIL import Image
from tqdm import tqdm
from typing_extensions import List, TypedDict

from .base import BaseAgent

try:
    from openai import OpenAI
except Exception:
    pass

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# embeddings = OpenAIEmbeddings()


class PaperMetadata(TypedDict):
    arxiv_id: str
    full_text: str


class PaperState(TypedDict, total=False):
    query: str
    context: str
    papers: List[PaperMetadata]
    summaries: List[str]
    final_summary: str


def describe_image(image: Image.Image) -> str:
    if "OpenAI" not in globals():
        print(
            "Vision transformer for summarizing images currently only implemented for OpenAI API."
        )
        return ""
    client = OpenAI()

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a scientific assistant who explains plots and scientific diagrams.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this scientific image or plot in detail.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def extract_and_describe_images(
    pdf_path: str, max_images: int = 5
) -> List[str]:
    doc = pymupdf.open(pdf_path)
    descriptions = []
    image_count = 0

    for page_index in range(len(doc)):
        if image_count >= max_images:
            break
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            if image_count >= max_images:
                break
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))

            try:
                desc = describe_image(image)
                descriptions.append(
                    f"Page {page_index + 1}, Image {img_index + 1}: {desc}"
                )
            except Exception as e:
                descriptions.append(
                    f"Page {page_index + 1}, Image {img_index + 1}: [Error: {e}]"
                )
            image_count += 1

    return descriptions


def remove_surrogates(text: str) -> str:
    return re.sub(r"[\ud800-\udfff]", "", text)


class ArxivAgent(BaseAgent):
    def __init__(
        self,
        llm="openai/o3-mini",
        summarize: bool = True,
        process_images=True,
        max_results: int = 3,
        download_papers: bool = True,
        rag_embedding=None,
        database_path="arxiv_papers",
        summaries_path="arxiv_generated_summaries",
        vectorstore_path="arxiv_vectorstores",
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.summarize = summarize
        self.process_images = process_images
        self.max_results = max_results
        self.database_path = database_path
        self.summaries_path = summaries_path
        self.vectorstore_path = vectorstore_path
        self.download_papers = download_papers
        self.rag_embedding = rag_embedding

        self.graph = self._build_graph()

        os.makedirs(self.database_path, exist_ok=True)

        os.makedirs(self.summaries_path, exist_ok=True)

    def _fetch_papers(self, query: str) -> List[PaperMetadata]:
        if self.download_papers:
            encoded_query = quote(query)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={self.max_results}"
            #            print(f"URL is {url}") # if verbose
            entries = []
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                feed = feedparser.parse(response.content)
                #                print(f"parsed response status is {feed.status}") # if verbose
                entries = feed.entries
                if feed.bozo:
                    raise Exception("Feed from arXiv looks like garbage =(")
            except requests.exceptions.Timeout:
                print("Request timed out while fetching papers.")
            except requests.exceptions.RequestException as e:
                print(f"Request error encountered while fetching papers: {e}")
            except ValueError as ve:
                print(f"Value error occurred while fetching papers: {ve}")
            except Exception as e:
                print(
                    f"An unexpected error occurred while fetching papers: {e}"
                )

            for i, entry in enumerate(entries):
                full_id = entry.id.split("/abs/")[-1]
                arxiv_id = full_id.split("/")[-1]
                title = entry.title.strip()
                # authors = ", ".join(author.name for author in entry.authors)
                pdf_url = f"https://arxiv.org/pdf/{full_id}.pdf"
                pdf_filename = os.path.join(
                    self.database_path, f"{arxiv_id}.pdf"
                )

                if os.path.exists(pdf_filename):
                    print(
                        f"Paper # {i + 1}, Title: {title}, already exists in database"
                    )
                else:
                    print(f"Downloading paper # {i + 1}, Title: {title}")
                    response = requests.get(pdf_url)
                    with open(pdf_filename, "wb") as f:
                        f.write(response.content)

        papers = []

        pdf_files = [
            f
            for f in os.listdir(self.database_path)
            if f.lower().endswith(".pdf")
        ]

        for i, pdf_filename in enumerate(pdf_files):
            full_text = ""
            arxiv_id = pdf_filename.split(".pdf")[0]
            vec_save_loc = self.vectorstore_path + "/" + arxiv_id

            if self.summarize and not os.path.exists(vec_save_loc):
                try:
                    loader = PyPDFLoader(
                        os.path.join(self.database_path, pdf_filename)
                    )
                    pages = loader.load()
                    full_text = "\n".join([p.page_content for p in pages])

                    if self.process_images:
                        image_descriptions = extract_and_describe_images(
                            os.path.join(self.database_path, pdf_filename)
                        )
                        full_text += (
                            "\n\n[Image Interpretations]\n"
                            + "\n".join(image_descriptions)
                        )

                except Exception as e:
                    full_text = f"Error loading paper: {e}"

            papers.append({
                "arxiv_id": arxiv_id,
                "full_text": full_text,
            })

        return papers

    def _fetch_node(self, state: PaperState) -> PaperState:
        papers = self._fetch_papers(state["query"])
        return {**state, "papers": papers}

    def _get_or_build_vectorstore(self, paper_text: str, arxiv_id: str):
        os.makedirs(self.vectorstore_path, exist_ok=True)

        persist_directory = os.path.join(self.vectorstore_path, arxiv_id)

        if os.path.exists(persist_directory):
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.rag_embedding,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            docs = splitter.create_documents([paper_text])
            vectorstore = Chroma.from_documents(
                docs, self.rag_embedding, persist_directory=persist_directory
            )

        return vectorstore.as_retriever(search_kwargs={"k": 5})

    def _summarize_node(self, state: PaperState) -> PaperState:
        prompt = ChatPromptTemplate.from_template("""
        You are a scientific assistant responsible for summarizing extracts from research papers, in the context of the following task: {context}
    
        Summarize the retrieved scientific content below.
    
        {retrieved_content}
        """)

        chain = prompt | self.llm | StrOutputParser()

        summaries = [None] * len(state["papers"])
        relevancy_scores = [0.0] * len(state["papers"])

        def process_paper(i, paper):
            arxiv_id = paper["arxiv_id"]
            summary_filename = os.path.join(
                self.summaries_path, f"{arxiv_id}_summary.txt"
            )

            try:
                cleaned_text = remove_surrogates(paper["full_text"])
                if self.rag_embedding:
                    retriever = self._get_or_build_vectorstore(
                        cleaned_text, arxiv_id
                    )

                    relevant_docs_with_scores = (
                        retriever.vectorstore.similarity_search_with_score(
                            state["context"], k=5
                        )
                    )

                    if relevant_docs_with_scores:
                        score = sum([
                            s for _, s in relevant_docs_with_scores
                        ]) / len(relevant_docs_with_scores)
                        relevancy_scores[i] = abs(1.0 - score)
                    else:
                        relevancy_scores[i] = 0.0

                    retrieved_content = "\n\n".join([
                        doc.page_content for doc, _ in relevant_docs_with_scores
                    ])
                else:
                    retrieved_content = cleaned_text

                summary = chain.invoke({
                    "retrieved_content": retrieved_content,
                    "context": state["context"],
                })

            except Exception as e:
                summary = f"Error summarizing paper: {e}"
                relevancy_scores[i] = 0.0

            with open(summary_filename, "w") as f:
                f.write(summary)

            return i, summary

        if "papers" not in state or len(state["papers"]) == 0:
            print(
                "No papers retrieved - bad query or network connection to ArXiv?"
            )
            return {**state, "summaries": None}

        with ThreadPoolExecutor(
            max_workers=min(32, len(state["papers"]))
        ) as executor:
            futures = [
                executor.submit(process_paper, i, paper)
                for i, paper in enumerate(state["papers"])
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Summarizing Papers",
            ):
                i, result = future.result()
                summaries[i] = result

        if self.rag_embedding:
            print(f"\nMax Relevancy Score: {max(relevancy_scores)}")
            print(f"Min Relevancy Score: {min(relevancy_scores)}")
            print(
                f"Median Relevancy Score: {statistics.median(relevancy_scores)}\n"
            )

        return {**state, "summaries": summaries}

    def _aggregate_node(self, state: PaperState) -> PaperState:
        summaries = state["summaries"]
        papers = state["papers"]
        formatted = []

        if (
            "summaries" not in state
            or state["summaries"] is None
            or "papers" not in state
            or state["papers"] is None
        ):
            return {**state, "final_summary": None}

        for i, (paper, summary) in enumerate(zip(papers, summaries)):
            citation = f"[{i + 1}] Arxiv ID: {paper['arxiv_id']}"
            formatted.append(f"{citation}\n\nSummary:\n{summary}")

        combined = "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(formatted)

        with open(self.summaries_path + "/summaries_combined.txt", "w") as f:
            f.write(combined)

        prompt = ChatPromptTemplate.from_template("""
            You are a scientific assistant helping extract insights from summaries of research papers.
            
            Here are the summaries of a large number of extracts from scientific papers:

            {Summaries}
            
            Your task is to read all the summaries and provide a response to this task: {context}
            """)

        chain = prompt | self.llm | StrOutputParser()

        final_summary = chain.invoke({
            "Summaries": combined,
            "context": state["context"],
        })

        with open(self.summaries_path + "/final_summary.txt", "w") as f:
            f.write(final_summary)

        return {**state, "final_summary": final_summary}

    def _build_graph(self):
        builder = StateGraph(PaperState)
        builder.add_node("fetch_papers", self._fetch_node)

        if self.summarize:
            builder.add_node("summarize_each", self._summarize_node)
            builder.add_node("aggregate", self._aggregate_node)

            builder.set_entry_point("fetch_papers")
            builder.add_edge("fetch_papers", "summarize_each")
            builder.add_edge("summarize_each", "aggregate")
            builder.set_finish_point("aggregate")

        else:
            builder.set_entry_point("fetch_papers")
            builder.set_finish_point("fetch_papers")

        graph = builder.compile()
        return graph

    def run(self, arxiv_search_query: str, context: str) -> str:
        result = self.graph.invoke({
            "query": arxiv_search_query,
            "context": context,
        })

        if self.summarize:
            return result.get("final_summary", "No summary generated.")
        else:
            return "\n\nFinished Fetching papers!"


if __name__ == "__main__":
    agent = ArxivAgent()
    result = agent.run(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )

    print(result)
