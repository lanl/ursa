import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from mp_api.client import MPRester
from tqdm import tqdm
from typing_extensions import List, TypedDict

from .base import BaseAgent


class PaperMetadata(TypedDict):
    arxiv_id: str
    full_text: str


class PaperState(TypedDict, total=False):
    query: str
    context: str
    papers: List[PaperMetadata]
    summaries: List[str]
    final_summary: str


def remove_surrogates(text: str) -> str:
    return re.sub(r"[\ud800-\udfff]", "", text)


class MaterialsProjectAgent(BaseAgent):
    def __init__(
        self,
        llm="openai/o3-mini",
        summarize: bool = True,
        max_results: int = 3,
        database_path: str = "mp_database",
        summaries_path: str = "mp_summaries",
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.summarize = summarize
        self.max_results = max_results
        self.database_path = database_path
        self.summaries_path = summaries_path

        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.summaries_path, exist_ok=True)

        self.graph = self._build_graph()

    def _fetch_node(self, state: Dict) -> Dict:
        f = state["query"]
        els = f["elements"]  # e.g. ["Ga","In"]
        bg = (f["band_gap_min"], f["band_gap_max"])
        e_above_hull = (0, 0)  # only on-hull (stable)
        mats = []
        with MPRester() as mpr:
            # get ALL matching materials…
            all_results = mpr.materials.summary.search(
                elements=els,
                band_gap=bg,
                energy_above_hull=e_above_hull,
                is_stable=True,  # equivalent filter
            )
            # …then take only the first `max_results`
            for doc in all_results[: self.max_results]:
                mid = doc.material_id
                data = doc.dict()
                # cache to disk
                path = os.path.join(self.database_path, f"{mid}.json")
                if not os.path.exists(path):
                    with open(path, "w") as f:
                        json.dump(data, f, indent=2)
                mats.append({"material_id": mid, "metadata": data})

        return {**state, "materials": mats}

    def _summarize_node(self, state: Dict) -> Dict:
        """Summarize each material via LLM over its metadata."""
        # prompt template
        prompt = ChatPromptTemplate.from_template("""
You are a materials-science assistant. Given the following metadata about a material, produce a concise summary focusing on its key properties:

{metadata}
        """)
        chain = prompt | self.llm | StrOutputParser()

        summaries = [None] * len(state["materials"])

        def process(i, mat):
            mid = mat["material_id"]
            meta = mat["metadata"]
            # flatten metadata to text
            text = "\n".join(f"{k}: {v}" for k, v in meta.items())
            # build or load summary
            summary_file = os.path.join(
                self.summaries_path, f"{mid}_summary.txt"
            )
            if os.path.exists(summary_file):
                with open(summary_file) as f:
                    return i, f.read()
            # optional: vectorize & retrieve, but here we just summarize full text
            result = chain.invoke({"metadata": text})
            with open(summary_file, "w") as f:
                f.write(result)
            return i, result

        with ThreadPoolExecutor(
            max_workers=min(8, len(state["materials"]))
        ) as exe:
            futures = [
                exe.submit(process, i, m)
                for i, m in enumerate(state["materials"])
            ]
            for future in tqdm(futures, desc="Summarizing materials"):
                i, summ = future.result()
                summaries[i] = summ

        return {**state, "summaries": summaries}

    def _aggregate_node(self, state: Dict) -> Dict:
        """Combine all summaries into a single, coherent answer."""
        combined = "\n\n----\n\n".join(
            f"[{i + 1}] {m['material_id']}\n\n{summary}"
            for i, (m, summary) in enumerate(
                zip(state["materials"], state["summaries"])
            )
        )

        prompt = ChatPromptTemplate.from_template("""
        You are a materials informatics assistant. Below are brief summaries of several materials:

        {summaries}

        Answer the user’s question in context:

        {context}
                """)
        chain = prompt | self.llm | StrOutputParser()
        final = chain.invoke({
            "summaries": combined,
            "context": state["context"],
        })
        return {**state, "final_summary": final}

    def _build_graph(self):
        g = StateGraph(dict)  # using plain dict for state
        g.add_node("fetch", self._fetch_node)
        if self.summarize:
            g.add_node("summarize", self._summarize_node)
            g.add_node("aggregate", self._aggregate_node)
            g.set_entry_point("fetch")
            g.add_edge("fetch", "summarize")
            g.add_edge("summarize", "aggregate")
            g.set_finish_point("aggregate")
        else:
            g.set_entry_point("fetch")
            g.set_finish_point("fetch")
        return g.compile()

    def run(self, mp_query: str, context: str) -> str:
        state = {"query": mp_query, "context": context}
        out = self.graph.invoke(state)
        if self.summarize:
            return out.get("final_summary", "")
        return json.dumps(out.get("materials", []), indent=2)


if __name__ == "__main__":
    agent = MaterialsProjectAgent()
    resp = agent.run(
        mp_query="LiFePO4",
        context="What is its band gap and stability, and any synthesis challenges?",
    )
    print(resp)
