"""Agent that reads model cards, recommends models, and runs them.

``ModelCardAgent`` works in four stages:

1. **Recognize.** Read model cards from Hugging Face, URLs, local files, and
   MCP servers such as Nomad. Each card becomes a normalized
   :class:`~ursa.model_cards.schema.ModelCard` with provenance for every
   fact, including exact input and output parameter formats.
2. **Understand the need.** Turn the user's request into structured
   :class:`~ursa.model_cards.schema.UserRequirements`.
3. **Recommend.** Rank catalog models with explainable criteria and, when
   an LLM is available, an evidence-bound review.
4. **Run.** Call the model's MCP tool with schema-validated arguments, or
   hand a precise brief to the ExecutionAgent.

Example::

    from langchain.chat_models import init_chat_model
    from ursa.agents import ModelCardAgent

    agent = ModelCardAgent(llm=init_chat_model("openai:gpt-5-mini"))
    state = agent.invoke(
        "Read hf://camlab-ethz/Poseidon-B and recommend a model for 2D "
        "compressible flow on a 24 GB GPU."
    )
    print(agent.format_result(state))
"""

import asyncio
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ursa.agents.base import AgentWithTools, BaseAgent
from ursa.model_cards.catalog import ModelCatalog, default_catalog_dir, tokenize
from ursa.model_cards.compat import Assessment, assess_models
from ursa.model_cards.data import DataProfile, inspect_data
from ursa.model_cards.extract import extract_model_card
from ursa.model_cards.llm_extract import refine_card
from ursa.model_cards.markdown import evidence_text
from ursa.model_cards.matching import (
    blend_review,
    catalog_filters,
    filter_catalog,
    rank_models,
)
from ursa.model_cards.report import (
    Citations,
    render_assessments,
    render_batch_table,
    render_card,
    render_catalog_answer,
    render_data_profile,
    render_recommendations,
    render_requirements,
    render_verdict,
)
from ursa.model_cards.requirements import (
    heuristic_requirements,
    parse_requirements,
)
from ursa.model_cards.running import (
    RunPlan,
    build_run_plan,
    execution_brief,
    save_batch_results,
    save_tool_result,
    summarize_tool_result,
    unique_path,
)
from ursa.model_cards.schema import (
    DataDescription,
    ModelCard,
    Recommendation,
    UserRequirements,
)
from ursa.model_cards.sources import (
    find_card_sources,
    is_card_source,
    load_document,
    mcp_tool_document,
    parse_source,
    search_hub,
)
from ursa.model_cards.taxonomy import DOMAINS, TASKS
from ursa.prompt_library.model_card_prompts import review_prompt, route_prompt
from ursa.util.mcp import load_mcp_tools_by_server
from ursa.util.rendering import file_artifact
from ursa.util.structured_output import invoke_structured

Intent = Literal["ingest", "describe", "list", "recommend", "assess", "run"]
MODEL_CARD_TOOL = "get_model_card"
_PREVIOUS_OUTPUT = re.compile(
    r"^The last agent output was:(?P<context>[\s\S]*?)The user stated:(?P<request>[\s\S]*)$"
)
_INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "run": re.compile(
        r"\b(?:run|execute|apply|infer(?:ence)?)\b|\bpredict\b[^.]{0,40}\b(?:for|on) (?:my|this|these|the)\b",
        re.I,
    ),
    "recommend": re.compile(
        r"\b(?:recommend\w*|suggest\w*|which (?:model|fm|foundation)|best (?:model|fm)|choose|select|"
        r"compare|find (?:me )?(?:a |the )?(?:model|fm)|what model)\b",
        re.I,
    ),
    "ingest": re.compile(
        r"\b(?:parse|read|ingest|add|register|import|load|recogni[sz]e)\b[^.]{0,40}\b(?:cards?|models?)\b",
        re.I,
    ),
    "describe": re.compile(
        r"\b(?:describe|explain|details?|summari[sz]e|tell me about)\b|"
        r"\bwhat (?:are|is) the (?:inputs?|outputs?|formats?|parameters?)\b",
        re.I,
    ),
    "list": re.compile(
        r"\b(?:list|catalog(?:ue)?|available models|what models|how many|number of models)\b|"
        r"\bmodels? (?:do|did) (?:i|we) have\b",
        re.I,
    ),
    "assess": re.compile(
        r"\bfine[- ]?tun\w*|\bcompatib\w*|\bcan (?:i|we) (?:just |directly )?(?:run|use|apply|perform)\b|"
        r"\b(?:is|are) (?:it|my data|this data|these data|the data) (?:usable|suitable|ready)\b",
        re.I,
    ),
}
# A model runs only when the request orders it. A sentence orders a run when
# its verb comes first ("Run X", "Now predict Y"), follows "please" or "can
# you", or chains onto another order ("Recommend a model and run it").
# Questions ("Can I just run inference?", "How many models run on a CPU?")
# mention running without asking for it.
_RUN_VERBS = r"(?:run|execute|apply|call|use|predict|compute|calculate)"
_ORDER_VERBS = r"(?:run|execute|apply|predict|compute|calculate)"
_NOT_HARDWARE = (
    r"(?!\s+(?:on|with)\s+(?:a |an |the |my |one )?"
    r"(?:cpu|gpu|laptop|workstation|cluster)s?\b)"
)
_QUESTION = re.compile(
    r"^\s*(?:(?:can|could|should|may|might|must|do|does|did|would|will|shall|is|are|am)\s+"
    r"(?:i|we|it|this|that|they|these|my|the)\b|how\b|what\b|which\b|why\b|"
    r"when\b|where\b|who\b|whether\b)",
    re.I,
)


def _order_patterns(verbs: str) -> tuple[re.Pattern[str], ...]:
    return (
        re.compile(
            r"\b(?:please|(?:can|could|would|will) you(?:\s+please)?|"
            r"i(?:'d| would) like you to|i (?:want|need) you to)\s+"
            rf"{verbs}\b{_NOT_HARDWARE}",
            re.I,
        ),
        re.compile(
            r"^\s*(?:(?:ok(?:ay)?|so|now|then|just|go ahead and|and)[\s,]+)?"
            rf"{verbs}\b{_NOT_HARDWARE}",
            re.I,
        ),
        re.compile(
            rf"(?:\band|\bthen|,)\s+(?:then\s+)?(?:run|execute|apply)\b{_NOT_HARDWARE}",
            re.I,
        ),
    )


_RUN_ORDERS = _order_patterns(_RUN_VERBS)
_STRONG_ORDERS = _order_patterns(_ORDER_VERBS)


def _orders_run(request: str, *, strong: bool = False) -> bool:
    """Whether a sentence of ``request`` asks the agent to run a model now.

    ``strong`` leaves out "use" and "call", which also start requests that
    only ask for a model ("Use the catalog to find a model for ...").
    """
    polite, imperative, chained = _STRONG_ORDERS if strong else _RUN_ORDERS
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", request or ""):
        sentence = sentence.strip()
        if polite.search(sentence):
            return True
        if _QUESTION.match(sentence) or sentence.endswith("?"):
            continue
        if imperative.match(sentence) or chained.search(sentence):
            return True
    return False


_FEASIBILITY = re.compile(
    r"\b(?:can|could|should|may) (?:i|we) (?:just |directly |simply )?"
    r"(?:run|use|apply|perform|do|get)\b|\bdo (?:i|we) need\b|"
    r"\binference or (?:fine[- ]?tun\w*|train\w*)|\bfine[- ]?tun\w*\?",
    re.I,
)
_CHOICE_WORDS = re.compile(
    r"\b(?:recommend\w*|suggest\w*|best|choose|choices?|options?|fits?|suitable)\b",
    re.I,
)
# "Run it on the same file" points back at the data of the previous turn.
_SAME_DATA = re.compile(
    r"\b(?:same|that|this|those|these|previous|last|earlier)\s+"
    r"(?:data(?:set)?|files?|molecules|structures|sequences|samples)\b",
    re.I,
)


class ModelCardState(TypedDict, total=False):
    """State for one ModelCardAgent turn plus results kept across turns."""

    request: str
    context: str
    intents: list[str]
    card_sources: list[str]
    model_refs: list[str]
    ingested: list[str]
    ingest_errors: list[str]
    requirements: UserRequirements
    recommendations: list[Recommendation]
    data_profiles: list[DataProfile]
    assessments: list[Assessment]
    implied_targets: list[str]
    run_plan: RunPlan | None
    run_result: str
    llm_error: str
    report: str
    messages: Annotated[list, add_messages]


class RouteDecision(BaseModel):
    intents: list[Intent] = Field(default_factory=list)
    card_sources: list[str] = Field(
        default_factory=list, description="Card locations named in the request."
    )
    models: list[str] = Field(
        default_factory=list, description="Catalog models named in the request."
    )


class ModelReview(BaseModel):
    model_id: str
    fit: Literal["strong", "moderate", "weak", "incompatible"]
    rationale: str = ""
    caveats: list[str] = Field(default_factory=list)


class RankingReview(BaseModel):
    assessments: list[ModelReview] = Field(default_factory=list)


def _run_async(coro):
    """Run a coroutine from synchronous graph nodes."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _text(result: Any) -> str:
    return summarize_tool_result(result, limit=200_000)


class ModelCardAgent(AgentWithTools, BaseAgent[ModelCardState]):
    """Recognize scientific foundation-model cards, recommend models for a request, and run them.

    Give it model-card sources (Hugging Face ids, URLs, files, or connected
    Nomad/MCP model servers) and a scientific need. It reports each model's
    metadata and exact input/output formats, ranks models with explained
    criteria, and can run the chosen model through its MCP tool or the
    ExecutionAgent.
    """

    state_type = ModelCardState

    def __init__(
        self,
        llm: BaseChatModel,
        catalog_dir: str | Path | None = None,
        catalog_paths: list[str] | str | None = None,
        use_web: bool = False,
        hub_search: bool | None = None,
        llm_extraction: bool = True,
        refine_discovered: bool = False,
        llm_review: bool = True,
        llm_routing: bool = True,
        allow_run: bool = True,
        top_k: int = 3,
        discover_mcp_models: bool = True,
        mcp_model_tools: list[str] | None = None,
        custom_domains: dict | None = None,
        custom_tasks: dict | None = None,
        executor_kwargs: dict | None = None,
        write_reports: bool = True,
        fetch_timeout: float = 30.0,
        max_run_items: int = 100,
        **kwargs,
    ):
        self._base_llm = llm
        super().__init__(llm=llm, tools=[], **kwargs)
        self.use_web = use_web
        self.hub_search = use_web if hub_search is None else hub_search
        self.llm_extraction = llm_extraction
        # Discovery stores every model a server offers before the first
        # answer; one LLM review per card would delay that answer by minutes.
        self.refine_discovered = refine_discovered
        self.llm_review = llm_review
        self.llm_routing = llm_routing
        self.allow_run = allow_run
        self.top_k = max(1, int(top_k))
        self.discover_mcp_models = discover_mcp_models
        self.mcp_model_tools = list(mcp_model_tools or [])
        self.executor_kwargs = dict(executor_kwargs or {})
        self.write_reports = write_reports
        self.fetch_timeout = fetch_timeout
        self.max_run_items = max(1, int(max_run_items))
        self.domains = DOMAINS.extended(custom_domains)
        self.tasks = TASKS.extended(custom_tasks)
        self.catalog_dir = (
            Path(catalog_dir).expanduser()
            if catalog_dir
            else default_catalog_dir(self.group)
        )
        self.catalog = ModelCatalog(self.catalog_dir)
        if isinstance(catalog_paths, str):
            catalog_paths = [p for p in catalog_paths.split(",") if p.strip()]
        self.catalog_paths = [str(p).strip() for p in catalog_paths or []]
        self._seeded = False
        self._discovered = False
        # Why discovery found no served models, for an empty catalog, and
        # servers whose every card request failed.
        self._discovery_hint: str | None = None
        self._discovery_warnings: list[str] = []
        self._tool_sources: dict[str, str] = {}
        # Server name -> (its get_model_card tool, the other tools it serves).
        self._card_servers: dict[str | None, tuple[Any, list[Any]]] = {}
        self._execution_agent = None

    # Tools from MCP servers ------------------------------------------------

    async def add_mcp_tools(self, client, tool_name=None) -> dict[str, str]:
        """Attach MCP tools and remember which server provides each one.

        Tool names carry no server prefix, so with Nomad and another server
        that both offer ``get_model_card``, ``self.tools`` keeps only the last
        one. Discovery therefore keeps each server's card tool and the tools
        that server provides, loaded in the same pass as the attach.
        """
        selected = (
            None
            if tool_name is None
            else set(tool_name if isinstance(tool_name, list) else [tool_name])
        )

        def discover_and_apply():
            groups = asyncio.run(load_mcp_tools_by_server(client))
            tools, sources, servers = [], {}, {}
            for server, server_tools in groups:
                kept = [
                    tool
                    for tool in server_tools
                    if selected is None or tool.name in selected
                ]
                tools.extend(kept)
                if server is not None:
                    sources.update({tool.name: server for tool in kept})
                card_tool = next(
                    (t for t in kept if t.name == MODEL_CARD_TOOL), None
                )
                if card_tool is not None:
                    servers[server] = (
                        card_tool,
                        [t for t in kept if t.name != MODEL_CARD_TOOL],
                    )
            self.add_tool(tools)
            return sources, servers

        sources, servers = await asyncio.to_thread(discover_and_apply)
        self._tool_sources.update(sources)
        self._card_servers.update(servers)
        self._discovered = False
        return sources

    # Inputs and outputs ------------------------------------------------------

    def _normalize_inputs(self, inputs):
        if isinstance(inputs, str):
            return self.format_query(inputs)
        state = dict(inputs)
        if "request" not in state:
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    state["request"] = message.text
                    break
        return state

    def format_query(
        self, prompt: str, state: ModelCardState | None = None
    ) -> ModelCardState:
        query: dict[str, Any] = {
            "request": prompt,
            "context": "",
            "intents": [],
            "card_sources": [],
            "model_refs": [],
            "ingested": [],
            "ingest_errors": [],
            "data_profiles": [],
            "assessments": [],
            "implied_targets": [],
            "run_plan": None,
            "run_result": "",
            "report": "",
            "messages": [HumanMessage(content=prompt)],
        }
        for key in ("requirements", "recommendations"):
            if state and state.get(key):
                query[key] = state[key]
        return query  # type: ignore[return-value]

    def format_result(self, result: ModelCardState) -> str:
        return result.get("report") or "No report was produced."

    # Graph -------------------------------------------------------------------

    def _build_graph(self):
        self.add_node(self._route, "route")
        self.add_node(self._ingest, "ingest")
        self.add_node(self._requirements, "requirements")
        self.add_node(self._recommend, "recommend")
        self.add_node(self._assess, "assess")
        self.add_node(self._plan_run, "plan_run")
        self.add_node(self._execute_run, "execute_run")
        self.add_node(self._report, "report")
        self.graph.add_edge(START, "route")
        self.graph.add_edge("route", "ingest")
        self.graph.add_conditional_edges(
            "ingest",
            self._wrap_cond(_after_ingest, "after_ingest", "model_cards"),
            {"requirements": "requirements", "report": "report"},
        )
        self.graph.add_conditional_edges(
            "requirements",
            self._wrap_cond(
                _after_requirements, "after_requirements", "model_cards"
            ),
            {
                "recommend": "recommend",
                "assess": "assess",
                "plan_run": "plan_run",
                "report": "report",
            },
        )
        self.graph.add_conditional_edges(
            "recommend",
            self._wrap_cond(_after_recommend, "after_recommend", "model_cards"),
            {"assess": "assess", "plan_run": "plan_run", "report": "report"},
        )
        self.graph.add_conditional_edges(
            "assess",
            self._wrap_cond(_after_assess, "after_assess", "model_cards"),
            {"plan_run": "plan_run", "report": "report"},
        )
        self.graph.add_edge("plan_run", "execute_run")
        self.graph.add_edge("execute_run", "report")
        self.graph.add_edge("report", END)

    # Nodes -------------------------------------------------------------------

    def _route(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        events = self.events(config)
        request, context = _split_request(state.get("request", ""))
        events.emit("Reading request", stage="route", preview=request[:200])
        sources = find_card_sources(request, self.workspace)
        intents = [
            name
            for name, pattern in _INTENT_PATTERNS.items()
            if pattern.search(request)
        ]
        refs = self._mentioned_models(request)
        llm_error = ""
        if self.llm_routing:
            routed = invoke_structured(
                self.llm,
                RouteDecision,
                [
                    SystemMessage(content=route_prompt),
                    HumanMessage(content=request),
                ],
                config=self.build_config(tags=["route"]),
                context="model card routing",
                fallback=RouteDecision(),
                repair=1,
                include_raw=True,
            )
            decision = routed.parsed
            # The fallback carries no raw reply. A bad API key or an
            # unreachable endpoint ends here, and the agent goes on with its
            # rules, so the answer must say that the LLM took no part.
            if routed.raw is None and routed.attempts:
                llm_error = routed.attempts[-1].error[:300]
                events.emit(
                    "The LLM call failed, so the agent reads this request "
                    f"with rules only: {llm_error}",
                    stage="route",
                )
            for ref in decision.card_sources:
                source = parse_source(ref, self.workspace)
                # The router may list a data file the request names ("I have
                # molecules.smi") as a card source. Reading it put a bogus
                # model in the catalog, which later answers then chose, so
                # the LLM's sources pass the same check as the request's.
                if (
                    source is not None
                    and is_card_source(source)
                    and source.label not in sources
                    and ref in request
                ):
                    sources.append(source.label)
            for name in decision.models:
                card = self.catalog.get(name)
                if card is not None and card.id not in refs:
                    refs.append(card.id)
            intents = list(dict.fromkeys([*decision.intents, *intents]))
        if sources and "ingest" not in intents:
            intents.insert(0, "ingest")
        if "run" not in intents and _orders_run(request, strong=True):
            intents.append("run")  # "Predict the boiling point of ..."
        if "run" in intents and not _orders_run(request):
            # Mentioning a run is not asking for one; this also overrides an
            # LLM router that read "can I just run inference?" as a run.
            intents.remove("run")
        if (
            "run" not in intents
            and "assess" not in intents
            and _FEASIBILITY.search(request)
        ):
            intents.append("assess")
        if (
            "list" in intents
            and "recommend" in intents
            and not _CHOICE_WORDS.search(request)
        ):
            intents.remove(
                "recommend"
            )  # "Which models do I have?" is inventory
        if "run" in intents and not refs:
            previous = _previous_choice(
                request, state.get("recommendations") or [], context
            )
            if previous is not None:
                refs.append(previous)
            elif "recommend" not in intents:
                intents.append("recommend")
        if not intents:
            intents = ["describe"] if refs else ["recommend"]
        if intents == ["ingest"]:
            intents.append("describe")
        events.emit(
            "Planned work",
            stage="route_result",
            intents=intents,
            sources=sources,
            models=refs,
        )
        return {
            "request": request,
            "context": context,
            "intents": intents,
            "card_sources": sources,
            "model_refs": refs,
            "llm_error": llm_error,
        }

    def _mentioned_models(self, text: str) -> list[str]:
        found = []
        lowered = text.lower()
        cards = self.catalog.all()
        for card in cards:
            for name in (card.name, *card.aliases, card.id):
                if len(name) >= 3 and re.search(
                    rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", lowered
                ):
                    found.append(card.id)
                    break
        # "the QM9 model" names mist_models---mist_26p9M_kkgx0omx_qm9 by a
        # token that only that card's name has. Tokens with a digit are
        # identifiers; plain words such as "heat" are not.
        owners: dict[str, set[str]] = {}
        for card in cards:
            for name in (card.name, *card.aliases):
                for token in re.split(r"[-_ ./:]+", name.lower()):
                    if (
                        len(token) >= 3
                        and re.search(r"\d", token)
                        and re.search(r"[a-z]", token)
                    ):
                        owners.setdefault(token, set()).add(card.id)
        for token, ids in owners.items():
            if len(ids) == 1 and re.search(
                rf"(?<![\w-]){re.escape(token)}(?![\w-])", lowered
            ):
                found.extend(ids - set(found))
        return found

    def _ingest(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        events = self.events(config)
        self._seed_catalog(events)
        self._discover_mcp(events)
        ingested, errors = [], []
        for ref in state.get("card_sources", []):
            events.emit("Reading model card", stage="ingest", source=ref)
            try:
                card = self.read_card(ref, config=config)
            except Exception as exc:  # noqa: BLE001 - report every failure to the user
                errors.append(f"{ref}: {type(exc).__name__}: {exc}")
                events.emit(
                    "Model card could not be read",
                    stage="ingest_error",
                    source=ref,
                    error=str(exc),
                )
                continue
            ingested.append(card.id)
            completeness = (
                f", completeness {card.completeness.score:.0%}"
                if card.completeness
                else ""
            )
            events.emit(
                f"Read {card.name}",
                stage="ingest_result",
                model=card.id,
                preview=f"{len(card.inputs)} input(s), {len(card.outputs)} output(s){completeness}",
            )
        refs = list(
            dict.fromkeys([
                *state.get("model_refs", []),
                *self._mentioned_models(state.get("request", "")),
            ])
        )
        return {
            "ingested": ingested,
            "ingest_errors": errors,
            "model_refs": refs,
        }

    def read_card(
        self,
        ref: str,
        *,
        config: RunnableConfig | None = None,
        refine: bool = True,
    ) -> ModelCard:
        """Read, normalize, refine, and store the card at ``ref``."""
        doc = load_document(
            ref, workspace=self.workspace, timeout=self.fetch_timeout
        )
        return self._store(doc, config=config, refine=refine)

    def _store(
        self,
        doc,
        *,
        config: RunnableConfig | None = None,
        refine: bool = True,
    ) -> ModelCard:
        card = extract_model_card(doc, domains=self.domains, tasks=self.tasks)
        cached = self.catalog.find_source(card.source.uri)
        if cached is not None and cached.source.sha256 == card.source.sha256:
            return cached
        if refine and self.llm_extraction and doc.text.strip():
            card = refine_card(
                card,
                doc.text,
                self.llm,
                domains=self.domains,
                tasks=self.tasks,
                config=self.build_config(tags=["extract"])
                if config is not None
                else None,
            )
        return self.catalog.add(
            card, text=evidence_text(doc.text) if doc.text else None
        )

    def _seed_catalog(self, events) -> None:
        if self._seeded:
            return
        self._seeded = True
        for root in self.catalog_paths:
            path = Path(root).expanduser()
            if not path.is_absolute():
                path = self.workspace / path
            if not path.exists():
                events.emit(
                    "Catalog path not found", stage="discover", path=str(path)
                )
                continue
            entries = (
                [path]
                if path.is_file()
                else sorted(
                    p for p in path.iterdir() if not p.name.startswith(".")
                )
            )
            for entry in entries:
                if entry.is_file() and entry.suffix.lower() not in {
                    ".md",
                    ".yaml",
                    ".yml",
                    ".json",
                }:
                    continue
                try:
                    self.read_card(str(entry), refine=self.refine_discovered)
                except Exception as exc:  # noqa: BLE001 - one bad card must not stop seeding
                    events.emit(
                        "Skipped card",
                        stage="discover",
                        path=str(entry),
                        error=str(exc),
                    )

    def _discover_mcp(self, events) -> None:
        """Register models served over MCP, such as Nomad SciFM tools."""
        if self._discovered or not self.discover_mcp_models:
            return
        self._discovered = True
        self._discovery_hint = None
        self._discovery_warnings = []
        servers = dict(self._card_servers)
        card_tool = self.tools.get(MODEL_CARD_TOOL)
        if not servers and card_tool is not None:
            # Tools attached with add_tool instead of add_mcp_tools.
            server = self._tool_sources.get(MODEL_CARD_TOOL)
            servers[server] = (
                card_tool,
                [
                    tool
                    for name, tool in self.tools.items()
                    if name != MODEL_CARD_TOOL
                    and self._tool_sources.get(name) == server
                ],
            )
        candidates = []
        for server, (card_tool, server_tools) in servers.items():
            for tool in server_tools:
                if self._tool_is_async_only(tool):
                    candidates.append((tool.name, tool, server, card_tool))
        listed = {name for name, *_ in candidates}
        for name in self.mcp_model_tools:
            tool = self.tools.get(name)
            if (
                tool is not None
                and name not in listed
                and self._tool_is_async_only(tool)
            ):
                server = self._tool_sources.get(name)
                candidates.append((name, tool, server, None))
        if not candidates:
            self._discovery_hint = self._no_served_models_hint()
            events.emit(self._discovery_hint, stage="discover")
            return
        found = []
        asked: Counter[str | None] = Counter()
        failed: dict[str | None, list[str]] = {}
        for name, tool, server, card_tool in candidates:
            text = ""
            if card_tool is not None:
                asked[server] += 1
                try:
                    text = _text(
                        _run_async(card_tool.ainvoke({"tool_name": name}))
                    )
                except Exception as exc:  # noqa: BLE001 - not every tool on the server is a model
                    if name not in self.mcp_model_tools:
                        failed.setdefault(server, []).append(f"{name}: {exc}")
                        continue
            schema = (
                tool.args_schema if isinstance(tool.args_schema, dict) else {}
            )
            doc = mcp_tool_document(
                name=name,
                server=server,
                description=tool.description,
                input_schema=schema,
                card_text=text,
                nomad=card_tool is not None,
            )
            try:
                found.append(self._store(doc, refine=self.refine_discovered).id)
            except Exception as exc:  # noqa: BLE001
                events.emit(
                    "Skipped MCP model",
                    stage="discover",
                    tool=name,
                    error=str(exc),
                )
        for server, errors in failed.items():
            if len(errors) == asked[server]:
                where = f"server '{server}'" if server else "the MCP server"
                self._discovery_warnings.append(
                    f"`{MODEL_CARD_TOOL}` on {where} failed for all "
                    f"{len(errors)} tool(s), so none of them counts as a "
                    f"model. The first error was: {errors[0][:300]}"
                )
        events.emit(
            f"Found {len(found)} model(s) on MCP servers",
            stage="discover",
            models=found,
        )
        for warning in self._discovery_warnings:
            events.emit(warning, stage="discover")

    def _no_served_models_hint(self) -> str:
        """Why discovery found nothing to ask a model card for."""
        if not any(self._tool_is_async_only(t) for t in self.tools.values()):
            return (
                "No MCP server is connected, so no served models were found. "
                "Add your server under `mcp_servers` in the URSA config, or "
                "in Settings > MCP tools in the dashboard."
            )
        names = sorted({s for s in self._tool_sources.values() if s})
        where = (
            "Server " + ", ".join(f"'{n}'" for n in names)
            if names
            else "The connected MCP server"
        )
        return (
            f"{where} offers no `{MODEL_CARD_TOOL}` tool, so its model cards "
            "cannot be read. List its model tools under `mcp_model_tools` "
            "in the agent config, or add card files with `catalog_paths`."
        )

    def _requirements(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        events = self.events(config)
        events.emit("Reading requirements", stage="requirements")
        text = state.get("request", "")
        # The requirements come from the new request alone. The previous
        # answer, which a dashboard session sends along, names other files,
        # quantities, and domains; read into the requirements, they would
        # answer the last question again. It only resolves "the same file".
        req = parse_requirements(
            text,
            self.llm,
            domains=self.domains,
            tasks=self.tasks,
            workspace=self.workspace,
            config=self.build_config(tags=["requirements"]),
        )
        context = state.get("context") or ""
        if context and not req.data_paths and _SAME_DATA.search(text):
            earlier = heuristic_requirements(
                context[:4000], workspace=self.workspace
            )
            req.data_paths = earlier.data_paths[:1]
        profiles = []
        for name in req.data_paths:
            path = Path(name).expanduser()
            if not path.is_absolute():
                path = self.workspace / path
            profile = inspect_data(path)
            profiles.append(profile)
            events.emit(
                f"Inspected {profile.name}",
                stage="inspect_result",
                preview=profile.summary(),
            )
            _describe_data(req, name, profile)
        events.emit(
            "Requirements understood",
            stage="requirements_result",
            preview=render_requirements(req),
        )
        return {"requirements": req, "data_profiles": profiles}

    def _recommend(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        events = self.events(config)
        req = state["requirements"]
        cards = self.catalog.all()
        events.emit(f"Ranking {len(cards)} catalog model(s)", stage="recommend")
        recs = rank_models(req, cards, domains=self.domains, tasks=self.tasks)
        if self.hub_search and (
            not recs or recs[0].fit in {"weak", "incompatible"}
        ):
            added = self._search_hub(req, events)
            if added:
                cards = self.catalog.all()
                recs = rank_models(
                    req, cards, domains=self.domains, tasks=self.tasks
                )
        shortlist = recs[: max(self.top_k, 5)]
        if self.llm_review and shortlist:
            shortlist = self._review(req, shortlist)
        events.emit(
            "Ranked models",
            stage="recommend_result",
            preview="\n".join(
                f"{r.rank}. {r.name} ({r.fit}, {r.score:.2f})"
                for r in shortlist[: self.top_k]
            ),
        )
        return {"recommendations": shortlist}

    def _search_hub(self, req: UserRequirements, events) -> list[str]:
        terms = [
            self.domains.get(d).label if self.domains.get(d) else d
            for d in req.domains[:2]
        ]
        terms += [
            self.tasks.get(t).label if self.tasks.get(t) else t
            for t in req.tasks[:1]
        ]
        added: list[str] = []
        for term in dict.fromkeys(terms or req.keywords[:2]):
            try:
                hits = search_hub(term, limit=5, timeout=self.fetch_timeout)
            except Exception as exc:  # noqa: BLE001 - network search is best-effort
                events.emit(
                    "Hub search failed",
                    stage="recommend",
                    query=term,
                    error=str(exc),
                )
                continue
            for hit in hits[:3]:
                try:
                    added.append(self.read_card(f"hf://{hit['id']}").id)
                except Exception:  # noqa: BLE001
                    continue
        events.emit(
            f"Added {len(added)} model(s) from Hugging Face search",
            stage="recommend",
            models=added,
        )
        return added

    def _review(
        self, req: UserRequirements, recs: list[Recommendation]
    ) -> list[Recommendation]:
        candidates = []
        for rec in recs:
            card = self.catalog.get(rec.model_id)
            if card is not None:
                candidates.append({
                    "model_id": rec.model_id,
                    "score": rec.score,
                    "blockers": rec.blockers,
                    "card": card.brief(),
                })
        if not candidates:
            return recs
        review = invoke_structured(
            self.llm,
            RankingReview,
            [
                SystemMessage(content=review_prompt),
                HumanMessage(
                    content=(
                        f"Requirements: {req.model_dump_json(exclude_defaults=True)}\n\n"
                        f"Candidates:\n{json.dumps(candidates, indent=1, default=str)[:30000]}"
                    )
                ),
            ],
            config=self.build_config(tags=["review"]),
            context="model recommendation review",
            fallback=RankingReview(),
            repair=1,
        )
        known = {rec.model_id for rec in recs}
        assessments = {
            item.model_id: (item.fit, item.rationale, item.caveats)
            for item in review.assessments
            if item.model_id in known
        }
        return blend_review(recs, assessments) if assessments else recs

    def _assess(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        """Check the user's data against the catalog and pick a pathway.

        Every catalog card is checked; the checks are cheap and a model the
        ranking missed may be the only one that fits. Named models are
        checked alone. When the user names no quantity but the data has a
        target column, that column is the quantity to predict.
        """
        events = self.events(config)
        profiles = state.get("data_profiles") or []
        profile = profiles[0] if profiles else None
        refs = state.get("model_refs") or []
        ranked = [rec.model_id for rec in state.get("recommendations", [])]
        if refs:
            cards = [self.catalog.get(ref) for ref in refs]
        else:
            order = {model_id: i for i, model_id in enumerate(ranked)}
            cards = sorted(
                self.catalog.all(), key=lambda c: order.get(c.id, len(order))
            )
        cards = [card for card in cards if card is not None]
        req = state.get("requirements")
        if profile is None and not refs and not (req and req.outputs):
            # Without data, a model, or a wanted quantity, every card would
            # read "not enough information"; the report asks for them instead.
            return {"assessments": [], "implied_targets": []}
        implied: list[str] = []
        if req is not None and not req.outputs and profile and profile.labels:
            implied = list(profile.labels[:3])
            req = req.model_copy(
                update={"outputs": [DataDescription(name=n) for n in implied]}
            )
        events.emit(
            f"Checking {len(cards)} model(s) against the data",
            stage="assess",
            models=[card.id for card in cards],
        )
        assessments = assess_models(
            cards,
            profile,
            req,
            texts={card.id: self.catalog.text(card.id) for card in cards},
            domains=self.domains,
        )
        events.emit(
            "Pathways decided",
            stage="assess_result",
            preview="\n".join(f"{a.name}: {a.label}" for a in assessments[:8]),
        )
        return {"assessments": assessments, "implied_targets": implied}

    def _plan_run(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        events = self.events(config)
        card = None
        for ref in state.get("model_refs", []):
            card = self.catalog.get(ref)
            if card is not None:
                break
        assessments = state.get("assessments") or []
        if card is None and assessments:
            # Run only a model whose card says it can take the data and give
            # what the user wants; never the best of several bad fits.
            usable = self._best_for(
                state.get("request", ""),
                [
                    a
                    for a in assessments
                    if a.pathway in {"inference", "inference_after_conversion"}
                ],
            )
            if usable is None:
                events.emit("No model can run on the data", stage="plan_run")
                return {
                    "run_plan": None,
                    "run_result": (
                        "Not run: no catalog model can run inference on this data as it is "
                        "(see the pathways above)."
                    ),
                }
            card = self.catalog.get(usable.model_id)
        if card is None:
            for rec in state.get("recommendations", []):
                if rec.fit != "incompatible":
                    card = self.catalog.get(rec.model_id)
                    break
        if card is None:
            events.emit("No runnable model selected", stage="plan_run")
            return {
                "run_plan": None,
                "run_result": "No compatible model was selected, so nothing was run.",
            }
        events.emit(
            f"Planning a run of {card.name}", stage="plan_run", model=card.id
        )
        plan = build_run_plan(
            card,
            state.get("request", ""),
            state.get("requirements"),
            llm=self.llm,
            config=self.build_config(tags=["plan_run"]),
            profiles=state.get("data_profiles") or [],
            max_items=self.max_run_items,
        )
        named = next((a for a in assessments if a.model_id == card.id), None)
        if named is not None and named.pathway not in {
            "inference",
            "inference_after_conversion",
        }:
            plan.warnings.insert(0, f"{named.label}: {named.summary}")
        return {"run_plan": plan}

    def _best_for(
        self, request: str, assessments: list[Assessment]
    ) -> Assessment | None:
        """The usable model the request points to ("the boiling point model").

        Direct inference beats inference after conversion. Within a pathway,
        the model whose outputs share the most words with the request wins;
        ties keep the assessment order.
        """
        words = set(tokenize(request))

        def overlap(assessment: Assessment) -> int:
            card = self.catalog.get(assessment.model_id)
            if card is None:
                return 0
            offered = " ".join(
                [s.name for s in card.outputs]
                + [c.name for s in card.outputs for c in s.channels]
            )
            return len(words & set(tokenize(offered)))

        ranked = sorted(
            enumerate(assessments),
            key=lambda item: (
                item[1].pathway != "inference",
                -overlap(item[1]),
                item[0],
            ),
        )
        return ranked[0][1] if ranked else None

    def _execute_run(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        events = self.events(config)
        plan = state.get("run_plan")
        if plan is None:
            return {}
        if not self.allow_run:
            return {
                "run_result": "Running models is disabled for this agent (allow_run=False); the plan is shown above."
            }
        if plan.mode == "unavailable":
            return {
                "run_result": "The card does not document how to load or call the model."
            }
        if plan.mode == "mcp_tool":
            if plan.missing_arguments or plan.argument_errors:
                needs = ", ".join(plan.missing_arguments) or "; ".join(
                    plan.argument_errors
                )
                return {
                    "run_result": f"Not run: the tool needs valid values for {needs}. Provide them and ask again."
                }
            tool = self.tools.get(plan.interface.name)
            if tool is None:
                return {
                    "run_result": f"MCP tool '{plan.interface.name}' is not connected; check the MCP server configuration."
                }
            if plan.batch:
                return self._execute_batch(plan, tool, events)
            events.emit(
                f"Calling {plan.interface.name}",
                stage="run",
                model=plan.model_id,
                arguments=plan.arguments,
            )
            try:
                result = _text(_run_async(tool.ainvoke(plan.arguments)))
            except Exception as exc:  # noqa: BLE001 - surface tool errors to the user
                events.emit(
                    "Model call failed",
                    stage="run_error",
                    model=plan.model_id,
                    error=str(exc),
                )
                return {"run_result": f"The model call failed: {exc}"}
            path = save_tool_result(result, plan, self.workspace)
            events.emit(
                "Model call finished",
                stage="run_result",
                model=plan.model_id,
                output_path=str(path),
                artifacts=[file_artifact(path, title="Model output")],
            )
            return {
                "run_result": f"Saved to `{path.relative_to(self.workspace)}`.\n\n```\n{result[:3000]}\n```"
            }
        card = self.catalog.get(plan.model_id)
        brief = execution_brief(card, plan, state.get("request", ""))
        events.emit(
            "Handing the run to the ExecutionAgent",
            stage="run",
            model=plan.model_id,
            preview=brief[:400],
        )
        executor = self._executor()
        run_config: dict[str, Any] = {
            "configurable": {
                "thread_id": f"{self.thread_id}-fm-run",
                "model_catalog_dir": str(self.catalog_dir),
            }
        }
        if config and config.get("callbacks"):
            run_config["callbacks"] = config["callbacks"]
        result = executor.invoke(
            {"messages": [HumanMessage(content=brief)]}, config=run_config
        )
        summary = executor.format_result(result)
        events.emit(
            "Run finished",
            stage="run_result",
            model=plan.model_id,
            preview=summary[:400],
        )
        return {"run_result": summary}

    def _execute_batch(self, plan: RunPlan, tool, events) -> dict:
        """Call the tool once per item of the user's file, one call at a time."""
        source = Path(plan.batch_source or "").name
        events.emit(
            f"Calling {plan.interface.name} on {len(plan.batch)} item(s) of {source}",
            stage="run",
            model=plan.model_id,
        )

        async def call_all() -> list[dict[str, Any]]:
            rows = []
            for arguments, name in zip(plan.batch, plan.batch_names):
                row: dict[str, Any] = {"arguments": arguments, "name": name}
                try:
                    text = _text(await tool.ainvoke(arguments))
                    try:
                        row["result"] = json.loads(text)
                    except ValueError:
                        row["result"] = text
                except Exception as exc:  # noqa: BLE001 - one failed item must not stop the batch
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
            return rows

        rows = _run_async(call_all())
        json_path, csv_path = save_batch_results(rows, plan, self.workspace)
        failed = sum(1 for row in rows if row.get("error"))
        events.emit(
            "Model calls finished",
            stage="run_result",
            model=plan.model_id,
            output_path=str(csv_path),
            artifacts=[file_artifact(csv_path, title="Model outputs")],
        )
        table = render_batch_table(rows)
        summary = (
            f"Ran {len(rows) - failed} of {len(rows)} item(s) from `{source}`"
            + (f"; {failed} failed" if failed else "")
            + f". Saved to `{csv_path.relative_to(self.workspace)}` and "
            f"`{json_path.relative_to(self.workspace)}`."
        )
        return {"run_result": f"{summary}\n\n{table}"}

    def _executor(self):
        if self._execution_agent is None:
            from ursa.agents.execution_agent import ExecutionAgent
            from ursa.model_cards.tools import model_card_tools

            mcp_tools = [
                tool
                for tool in self.tools.values()
                if self._tool_is_async_only(tool)
            ]
            kwargs = {"use_web": self.use_web, **self.executor_kwargs}
            self._execution_agent = ExecutionAgent(
                llm=self._base_llm,
                workspace=self.workspace,
                group=self.group,
                thread_id=f"{self.thread_id}-fm-run",
                extra_tools=model_card_tools() + mcp_tools,
                **kwargs,
            )
        return self._execution_agent

    def _report(
        self, state: ModelCardState, config: RunnableConfig | None = None
    ) -> dict:
        events = self.events(config)
        intents = state.get("intents", [])
        sections: list[str] = []
        citations = Citations(self.workspace)
        ingested = [
            self.catalog.get(model_id) for model_id in state.get("ingested", [])
        ]
        ingested = [card for card in ingested if card is not None]
        if ingested:
            sections.append(
                "## Model cards read\n\n"
                + "\n\n".join(render_card(card) for card in ingested)
            )
        if state.get("ingest_errors"):
            sections.append(
                "## Cards that could not be read\n\n"
                + "\n".join(f"- {e}" for e in state["ingest_errors"])
            )
        if "describe" in intents:
            shown = {card.id for card in ingested}
            described = [
                self.catalog.get(ref)
                for ref in state.get("model_refs", [])
                if ref not in shown
            ]
            described = [card for card in described if card is not None]
            if described:
                sections.append(
                    "## Model details\n\n"
                    + "\n\n".join(render_card(card) for card in described)
                )
        profiles = state.get("data_profiles") or []
        if profiles:
            sections.append(
                "## Your data\n\n"
                + "\n\n".join(
                    render_data_profile(p, citations) for p in profiles
                )
            )
        assessments = state.get("assessments") or []
        if assessments:
            title = (
                "## Inference or fine-tuning?"
                if "assess" in intents
                else "## Check before the run"
                if "run" in intents and state.get("model_refs")
                else "## Your choices for this data"
            )
            cards = {
                card.id: card
                for card in (self.catalog.get(a.model_id) for a in assessments)
                if card is not None
            }
            req = state.get("requirements")
            profile = profiles[0] if profiles else None
            verdict = render_verdict(
                assessments,
                cards,
                citations,
                wanted=[o.name for o in req.outputs] if req else [],
                implied=state.get("implied_targets") or [],
                profile=profile,
            )
            sections.append(
                f"{title}\n\n"
                + (f"{verdict}\n\n" if verdict else "")
                + render_assessments(assessments, cards, citations, profile)
            )
        elif "assess" in intents:
            sections.append(
                "## Inference or fine-tuning?\n\nThe answer depends on your data and on "
                "the quantity you want. Name the data file (for example, "
                "`I have molecules.smi`) and the quantity, or the model you have in mind."
            )
        if "list" in intents:
            cards = self.catalog.all()
            req = state.get("requirements")
            request = state.get("request", "")
            filters = (
                catalog_filters(
                    req, request, domains=self.domains, tasks=self.tasks
                )
                if req
                else []
            )
            matches, maybes = (
                filter_catalog(
                    req, cards, request, domains=self.domains, tasks=self.tasks
                )
                if req
                else ([], [])
            )
            sections.append(
                "## Model catalog\n\n"
                + render_catalog_answer(
                    cards, matches, maybes, filters, citations
                )
            )
        problems = list(self._discovery_warnings)
        if self._discovery_hint and not len(self.catalog):
            problems.insert(0, self._discovery_hint)
        if problems:
            sections.append(
                "## Model discovery\n\n"
                + "\n".join(f"- {problem}" for problem in problems)
            )
        if state.get("llm_error"):
            sections.append(
                "## LLM\n\nThe LLM call failed, so the agent read this request "
                "with its rules only. Check the model settings and the API "
                f"key. The error was: {state['llm_error']}"
            )
        if state.get("requirements") is not None and set(intents) & {
            "recommend",
            "run",
            "assess",
        }:
            sections.append(
                "## Requirements understood\n\n"
                + render_requirements(state["requirements"])
            )
        if "recommend" in intents and not assessments:
            # With data, the pathway list above is the recommendation.
            sections.append(
                "## Recommendations\n\n"
                + render_recommendations(
                    state.get("recommendations", []), limit=self.top_k
                )
            )
        plan = state.get("run_plan")
        if "run" in intents:
            lines = ["## Run"]
            if plan is not None:
                via = {
                    "mcp_tool": "its MCP tool",
                    "execution_agent": "the ExecutionAgent",
                }.get(plan.mode, plan.mode.replace("_", " "))
                lines.append(f"Model: **{plan.model_name}** via {via}")
                if plan.batch:
                    lines.append(
                        f"Arguments: one call per item of `{Path(plan.batch_source or '').name}` "
                        f"({len(plan.batch)} calls), such as `{json.dumps(plan.batch[0])[:300]}`"
                    )
                elif plan.arguments:
                    lines.append(
                        f"Arguments: `{json.dumps(plan.arguments)[:500]}`"
                    )
                for warning in plan.warnings:
                    lines.append(f"- Warning: {warning}")
            lines.append(state.get("run_result") or "Not run.")
            sections.append("\n\n".join(lines))
        if not sections:
            sections.append(
                f"No model cards were read. The catalog holds {len(self.catalog)} model(s); "
                "give a Hugging Face id, URL, or file path to read a card, or describe your need."
            )
        sources = citations.render()
        if sources:
            sections.append(sources)
        report = "\n\n".join(sections)
        artifacts = []
        if self.write_reports:
            path = self._write_report(report, state)
            artifacts.append(file_artifact(path, title="Model card report"))
        events.emit(
            "Report ready",
            stage="report",
            **({"artifacts": artifacts} if artifacts else {}),
        )
        return {"report": report, "messages": [AIMessage(content=report)]}

    def _write_report(self, report: str, state: ModelCardState) -> Path:
        path = unique_path(self.workspace / "fm_reports", "report", ".md")
        path.write_text(report + "\n", encoding="utf-8")
        payload = {
            "request": state.get("request"),
            "intents": state.get("intents"),
            "ingested": state.get("ingested"),
            "requirements": state["requirements"].model_dump()
            if state.get("requirements")
            else None,
            "recommendations": [
                rec.model_dump() for rec in state.get("recommendations", [])
            ],
            "run_plan": state["run_plan"].model_dump()
            if state.get("run_plan")
            else None,
        }
        path.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return path

    def close(self) -> None:
        if self._execution_agent is not None:
            self._execution_agent.close()
        super().close()

    async def aclose(self) -> None:
        if self._execution_agent is not None:
            await self._execution_agent.aclose()
        await super().aclose()


_ORDINALS = {
    "top": 1,
    "best": 1,
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
}
_ORDINAL = re.compile(
    r"\b(top|best|first|1st|second|2nd|third|3rd)\b|#\s*(\d+)|"
    r"\b(?:number|option|rank)\s*(\d+)\b",
    re.I,
)
_ANAPHORA = re.compile(
    r"\b(?:that|this|the recommended|the same|the chosen)\s+(?:model|one|fm)\b",
    re.I,
)


# The dashboard sends a session as one transcript: earlier turns as
# "User[agent]: ..." and "Assistant[agent]: ...", then "User: <new message>".
_TRANSCRIPT_TURN = re.compile(
    r"(?:^|\n\n)(?P<role>User|Assistant|System)(?P<tag>\[[^\]\n]*\])?: "
)
# A ranked line of render_recommendations, such as
# "1. **toy-heat-1d** (`mcp:toy:toy-heat-1d`): strong fit, score 0.93".
_RANKED_LINE = re.compile(
    r"^\s*(?P<rank>\d+)\. \*\*.+?\*\* \(`(?P<id>[^`]+)`\): "
    r"(?P<fit>strong|moderate|weak|incompatible) fit",
    re.M,
)


def _split_request(raw: str) -> tuple[str, str]:
    """Return the newest request and the previous answer as context.

    The terminal interface prefixes another agent's answer ("The last agent
    output was: ... The user stated: ..."); the dashboard sends the whole
    session transcript. Only the newest user message is the request. Treating
    earlier turns as part of it re-read old cards and re-ran an old "run".
    """
    text = (raw or "").strip()
    match = _PREVIOUS_OUTPUT.match(text)
    if match:
        return match.group("request").strip(), match.group("context").strip()
    turns = list(_TRANSCRIPT_TURN.finditer(text))
    if not turns or turns[-1].group("role") != "User":
        return text, ""
    if turns[0].start() != 0 and not any(t.group("tag") for t in turns):
        return text, ""  # a request that merely contains "User: "
    context = ""
    for turn, following in zip(turns, turns[1:]):
        if turn.group("role") == "Assistant":
            context = text[turn.end() : following.start()].strip()
    return text[turns[-1].end() :].strip(), context


def _previous_choice(
    request: str, recs: list[Recommendation], context: str = ""
) -> str | None:
    """Resolve "run the top model" or "use #2" against earlier recommendations.

    A dashboard run starts a fresh agent, so when no recommendations are in
    state, the ranking in the previous report (the context) stands in.
    """
    ranked = sorted(
        (r.rank, r.model_id) for r in recs if r.fit != "incompatible"
    )
    if not ranked:
        ranked = sorted(
            (int(m.group("rank")), m.group("id"))
            for m in _RANKED_LINE.finditer(context)
            if m.group("fit") != "incompatible"
        )
    if not ranked:
        return None
    match = _ORDINAL.search(request)
    if match:
        rank = _ORDINALS.get((match.group(1) or "").lower()) or int(
            match.group(2) or match.group(3) or 0
        )
        return ranked[rank - 1][1] if 1 <= rank <= len(ranked) else None
    if _ANAPHORA.search(request):
        return ranked[0][1]
    return None


def _after_ingest(state: ModelCardState) -> str:
    intents = set(state.get("intents", []))
    return (
        "requirements"
        if intents & {"recommend", "assess", "run", "list"}
        else "report"
    )


def _after_requirements(state: ModelCardState) -> str:
    intents = set(state.get("intents", []))
    data = bool(state.get("data_profiles"))
    if "recommend" in intents or (
        "run" in intents and not state.get("model_refs") and not data
    ):
        return "recommend"
    if "assess" in intents or data:
        return "assess"
    return "plan_run" if "run" in intents else "report"


def _after_recommend(state: ModelCardState) -> str:
    intents = set(state.get("intents", []))
    if "assess" in intents or state.get("data_profiles"):
        return "assess"
    return "plan_run" if "run" in intents else "report"


def _after_assess(state: ModelCardState) -> str:
    return "plan_run" if "run" in state.get("intents", []) else "report"


def _describe_data(
    req: UserRequirements, name: str, profile: DataProfile
) -> None:
    """Replace the bare file entry in the requirements with what the file holds."""
    words = [profile.representation or "", profile.kind.replace("_", " ")]
    names = [v.name for v in profile.variables]
    description = DataDescription(
        name=" ".join(w for w in words if w) or profile.name,
        # The variable names help ranking match the card's inputs.
        description="; ".join(
            [profile.summary()]
            + ([f"variables: {', '.join(names[:40])}"] if names else [])
        ),
        kind=profile.kind if profile.kind != "other" else None,
        format=profile.format,
        shape=" x ".join(str(n) for n in profile.grid) or None,
        path=name,
    )
    req.inputs = [i for i in req.inputs if i.path != name] + [description]
