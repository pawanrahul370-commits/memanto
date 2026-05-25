"""LangGraph customer-support agent backed by MemantoStore.

Graph shape:

    START -> recall_context -> respond -> extract_and_store -> END

* ``recall_context`` reads from the cross-thread store using the current
  user's namespace and the latest user message as the semantic query.
* ``respond`` calls the LLM with the recalled memories injected as system
  context. This is the part the demo recording shows working without any
  prior turn in the current thread.
* ``extract_and_store`` asks the LLM to extract atomic preferences/facts
  from the latest user message and writes each one to the store via
  ``store.aput`` so the next session can use them.

The graph is compiled with both a checkpointer (short-term thread state)
and the MemantoStore (long-term cross-thread memory). That split is the
whole point - see ``examples/langgraph-memanto/README.md`` for why.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Literal

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field, model_validator

from langgraph.store.base import BaseStore

from memanto.cli.client.sdk_client import SdkClient

from memanto_store import MemantoStore
from state import SupportState

logger = logging.getLogger(__name__)

MemoryKind = Literal[
    "fact",
    "preference",
    "goal",
    "decision",
    "observation",
    "instruction",
    "relationship",
    "context",
    "commitment",
]


class ExtractedMemory(BaseModel):
    """One atomic piece of memory extracted from a user message."""

    kind: MemoryKind = Field(
        default="fact",
        description="The semantic category of this memory. Use 'preference' for "
        "user likes/dislikes, 'fact' for objective claims, 'commitment' for "
        "promises, 'goal' for stated aims.",
    )
    title: str = Field(
        default="",
        description="Short title, under 80 characters. Leave blank to auto-derive.",
    )
    content: str = Field(description="Single atomic claim. Be concise and specific.")
    confidence: float = Field(
        default=0.9, ge=0.0, le=1.0, description="0.95+ for explicit statements."
    )
    tags: list[str] = Field(
        default_factory=list, description="Lowercase tags for categorization."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: dict) -> dict:
        """Accept 'value' as an alias for 'content' (some models prefer it)."""
        if isinstance(data, dict):
            if "content" not in data and "value" in data:
                data["content"] = data.pop("value")
            if not data.get("title") and data.get("content"):
                data["title"] = str(data["content"])[:80]
        return data


class ExtractedMemories(BaseModel):
    """The structured-output response from the extractor LLM."""

    memories: list[ExtractedMemory] = Field(
        default_factory=list,
        description="Zero or more atomic memories. Empty list if the message "
        "contains no durable facts (greetings, vague questions, small talk).",
    )


def _default_llm() -> ChatOpenAI:
    """Build the default LLM. Routes through OpenRouter (matches the CrewAI example).

    Requires ``OPENROUTER_API_KEY``. Override the model via ``LANGGRAPH_LLM``.
    """
    model = os.environ.get("LANGGRAPH_LLM", "openai/gpt-oss-120b:free")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
            "your OpenRouter key (https://openrouter.ai/keys - free tier available)."
        )
    return ChatOpenAI(
        model=model,
        temperature=0.2,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _user_id_from_config(config) -> str:
    if not config:
        return "anonymous"
    return config.get("configurable", {}).get("user_id", "anonymous")


def build_support_graph(
    client: SdkClient,
    agent_id: str,
    llm: ChatOpenAI | None = None,
):
    """Compile a customer-support graph with MemantoStore + InMemorySaver.

    Args:
        client: Active Memanto SdkClient (output of MemantoSetup.setup).
        agent_id: The Memanto agent ID this graph writes against.
        llm: Optional override for the underlying chat model.

    Returns:
        A compiled LangGraph. Invoke with ``config={"configurable": {
        "thread_id": "...", "user_id": "..."}}``.
    """
    chat = llm or _default_llm()
    # json_mode guarantees a valid JSON envelope even when the model is under load.
    extractor = chat.with_structured_output(ExtractedMemories, method="json_mode")
    # MemantoStore is created here and passed to compile(store=...) below.
    # LangGraph then injects it into any node that declares `*, store: BaseStore`.
    store = MemantoStore(client, agent_id)

    async def recall_context(
        state: SupportState, config, *, store: BaseStore
    ) -> dict:
        """Pull cross-thread memories for this user, scoped by user_id."""
        user_id = _user_id_from_config(config)
        last_user_msg = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        query = last_user_msg.content if last_user_msg else "*"

        memories = await store.asearch(
            (user_id, "memories"),
            query=str(query),
            limit=8,
        )

        if not memories:
            logger.info("recall_context: no prior memories for user=%s", user_id)
            return {}

        formatted = "\n".join(
            f"- [{m.value.get('kind', 'fact')}] "
            f"{m.value.get('title') or m.value.get('content', '')}: "
            f"{m.value.get('content', '')}"
            for m in memories
        )
        logger.info(
            "recall_context: loaded %d memories for user=%s", len(memories), user_id
        )
        context_msg = SystemMessage(
            content=(
                "Known facts about the current user (recalled from long-term memory, "
                "not from this thread's history):\n" + formatted
            )
        )
        return {"messages": [context_msg]}

    async def respond(state: SupportState, config) -> dict:
        """Generate the assistant's reply with the recalled context in scope."""
        system_prefix = SystemMessage(
            content=(
                "You are a helpful customer-support assistant. "
                "Long-term facts about the current user may appear as a system "
                "message below this one; respect those preferences. Be concise."
            )
        )
        reply = await chat.ainvoke([system_prefix] + state["messages"])
        return {"messages": [reply]}

    async def extract_and_store(
        state: SupportState, config, *, store: BaseStore
    ) -> dict:
        """Pull atomic preferences/facts from the latest user turn and persist them."""
        user_id = _user_id_from_config(config)

        last_user_msg = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if not last_user_msg:
            return {}

        try:
            extracted: ExtractedMemories = await extractor.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You are a memory extractor. From the user message, "
                            "extract every atomic durable fact, preference, or "
                            "commitment worth remembering long-term.\n"
                            "Return an EMPTY memories list for greetings, vague "
                            "questions, or small talk with nothing durable.\n"
                            "Valid kinds: fact, preference, goal, decision, "
                            "observation, instruction, relationship, context, "
                            "commitment."
                        )
                    ),
                    HumanMessage(content=str(last_user_msg.content)),
                ]
            )
        except openai.RateLimitError:
            raise  # let RetryPolicy handle backoff and retry
        except Exception as e:
            logger.warning("extract_and_store: extraction failed: %s", e)
            return {}

        for mem in extracted.memories:
            await store.aput(
                (user_id, "memories"),
                str(uuid.uuid4()),
                {
                    "kind": mem.kind,
                    "title": mem.title,
                    "content": mem.content,
                    "confidence": mem.confidence,
                    "tags": mem.tags,
                },
            )
        if extracted.memories:
            logger.info(
                "extract_and_store: persisted %d memories for user=%s",
                len(extracted.memories),
                user_id,
            )
        return {}

    # Retry LLM nodes on 429 rate-limit errors from OpenRouter's free tier.
    # initial_interval=32s is just above OpenRouter's "retry after 30s" window.
    _rate_limit_retry = RetryPolicy(
        initial_interval=32.0,
        backoff_factor=1.5,
        max_interval=120.0,
        max_attempts=5,
        retry_on=openai.RateLimitError,
    )

    builder = StateGraph(SupportState)
    builder.add_node("recall_context", recall_context)
    builder.add_node("respond", respond, retry_policy=_rate_limit_retry)
    builder.add_node("extract_and_store", extract_and_store, retry_policy=_rate_limit_retry)
    builder.add_edge(START, "recall_context")
    builder.add_edge("recall_context", "respond")
    builder.add_edge("respond", "extract_and_store")
    builder.add_edge("extract_and_store", END)

    return builder.compile(
        checkpointer=InMemorySaver(),
        store=store,
    )


def latest_assistant_text(messages: list) -> str:
    """Return the most recent AIMessage content as a plain string."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                ]
                return "".join(parts)
    return "(no assistant reply)"
