import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agents.agents import (
    REFUSAL,
    AnswerAgent,
    CheckerAgent,
    RetrieverAgent,
    RouterAgent,
    citations_from_text,
    is_refusal,
)
from app.core.config import Settings
from app.core.decorators import timed
from app.schemas.models import Citation, RetrievedChunk, RouterDecision, Verdict
from app.utils.progress import ProgressBus

logger = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]
StreamWriter = Callable[[dict[str, Any]], None]


class OrchestrationState(TypedDict):
    question: str
    request_id: str
    deadline: float
    steps: int
    decision: RouterDecision | None
    chunks: list[RetrievedChunk]
    retry_reason: str | None
    attempt: int
    answer: str
    verdict: Verdict | None
    final_token: str | None
    citations: list[Citation]


class StepLimitExceeded(Exception):
    pass


class Orchestrator:
    def __init__(
        self,
        router: RouterAgent,
        retriever: RetrieverAgent,
        answerer: AnswerAgent,
        checker: CheckerAgent,
        bus: ProgressBus,
        settings: Settings,
    ):
        self.router = router
        self.retriever = retriever
        self.answerer = answerer
        self.checker = checker
        self.bus = bus
        self.settings = settings
        self.graph = self._build_graph()

    @staticmethod
    def _emit(writer: StreamWriter, event: str, data: dict[str, Any]) -> None:
        writer({"event": event, "data": data})

    def _step(
        self,
        writer: StreamWriter,
        state: OrchestrationState,
        agent: str,
        *,
        count: bool = True,
        **data: Any,
    ) -> int:
        steps = state["steps"]
        if count:
            steps += 1
            if steps > self.settings.max_agent_steps:
                raise StepLimitExceeded(
                    f"more than {self.settings.max_agent_steps} agent steps"
                )
        payload = {"request_id": state["request_id"], "agent": agent, **data}
        self.bus.publish(payload)
        self._emit(writer, "agent_step", payload)
        return steps

    def _budget(self, state: OrchestrationState) -> float:
        remaining = state["deadline"] - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request deadline exceeded")
        return min(remaining, self.settings.agent_step_timeout)

    def _finish(self, writer: StreamWriter, state: OrchestrationState) -> None:
        final_answer = state["final_token"] or state["answer"]
        if state["final_token"] is not None:
            self._emit(writer, "token", {"content": state["final_token"]})
        self._emit(writer, "answer", {"content": final_answer})
        self._emit(
            writer,
            "citations",
            {"citations": [citation.model_dump() for citation in state["citations"]]},
        )
        self._emit(writer, "done", {"request_id": state["request_id"]})
        self.bus.publish(
            {
                "request_id": state["request_id"],
                "agent": "orchestrator",
                "status": "done",
            }
        )

    def _build_graph(self):
        graph = StateGraph(OrchestrationState)
        graph.add_node("route", self._route)
        graph.add_node("reply", self._reply)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate_answer", self._answer)
        graph.add_node("answer_success", self._answer_success)
        graph.add_node("checker", self._check)
        graph.add_node("verified", self._verified)
        graph.add_node("reject", self._reject)
        graph.add_node("refusal", self._refusal)
        graph.add_node("complete", self._complete)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            self._route_destination,
            {"reply": "reply", "retrieve": "retrieve"},
        )
        graph.add_edge("reply", "complete")
        graph.add_conditional_edges(
            "retrieve",
            self._retrieval_destination,
            {"refusal": "refusal", "answer": "generate_answer"},
        )
        graph.add_conditional_edges(
            "generate_answer",
            self._answer_destination,
            {
                "refusal": "refusal",
                "answer_success": "answer_success",
                "checker": "checker",
            },
        )
        graph.add_edge("answer_success", "complete")
        graph.add_conditional_edges(
            "checker",
            self._checker_destination,
            {"verified": "verified", "reject": "reject"},
        )
        graph.add_edge("verified", "complete")
        graph.add_conditional_edges(
            "reject",
            self._retry_destination,
            {"answer": "generate_answer", "refusal": "refusal"},
        )
        graph.add_edge("refusal", "complete")
        graph.add_edge("complete", END)
        return graph.compile()

    async def _route(self, state: OrchestrationState) -> dict[str, Any]:
        writer = get_stream_writer()
        steps = self._step(writer, state, "router", status="routing")
        decision = await asyncio.wait_for(
            self.router.route(state["question"]), self._budget(state)
        )
        self._step(
            writer,
            state,
            "router",
            count=False,
            status="routed",
            route=decision.route,
            reason=decision.reason,
        )
        return {"decision": decision, "steps": steps}

    async def _reply(self, state: OrchestrationState) -> dict[str, Any]:
        decision = state["decision"]
        if decision is None:
            raise RuntimeError("router did not return a decision")
        if decision.route == "small_talk":
            token = decision.reply or "Hello! Ask me a question about the documents you uploaded."
        else:
            token = REFUSAL
        return {"final_token": token}

    @staticmethod
    def _route_destination(state: OrchestrationState) -> str:
        decision = state["decision"]
        if decision is None:
            raise RuntimeError("router did not return a decision")
        return "retrieve" if decision.route == "document_qa" else "reply"

    async def _retrieve(self, state: OrchestrationState) -> dict[str, Any]:
        writer = get_stream_writer()
        steps = self._step(writer, state, "retriever", status="retrieving")
        chunks = await asyncio.wait_for(
            self.retriever.retrieve(state["question"]), self._budget(state)
        )
        self._step(
            writer,
            state,
            "retriever",
            count=False,
            status="retrieved",
            chunks=len(chunks),
        )
        return {"chunks": chunks, "steps": steps}

    @staticmethod
    def _retrieval_destination(state: OrchestrationState) -> str:
        return "answer" if state["chunks"] else "refusal"

    async def _answer(self, state: OrchestrationState) -> dict[str, Any]:
        writer = get_stream_writer()
        steps = self._step(
            writer, state, "answer", status="generating", attempt=state["attempt"]
        )
        parts: list[str] = []
        async with aclosing(
            self.answerer.stream(
                state["question"], state["chunks"], state["retry_reason"]
            )
        ) as stream:
            async for token in stream:
                if time.monotonic() > state["deadline"]:
                    raise TimeoutError("request deadline exceeded")
                parts.append(token)
                self._emit(writer, "token", {"content": token})
        return {"answer": "".join(parts).strip(), "steps": steps}

    def _answer_destination(self, state: OrchestrationState) -> str:
        if is_refusal(state["answer"]):
            return "refusal"
        if not self.settings.enable_checker:
            return "answer_success"
        return "checker"

    async def _answer_success(self, state: OrchestrationState) -> dict[str, Any]:
        return {
            "citations": citations_from_text(state["answer"], state["chunks"]),
        }

    async def _check(self, state: OrchestrationState) -> dict[str, Any]:
        writer = get_stream_writer()
        steps = self._step(
            writer, state, "checker", status="checking", attempt=state["attempt"]
        )
        try:
            verdict = await asyncio.wait_for(
                self.checker.verify(
                    state["question"], state["answer"], state["chunks"]
                ),
                self._budget(state),
            )
        except Exception as exc:
            logger.warning(
                "checker unavailable (%r); accepting answer unverified", exc
            )
            verdict = Verdict(
                supported=True,
                reason="checker unavailable",
                citations=citations_from_text(state["answer"], state["chunks"]),
            )
            self._step(
                writer,
                state,
                "checker",
                count=False,
                status="skipped",
                reason=verdict.reason,
            )
        else:
            self._step(
                writer,
                state,
                "checker",
                count=False,
                status="supported" if verdict.supported else "rejected",
                reason=verdict.reason,
            )
        return {"verdict": verdict, "steps": steps}

    @staticmethod
    def _checker_destination(state: OrchestrationState) -> str:
        verdict = state["verdict"]
        if verdict is None:
            raise RuntimeError("checker did not return a verdict")
        return "verified" if verdict.supported else "reject"

    async def _verified(self, state: OrchestrationState) -> dict[str, Any]:
        verdict = state["verdict"]
        if verdict is None:
            raise RuntimeError("checker did not return a verdict")
        return {"citations": verdict.citations}

    async def _reject(self, state: OrchestrationState) -> dict[str, Any]:
        verdict = state["verdict"]
        if verdict is None:
            raise RuntimeError("checker did not return a verdict")
        writer = get_stream_writer()
        self._emit(
            writer,
            "retry",
            {
                "reason": verdict.reason,
                "message": "Previous draft was not supported by the documents and was discarded.",
            },
        )
        return {"attempt": state["attempt"] + 1, "retry_reason": verdict.reason}

    def _retry_destination(self, state: OrchestrationState) -> str:
        return (
            "answer"
            if state["attempt"] <= self.settings.max_answer_retries + 1
            else "refusal"
        )

    async def _refusal(self, state: OrchestrationState) -> dict[str, Any]:
        return {"final_token": REFUSAL, "citations": []}

    async def _complete(self, state: OrchestrationState) -> dict[str, Any]:
        self._finish(get_stream_writer(), state)
        return {}

    @timed
    async def run(self, question: str) -> AsyncIterator[Event]:
        request_id = uuid4().hex[:8]
        initial_state: OrchestrationState = {
            "question": question,
            "request_id": request_id,
            "deadline": time.monotonic() + self.settings.request_timeout,
            "steps": 0,
            "decision": None,
            "chunks": [],
            "retry_reason": None,
            "attempt": 1,
            "answer": "",
            "verdict": None,
            "final_token": None,
            "citations": [],
        }

        try:
            async with aclosing(
                self.graph.astream(initial_state, stream_mode="custom")
            ) as events:
                async for item in events:
                    yield item["event"], item["data"]
        except Exception as exc:
            logger.exception("ask failed request_id=%s", request_id)
            if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                message = "The request timed out. Please try again."
            elif isinstance(exc, StepLimitExceeded):
                message = "The request needed too many steps and was stopped."
            else:
                message = "The request could not be completed."
            self.bus.publish(
                {"request_id": request_id, "agent": "orchestrator", "status": "error"}
            )
            yield "error", {"message": message, "request_id": request_id}
