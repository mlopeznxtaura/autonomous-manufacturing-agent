"""
LangGraph autonomous operator assistant for the production floor.
Ingests telemetry, identifies issues, surfaces alerts, triggers escalations.
SDKs: LangGraph, LangChain, Ollama
"""
import time
import json
from typing import Optional, Dict, Any, List, TypedDict, Annotated
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


OPERATOR_SYSTEM_PROMPT = """You are an autonomous manufacturing floor operator assistant for {line_id}.

Your responsibilities:
1. Monitor production telemetry and quality inspection results
2. Identify anomalies, trends, and potential line stop conditions
3. Proactively alert the team to issues before they escalate
4. Recommend corrective actions with reasoning
5. Trigger escalation workflows when needed

Current production context:
{context}

Guidelines:
- Be specific and data-driven in your assessments
- Prioritize critical defects and machine faults
- Suggest root causes when patterns are clear
- Escalate immediately if: defect rate > 5%, machine temp > threshold, repeated pattern failures
- Keep responses concise but actionable
"""


class OperatorState(TypedDict):
    messages: List[Any]
    line_id: str
    telemetry_snapshot: Dict[str, Any]
    recent_inspections: List[Dict]
    alert_queue: List[Dict]
    should_escalate: bool
    escalation_reason: str
    action_taken: str


@dataclass
class ProductionContext:
    line_id: str
    defect_rate_pct: float
    parts_per_hour: int
    machine_temperatures: Dict[str, float]
    recent_alerts: List[str]
    active_faults: List[str]
    shift_duration_min: int

    def to_prompt_str(self) -> str:
        return (
            f"Line: {self.line_id} | "
            f"Defect rate: {self.defect_rate_pct:.1f}% | "
            f"Throughput: {self.parts_per_hour} parts/hr | "
            f"Active faults: {', '.join(self.active_faults) or 'none'} | "
            f"Shift duration: {self.shift_duration_min} min"
        )


class ManufacturingOperatorAgent:
    """
    Autonomous production floor operator using LangGraph.
    Analyzes telemetry, makes decisions, and takes actions.
    """

    def __init__(
        self,
        line_id: str = "LINE_A",
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
        escalation_callback=None,
        kafka_producer=None,
    ):
        self.line_id = line_id
        self.escalation_callback = escalation_callback
        self.kafka_producer = kafka_producer
        self.llm = ChatOllama(model=model, base_url=ollama_url, temperature=0.2)
        self._tools = self._build_tools()
        self._graph = self._build_graph()
        self._context = ProductionContext(
            line_id=line_id, defect_rate_pct=0.0,
            parts_per_hour=0, machine_temperatures={},
            recent_alerts=[], active_faults=[],
            shift_duration_min=0,
        )
        print(f"[OperatorAgent] Initialized for {line_id} | model={model}")

    def _build_tools(self):
        line_id = self.line_id

        @tool
        def get_defect_rate(time_window_minutes: int = 10) -> str:
            """Get the defect rate over the last N minutes."""
            return json.dumps({
                "line_id": line_id,
                "window_minutes": time_window_minutes,
                "defect_rate_pct": 2.3,
                "parts_inspected": 145,
                "defects_found": 3,
            })

        @tool
        def get_machine_status(machine_id: str) -> str:
            """Get the current status and readings of a machine."""
            return json.dumps({
                "machine_id": machine_id,
                "status": "running",
                "temperature_c": 78.5,
                "vibration_mm_s": 2.1,
                "cycle_time_s": 4.2,
                "uptime_pct": 98.7,
            })

        @tool
        def trigger_escalation(reason: str, severity: str = "high") -> str:
            """Trigger an escalation workflow to notify supervisors."""
            return json.dumps({
                "escalation_id": f"ESC_{int(time.time())}",
                "reason": reason,
                "severity": severity,
                "notified": ["supervisor", "quality_lead"],
                "triggered_at": time.time(),
            })

        @tool
        def adjust_inspection_threshold(new_confidence: float, reason: str) -> str:
            """Adjust YOLO inspection confidence threshold."""
            return json.dumps({
                "action": "threshold_adjusted",
                "new_confidence": new_confidence,
                "reason": reason,
                "applied_at": time.time(),
            })

        @tool
        def get_recent_defects(limit: int = 20) -> str:
            """Get the most recent defect records from the inspection database."""
            return json.dumps({
                "defects": [
                    {"part_id": f"part_{i:04d}", "type": "scratch", "severity": "minor", "timestamp": time.time() - i * 30}
                    for i in range(min(limit, 5))
                ]
            })

        return [get_defect_rate, get_machine_status, trigger_escalation,
                adjust_inspection_threshold, get_recent_defects]

    def _build_graph(self):
        tool_node = ToolNode(self._tools)
        llm_with_tools = self.llm.bind_tools(self._tools)

        def agent_node(state: OperatorState) -> OperatorState:
            context_str = self._context.to_prompt_str()
            system = OPERATOR_SYSTEM_PROMPT.format(
                line_id=state["line_id"], context=context_str
            )
            messages = [SystemMessage(content=system)] + state["messages"]
            response = llm_with_tools.invoke(messages)
            state["messages"].append(response)
            state["action_taken"] = response.content[:200] if response.content else ""
            return state

        def should_use_tools(state: OperatorState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        def check_escalation(state: OperatorState) -> OperatorState:
            last_content = state["messages"][-1].content if state["messages"] else ""
            escalate_keywords = ["escalate", "critical", "line stop", "immediate", "shutdown"]
            state["should_escalate"] = any(kw in last_content.lower() for kw in escalate_keywords)
            if state["should_escalate"]:
                state["escalation_reason"] = last_content[:300]
            return state

        graph = StateGraph(OperatorState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_node("check_escalation", check_escalation)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: "check_escalation"})
        graph.add_edge("tools", "agent")
        graph.add_edge("check_escalation", END)
        return graph.compile()

    def analyze(self, event: str, context: Optional[ProductionContext] = None) -> Dict[str, Any]:
        """Analyze a production event and return agent response + actions."""
        if context:
            self._context = context

        initial_state = OperatorState(
            messages=[HumanMessage(content=event)],
            line_id=self.line_id,
            telemetry_snapshot={},
            recent_inspections=[],
            alert_queue=[],
            should_escalate=False,
            escalation_reason="",
            action_taken="",
        )

        final_state = self._graph.invoke(initial_state)
        result = {
            "line_id": self.line_id,
            "event": event,
            "response": final_state["action_taken"],
            "should_escalate": final_state["should_escalate"],
            "escalation_reason": final_state["escalation_reason"],
            "messages_count": len(final_state["messages"]),
        }

        if final_state["should_escalate"] and self.escalation_callback:
            self.escalation_callback(final_state["escalation_reason"])

        if self.kafka_producer:
            self.kafka_producer.emit_agent_action(
                action_type="analysis",
                reasoning=final_state["action_taken"][:200],
                parameters={"event": event, "escalated": final_state["should_escalate"]},
            )

        return result

    def update_context(self, defect_rate: float, parts_per_hour: int,
                       temperatures: Dict[str, float], faults: List[str]):
        self._context.defect_rate_pct = defect_rate
        self._context.parts_per_hour = parts_per_hour
        self._context.machine_temperatures = temperatures
        self._context.active_faults = faults
        self._context.shift_duration_min += 1
