import os
import time
import json
from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun

# ✅ Official OpenAI via LangChain (no proxy/base_url)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ======================
# 0) Environment / Keys
# ======================
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env or shell.")

# LLM + Embeddings (LangChain, official OpenAI endpoint by default)
llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-small",
)


def llm_complete(prompt: str) -> str:
    """
    LlamaIndex-like helper:
    - LlamaIndex: Settings.llm.complete(prompt).text
    - LangChain: llm.invoke(prompt).content
    """
    msg = llm.invoke(prompt)
    return getattr(msg, "content", str(msg))


# ======================
# 1. Core Data Structure Definitions
# ======================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class ToolType(Enum):
    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    GENERATION = "generation"

@dataclass
class PerformanceMetrics:
    execution_time: float
    cost_estimate: float
    memory_usage: float
    success_rate: float

@dataclass
class ToolExecutionResult:
    tool_name: str
    status: TaskStatus
    result: Any
    performance: PerformanceMetrics
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    optimization_suggestions: List[str] = None
    retry_count: int = 0

@dataclass
class TaskNode:
    task_id: str
    tool_name: str
    tool_type: ToolType
    params: Dict[str, Any]
    dependencies: List[str]
    priority: int
    max_retries: int
    timeout: float
    is_critical: bool
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[ToolExecutionResult] = None

class SystemState(TypedDict):
    # User request
    user_question: str

    # Task management
    task_dag: List[TaskNode]
    execution_queue: List[str]
    completed_tasks: Dict[str, ToolExecutionResult]
    failed_tasks: Dict[str, ToolExecutionResult]

    # System state
    system_load: float
    available_tools: List[str]
    performance_history: List[PerformanceMetrics]

    # Business data
    stock_data: str
    news_data: str
    sentiment_analysis: str
    final_recommendation: str

    # Control flow
    should_continue: bool
    early_exit_triggered: bool
    current_phase: str
    error_count: int


# ======================
# 2. Tool Agent Implementation Layer
# ======================

class BaseToolAgent:
    def __init__(self, name: str, tool_type: ToolType):
        self.name = name
        self.tool_type = tool_type
        self.execution_history: List[ToolExecutionResult] = []

    def execute(self, params: Dict[str, Any]) -> ToolExecutionResult:
        start_time = time.time()
        try:
            result = self._execute_core(params)
            execution_time = time.time() - start_time

            performance = PerformanceMetrics(
                execution_time=execution_time,
                cost_estimate=self._estimate_cost(params),
                memory_usage=self._get_memory_usage(),
                success_rate=self._calculate_success_rate(),
            )

            out = ToolExecutionResult(
                tool_name=self.name,
                status=TaskStatus.COMPLETED,
                result=result,
                performance=performance,
                optimization_suggestions=self._generate_optimization_suggestions(performance),
            )
            self.execution_history.append(out)
            return out

        except Exception as e:
            execution_time = time.time() - start_time
            out = ToolExecutionResult(
                tool_name=self.name,
                status=TaskStatus.FAILED,
                result=None,
                performance=PerformanceMetrics(execution_time, 0, 0, 0),
                error_code="EXECUTION_ERROR",
                error_message=str(e),
            )
            self.execution_history.append(out)
            return out

    def _execute_core(self, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def _estimate_cost(self, params: Dict[str, Any]) -> float:
        return 0.01  # Base cost estimate

    def _get_memory_usage(self) -> float:
        return 10.0  # MB

    def _calculate_success_rate(self) -> float:
        if not self.execution_history:
            return 1.0
        successful = sum(1 for r in self.execution_history if r.status == TaskStatus.COMPLETED)
        return successful / len(self.execution_history)

    def _generate_optimization_suggestions(self, performance: PerformanceMetrics) -> List[str]:
        suggestions = []
        if performance.execution_time > 5.0:
            suggestions.append("Consider adding caching to reduce execution time")
        if performance.memory_usage > 100.0:
            suggestions.append("Optimize memory usage; consider batch processing")
        return suggestions


class StockPriceAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("stock_price_agent", ToolType.DATA_RETRIEVAL)

    def _execute_core(self, params: Dict[str, Any]) -> str:
        ticker = params.get("ticker", "UNKNOWN")
        time.sleep(0.5)  # simulate latency
        return (
            f"{ticker} average price over the past 7 days is $245.6, "
            f"up 5.3%. Market cap is approximately $780B."
        )


class NewsAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("news_agent", ToolType.DATA_RETRIEVAL)

    def _execute_core(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        try:
            search = DuckDuckGoSearchRun()
            result = search.invoke(f"{query} latest news")
            return result[:1000]
        except Exception:
            return (
                "Tesla’s latest earnings report shows Q3 revenue of $23.4B, "
                "up 7.8% year-over-year. However, competition in the Chinese market "
                "is intensifying, leading to higher stock volatility. Analysts remain "
                "cautiously optimistic about progress in autonomous driving."
            )


class SentimentAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("sentiment_agent", ToolType.ANALYSIS)

    def _execute_core(self, params: Dict[str, Any]) -> str:
        text = params.get("text", "")
        if not text.strip():
            return "Neutral — no valid text available for analysis"

        prompt = f"""
Please analyze the sentiment of the following text and provide a detailed score:

Text: {text}

Output format:
- Overall sentiment: Positive / Negative / Neutral
- Sentiment intensity: 1–10
- Keywords: list key sentiment-driving terms
- Risk notes: if sentiment is negative, describe the main risks
"""
        return llm_complete(prompt)


# ======================
# 3. Top Agent: Core Orchestration Layer
# ======================

class TopAgent:
    def __init__(self):
        self.tool_agents = {
            "stock_price": StockPriceAgent(),
            "news": NewsAgent(),
            "sentiment": SentimentAgent(),
        }
        self.monitoring_dashboard = MonitoringDashboard()

    def plan_tasks(self, user_question: str) -> List[TaskNode]:
        planning_prompt = f"""
As the Top Agent of an investment analysis system, create a detailed execution plan
for the following user question:

Question: {user_question}

Available tools:
1. stock_price – retrieve stock price data (high priority, critical)
2. news – retrieve related news (medium priority, degradable)
3. sentiment – sentiment analysis (low priority, depends on news)

Return the task DAG in JSON format:
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "tool_name": "stock_price",
      "tool_type": "data_retrieval",
      "params": {{"ticker": "TSLA"}},
      "dependencies": [],
      "priority": 1,
      "max_retries": 3,
      "timeout": 10.0,
      "is_critical": true
    }},
    {{
      "task_id": "task_2",
      "tool_name": "news",
      "tool_type": "data_retrieval",
      "params": {{"query": "Tesla TSLA"}},
      "dependencies": [],
      "priority": 2,
      "max_retries": 2,
      "timeout": 15.0,
      "is_critical": false
    }},
    {{
      "task_id": "task_3",
      "tool_name": "sentiment",
      "tool_type": "analysis",
      "params": {{"text": "{{news_data}}"}},
      "dependencies": ["task_2"],
      "priority": 3,
      "max_retries": 2,
      "timeout": 8.0,
      "is_critical": false
    }}
  ]
}}
"""
        plan_text = llm_complete(planning_prompt)
        try:
            plan_data = json.loads(plan_text)
            tasks: List[TaskNode] = []
            for task_data in plan_data["tasks"]:
                tasks.append(
                    TaskNode(
                        task_id=task_data["task_id"],
                        tool_name=task_data["tool_name"],
                        tool_type=ToolType(task_data["tool_type"]),
                        params=task_data["params"],
                        dependencies=task_data["dependencies"],
                        priority=task_data["priority"],
                        max_retries=task_data["max_retries"],
                        timeout=task_data["timeout"],
                        is_critical=task_data["is_critical"],
                    )
                )
            return tasks
        except Exception:
            return self._get_default_plan()

    def _get_default_plan(self) -> List[TaskNode]:
        return [
            TaskNode("task_1", "stock_price", ToolType.DATA_RETRIEVAL,
                     {"ticker": "TSLA"}, [], 1, 3, 10.0, True),
            TaskNode("task_2", "news", ToolType.DATA_RETRIEVAL,
                     {"query": "Tesla TSLA"}, [], 2, 2, 15.0, False),
            TaskNode("task_3", "sentiment", ToolType.ANALYSIS,
                     {"text": "{{news_data}}"}, ["task_2"], 3, 2, 8.0, False),
        ]

    def dynamic_dispatch(self, state: SystemState) -> Optional[TaskNode]:
        ready_tasks: List[TaskNode] = []
        for task in state["task_dag"]:
            if task.status == TaskStatus.PENDING:
                deps_satisfied = all(dep_id in state["completed_tasks"] for dep_id in task.dependencies)
                if deps_satisfied:
                    ready_tasks.append(task)

        if not ready_tasks:
            return None

        ready_tasks.sort(key=lambda t: (t.priority, -t.timeout))

        if state["system_load"] > 0.8:
            critical_tasks = [t for t in ready_tasks if t.is_critical]
            if critical_tasks:
                return critical_tasks[0]

        return ready_tasks[0]

    def execute_task(self, task: TaskNode, state: SystemState) -> ToolExecutionResult:
        processed_params = self._process_task_params(task.params, state)

        agent = self.tool_agents.get(task.tool_name)
        if not agent:
            return ToolExecutionResult(
                tool_name=task.tool_name,
                status=TaskStatus.FAILED,
                result=None,
                performance=PerformanceMetrics(0, 0, 0, 0),
                error_code="AGENT_NOT_FOUND",
                error_message=f"Tool agent {task.tool_name} not found",
            )

        task.status = TaskStatus.RUNNING
        result = agent.execute(processed_params)

        self.monitoring_dashboard.update_task_status(task.task_id, result)
        return result

    def _process_task_params(self, params: Dict[str, Any], state: SystemState) -> Dict[str, Any]:
        processed: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                dep_key = value[2:-2]
                if dep_key == "news_data":
                    processed[key] = state.get("news_data", "")
                elif dep_key == "stock_data":
                    processed[key] = state.get("stock_data", "")
                else:
                    processed[key] = ""
            else:
                processed[key] = value

        print(f"🔧 Processed task params: {processed}")
        return processed

    def should_retry_task(self, task: TaskNode, result: ToolExecutionResult) -> bool:
        if result.status != TaskStatus.FAILED:
            return False
        if result.retry_count >= task.max_retries:
            return False
        if not task.is_critical and result.retry_count > 1:
            return False
        return True

    def check_early_exit(self, state: SystemState) -> bool:
        critical_failed = any(task.is_critical and task.status == TaskStatus.FAILED for task in state["task_dag"])
        if critical_failed:
            return True
        if state["error_count"] > 3:
            return True
        return False

    def generate_final_recommendation(self, state: SystemState) -> str:
        prompt = f"""
Based on the following analysis results, provide a professional investment recommendation:

Stock data: {state.get('stock_data', 'Unavailable')}
News analysis: {state.get('news_data', 'Unavailable')[:200]}...
Sentiment analysis: {state.get('sentiment_analysis', 'Unavailable')}

System execution status:
- Completed tasks: {len(state['completed_tasks'])}
- Failed tasks: {len(state['failed_tasks'])}
- Data completeness: {"High" if len(state['completed_tasks']) >= 2 else "Medium" if len(state['completed_tasks']) >= 1 else "Low"}

Please provide:
1. Clear recommendation (Buy / Hold / Sell)
2. Risk assessment (High / Medium / Low)
3. Key reasoning (3–5 points)
4. Data reliability assessment
"""
        return llm_complete(prompt)


class MonitoringDashboard:
    def __init__(self):
        self.task_metrics: Dict[str, ToolExecutionResult] = {}
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "total_cost": 0.0,
        }

    def update_task_status(self, task_id: str, result: ToolExecutionResult):
        self.task_metrics[task_id] = result
        self.system_metrics["total_tasks"] += 1

        if result.status == TaskStatus.COMPLETED:
            self.system_metrics["completed_tasks"] += 1
        elif result.status == TaskStatus.FAILED:
            self.system_metrics["failed_tasks"] += 1

        self.system_metrics["total_cost"] += result.performance.cost_estimate

        total_time = sum(r.performance.execution_time for r in self.task_metrics.values())
        self.system_metrics["average_execution_time"] = total_time / max(1, len(self.task_metrics))

    def get_dashboard_summary(self) -> Dict[str, Any]:
        return {
            "system_metrics": self.system_metrics,
            "task_details": {tid: asdict(result) for tid, result in self.task_metrics.items()},
            "success_rate": self.system_metrics["completed_tasks"] / max(1, self.system_metrics["total_tasks"]),
        }


# ======================
# 4. Workflow Orchestration
# ======================

def initialize_system(state: SystemState):
    top_agent = TopAgent()
    task_dag = top_agent.plan_tasks(state["user_question"])

    return {
        "task_dag": task_dag,
        "execution_queue": [task.task_id for task in task_dag],
        "completed_tasks": {},
        "failed_tasks": {},
        "system_load": 0.3,
        "available_tools": ["stock_price", "news", "sentiment"],
        "performance_history": [],
        "should_continue": True,
        "early_exit_triggered": False,
        "current_phase": "planning_complete",
        "error_count": 0,
    }

def execute_next_task(state: SystemState):
    top_agent = TopAgent()

    if top_agent.check_early_exit(state):
        return {"early_exit_triggered": True, "should_continue": False, "current_phase": "early_exit"}

    next_task = top_agent.dynamic_dispatch(state)
    if not next_task:
        return {"should_continue": False, "current_phase": "all_tasks_completed"}

    result = top_agent.execute_task(next_task, state)
    next_task.status = result.status
    next_task.result = result

    updates: Dict[str, Any] = {"error_count": state["error_count"]}

    if result.status == TaskStatus.COMPLETED:
        updates["completed_tasks"] = {**state["completed_tasks"], next_task.task_id: result}
        if next_task.tool_name == "stock_price":
            updates["stock_data"] = result.result
        elif next_task.tool_name == "news":
            updates["news_data"] = result.result
        elif next_task.tool_name == "sentiment":
            updates["sentiment_analysis"] = result.result

    elif result.status == TaskStatus.FAILED:
        if top_agent.should_retry_task(next_task, result):
            next_task.status = TaskStatus.RETRYING
            result.retry_count += 1
            updates["error_count"] = state["error_count"] + 1
        else:
            updates["failed_tasks"] = {**state["failed_tasks"], next_task.task_id: result}
            updates["error_count"] = state["error_count"] + 1
            if not next_task.is_critical:
                next_task.status = TaskStatus.SKIPPED

    pending_tasks = [t for t in state["task_dag"] if t.status in [TaskStatus.PENDING, TaskStatus.RETRYING]]
    updates["should_continue"] = len(pending_tasks) > 0 and not updates.get("early_exit_triggered", False)

    return updates

def generate_final_report(state: SystemState):
    top_agent = TopAgent()
    recommendation = top_agent.generate_final_recommendation(state)
    dashboard_summary = top_agent.monitoring_dashboard.get_dashboard_summary()
    return {"final_recommendation": recommendation, "current_phase": "completed", "dashboard_summary": dashboard_summary}

def should_continue_execution(state: SystemState) -> str:
    if state.get("early_exit_triggered", False):
        return "generate_report"
    elif state.get("should_continue", True):
        return "execute_task"
    else:
        return "generate_report"


# ======================
# 5. Build workflow
# ======================
workflow = StateGraph(SystemState)
workflow.add_node("initialize", initialize_system)
workflow.add_node("execute_task", execute_next_task)
workflow.add_node("generate_report", generate_final_report)

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "execute_task")
workflow.add_conditional_edges(
    "execute_task",
    should_continue_execution,
    {"execute_task": "execute_task", "generate_report": "generate_report"},
)
workflow.add_edge("generate_report", END)

app = workflow.compile()


# ======================
# 6. Main
# ======================
if __name__ == "__main__":
    initial_state: SystemState = {
        "user_question": "How has Tesla (TSLA) performed recently? Is it worth buying?",
        "task_dag": [],
        "execution_queue": [],
        "completed_tasks": {},
        "failed_tasks": {},
        "system_load": 0.0,
        "available_tools": [],
        "performance_history": [],
        "stock_data": "",
        "news_data": "",
        "sentiment_analysis": "",
        "final_recommendation": "",
        "should_continue": True,
        "early_exit_triggered": False,
        "current_phase": "initialized",
        "error_count": 0,
    }

    try:
        print("Starting the enhanced multi-agent investment analysis system...")
        print("=" * 60)

        result = app.invoke(initial_state)

        print("\n=== Investment Analysis Report ===")
        print(f"Question: {result['user_question']}")
        print(f"Phase: {result['current_phase']}")

        print("\nData collected:")
        print(f"Stock data: {result.get('stock_data', 'Not available')}")
        print(
            f"News data: {result.get('news_data', 'Not available')[:100]}..."
            if result.get("news_data")
            else "News data: Not available"
        )
        print(f"Sentiment analysis: {result.get('sentiment_analysis', 'Not completed')}")

        print("\nFinal recommendation:")
        print(result.get("final_recommendation", "Unable to generate recommendation"))

        print("\nSystem execution summary:")
        dashboard = result.get("dashboard_summary", {})
        if dashboard:
            metrics = dashboard.get("system_metrics", {})
            print(f"- Total tasks: {metrics.get('total_tasks', 0)}")
            print(f"- Completed tasks: {metrics.get('completed_tasks', 0)}")
            print(f"- Failed tasks: {metrics.get('failed_tasks', 0)}")
            print(f"- Success rate: {dashboard.get('success_rate', 0):.2%}")
            print(f"- Avg execution time: {metrics.get('average_execution_time', 0):.2f}s")
            print(f"- Total cost estimate: ${metrics.get('total_cost', 0):.4f}")
        else:
            print("- Monitoring data not collected correctly")
            print(f"- Completed tasks: {len(result.get('completed_tasks', {}))}")
            print(f"- Failed tasks: {len(result.get('failed_tasks', {}))}")

        print(f"\nError count: {result.get('error_count', 0)}")
        if result.get("early_exit_triggered"):
            print("Early exit was triggered")

    except Exception as e:
        print(f"System execution error: {str(e)}")
        import traceback
        traceback.print_exc()
