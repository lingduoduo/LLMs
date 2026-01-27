# htn_core.py - HTN Task Decomposition Core Demo (100-line simplified version)
# Focus: how HTN decomposes a complex task into executable subtasks

from typing import Dict, Any

# ================ HTN Task Decomposition Rules ================

# Define how tasks are decomposed — this is the core of HTN
DECOMPOSITION_RULES = {
    "Review Contract": [
        {"name": "Analyze Structure", "type": "Primitive Task"},
        {"name": "Risk Assessment", "type": "Compound Task"},  # requires further decomposition
        {"name": "Generate Report", "type": "Primitive Task"},
    ],
    "Risk Assessment": [
        {"name": "Check Exclusivity", "type": "Primitive Task"},
        {"name": "Check Data Clauses", "type": "Primitive Task"},
        {"name": "Check Termination Conditions", "type": "Primitive Task"},
    ],
}

# ================ HTN Execution Engine ================

class HTNEngine:
    """HTN execution engine: demonstrates the core logic of task decomposition"""

    def __init__(self):
        self.depth = 0      # used to display decomposition levels
        self.risk_count = 0 # HTN dynamic feature: a risk counter

    def execute(self, task_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """HTN core: execute a task (decompose or run directly)"""

        context = context or {}
        indent = "  " * self.depth

        print(f"{indent}Executing task: {task_name}")

        # [HTN key decision] Does this task need decomposition?
        if task_name in DECOMPOSITION_RULES:
            return self._decompose_and_execute(task_name, context)
        else:
            return self._execute_primitive(task_name, context)

    def _decompose_and_execute(self, task_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """[HTN core] Decompose a compound task and execute recursively"""

        indent = "  " * self.depth
        print(f"{indent}Decomposing '{task_name}' into subtasks:")

        subtasks = DECOMPOSITION_RULES[task_name]

        # Show decomposition structure
        for i, subtask in enumerate(subtasks, 1):
            task_type = "[Primitive]" if subtask["type"] == "Primitive Task" else "[Compound]"
            print(f"{indent}  {i}. {task_type} {subtask['name']}")

        print(f"{indent}Starting subtask execution:")

        results = {}
        self.depth += 1

        for subtask in subtasks:
            result = self.execute(subtask["name"], context)
            results[subtask["name"]] = result

            # [Dynamic feature 1] Update context based on execution results
            if result.get("Risk Found"):
                self.risk_count += 1
                context["High Risk"] = True
                context["Risk Count"] = self.risk_count
                print(f"{indent}  [Dynamic Update] Found risk #{self.risk_count}; adjusting downstream strategy")

                # [Dynamic feature 2] Insert a new task dynamically if risk exceeds threshold
                if self.risk_count >= 2:
                    print(f"{indent}  [Dynamic Insertion] Too many risks; inserting deep analysis task")
                    deep_analysis_result = self._execute_primitive("Deep Risk Analysis", context)
                    results["Deep Risk Analysis"] = deep_analysis_result

        self.depth -= 1
        print(f"{indent}'{task_name}' completed")

        return {"Status": "Done", "Subtask Results": results}

    def _execute_primitive(self, task_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a primitive task (simulate actual work)"""

        indent = "  " * self.depth

        # [Dynamic feature 3] Choose an execution strategy based on context
        if task_name == "Check Exclusivity":
            if context.get("Contract Type") == "SaaS":
                print(f"{indent}  [Dynamic Strategy] Using SaaS-specific checks")
                print(f"{indent}  SaaS exclusivity check -> risk found")
                return {"Status": "Done", "Risk Found": True, "Clause": "Section 5.2"}
            else:
                print(f"{indent}  [Dynamic Strategy] Using standard checks")
                print(f"{indent}  Standard exclusivity check -> no risk")
                return {"Status": "Done", "Risk Found": False}

        elif task_name == "Check Data Clauses":
            if context.get("Contains Personal Data"):
                print(f"{indent}  [Dynamic Strategy] Using GDPR compliance checks")
                print(f"{indent}  GDPR data clause check -> risk found")
                return {"Status": "Done", "Risk Found": True, "Clause": "Section 8.3"}
            else:
                print(f"{indent}  [Dynamic Strategy] Using standard data checks")
                print(f"{indent}  Standard data clause check -> no risk")
                return {"Status": "Done", "Risk Found": False}

        elif task_name == "Deep Risk Analysis":
            print(f"{indent}  [Dynamic Task] Running deep risk analysis")
            return {"Status": "Done", "Findings": "Recommend involving legal counsel"}

        elif task_name == "Generate Report":
            # [Dynamic feature 4] Select report type based on risk level
            if context.get("High Risk"):
                print(f"{indent}  [Dynamic Strategy] Generating detailed high-risk report")
                return {"Status": "Done", "Report Type": "Detailed Risk Report"}
            else:
                print(f"{indent}  [Dynamic Strategy] Generating standard report")
                return {"Status": "Done", "Report Type": "Standard Report"}

        else:
            print(f"{indent}  Executing {task_name}")
            return {"Status": "Done", "Risk Found": False}

# ================ Demo Program ================

def demo_htn_decomposition():
    """Demonstrate HTN task decomposition"""

    print("=" * 60)
    print("HTN Task Decomposition Demo - Highlighting Dynamic Features")
    print("=" * 60)

    engine = HTNEngine()

    print("\n[Scenario 1] SaaS contract review - demonstrating HTN dynamic adaptation")
    print("-" * 40)
    context1 = {
        "Contract Type": "SaaS",
        "Contains Personal Data": True,
        "High Risk": False,
    }

    engine.execute("Review Contract", context1)

    print("\n[Scenario 2] Standard contract review - contrasting execution paths")
    print("-" * 40)

    # Reset risk counter to demonstrate a different scenario
    engine.risk_count = 0
    context2 = {
        "Contract Type": "Standard",
        "Contains Personal Data": False,
        "High Risk": False,
    }

    engine.execute("Review Contract", context2)


if __name__ == "__main__":
    demo_htn_decomposition()

'''
Note: Actual performance depends on the specific use case and data characteristics
(llm_clean)  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs   rag-optimization ±  /Users/linghuang/miniconda3/envs/llm_clean/bin/python /Users/linghuang/Git/LLMs/LLM-RAG/29/h
tn_core.py
============================================================
HTN Task Decomposition Demo - Highlighting Dynamic Features
============================================================

[Scenario 1] SaaS contract review - demonstrating HTN dynamic adaptation
----------------------------------------
Executing task: Review Contract
Decomposing 'Review Contract' into subtasks:
  1. [Primitive] Analyze Structure
  2. [Compound] Risk Assessment
  3. [Primitive] Generate Report
Starting subtask execution:
  Executing task: Analyze Structure
    Executing Analyze Structure
  Executing task: Risk Assessment
  Decomposing 'Risk Assessment' into subtasks:
    1. [Primitive] Check Exclusivity
    2. [Primitive] Check Data Clauses
    3. [Primitive] Check Termination Conditions
  Starting subtask execution:
    Executing task: Check Exclusivity
      [Dynamic Strategy] Using SaaS-specific checks
      SaaS exclusivity check -> risk found
    [Dynamic Update] Found risk #1; adjusting downstream strategy
    Executing task: Check Data Clauses
      [Dynamic Strategy] Using GDPR compliance checks
      GDPR data clause check -> risk found
    [Dynamic Update] Found risk #2; adjusting downstream strategy
    [Dynamic Insertion] Too many risks; inserting deep analysis task
      [Dynamic Task] Running deep risk analysis
    Executing task: Check Termination Conditions
      Executing Check Termination Conditions
  'Risk Assessment' completed
  Executing task: Generate Report
    [Dynamic Strategy] Generating detailed high-risk report
'Review Contract' completed

[Scenario 2] Standard contract review - contrasting execution paths
----------------------------------------
Executing task: Review Contract
Decomposing 'Review Contract' into subtasks:
  1. [Primitive] Analyze Structure
  2. [Compound] Risk Assessment
  3. [Primitive] Generate Report
Starting subtask execution:
  Executing task: Analyze Structure
    Executing Analyze Structure
  Executing task: Risk Assessment
  Decomposing 'Risk Assessment' into subtasks:
    1. [Primitive] Check Exclusivity
    2. [Primitive] Check Data Clauses
    3. [Primitive] Check Termination Conditions
  Starting subtask execution:
    Executing task: Check Exclusivity
      [Dynamic Strategy] Using standard checks
      Standard exclusivity check -> no risk
    Executing task: Check Data Clauses
      [Dynamic Strategy] Using standard data checks
      Standard data clause check -> no risk
    Executing task: Check Termination Conditions
      Executing Check Termination Conditions
  'Risk Assessment' completed
  Executing task: Generate Report
    [Dynamic Strategy] Generating standard report
'Review Contract' completed
'''