# htn_core.py - HTN任务分解核心演示（100行精简版）
# 重点展示：HTN如何将复杂任务分解为可执行的子任务

from typing import Dict, Any

# ================ HTN任务分解规则 ================

# 定义任务如何分解：这是HTN的核心
DECOMPOSITION_RULES = {
    "审查合同": [
        {"name": "分析结构", "type": "原子任务"},
        {"name": "风险评估", "type": "复合任务"},  # 需要进一步分解
        {"name": "生成报告", "type": "原子任务"}
    ],
    "风险评估": [
        {"name": "检查排他性", "type": "原子任务"},
        {"name": "检查数据条款", "type": "原子任务"},
        {"name": "检查终止条件", "type": "原子任务"}
    ]
}

# ================ HTN执行引擎 ================

class HTNEngine:
    """HTN执行引擎：演示任务分解的核心逻辑"""
    
    def __init__(self):
        self.depth = 0  # 用于显示分解层级
        self.risk_count = 0  # HTN动态特性：风险计数器
    
    def execute(self, task_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """HTN核心：执行任务（分解或直接执行）"""
        
        context = context or {}
        indent = "  " * self.depth
        
        print(f"{indent}执行任务: {task_name}")
        
        # 【HTN关键判断】是否需要分解？
        if task_name in DECOMPOSITION_RULES:
            return self._decompose_and_execute(task_name, context)
        else:
            return self._execute_primitive(task_name, context)
    
    def _decompose_and_execute(self, task_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """【HTN核心】分解复合任务并递归执行"""
        
        indent = "  " * self.depth
        print(f"{indent}分解 '{task_name}' 为子任务:")
        
        # 获取分解规则
        subtasks = DECOMPOSITION_RULES[task_name]
        
        # 显示分解结构
        for i, subtask in enumerate(subtasks, 1):
            task_type = "[原子]" if subtask["type"] == "原子任务" else "[复合]"
            print(f"{indent}  {i}. {task_type} {subtask['name']}")
        
        print(f"{indent}开始执行子任务:")
        
        # 递归执行每个子任务
        results = {}
        self.depth += 1  # 增加缩进层级
        
        for subtask in subtasks:
            result = self.execute(subtask["name"], context)
            results[subtask["name"]] = result
            
            # 【HTN动态特性1】根据执行结果动态调整上下文
            if result.get("发现风险"):
                self.risk_count += 1
                context["高风险"] = True
                context["风险计数"] = self.risk_count
                print(f"{indent}  [动态调整] 发现第{self.risk_count}个风险，调整后续策略")
                
                # 【HTN动态特性2】风险超过阈值时动态插入新任务
                if self.risk_count >= 2:
                    print(f"{indent}  [动态插入] 风险过多，插入深度分析任务")
                    deep_analysis_result = self._execute_primitive("深度风险分析", context)
                    results["深度风险分析"] = deep_analysis_result
        
        self.depth -= 1  # 恢复缩进层级
        print(f"{indent}'{task_name}' 执行完成")
        
        return {"状态": "完成", "子任务结果": results}
    
    def _execute_primitive(self, task_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行原子任务（模拟具体工作）"""
        
        indent = "  " * self.depth
        
        # 【HTN动态特性3】根据上下文动态选择执行策略
        if task_name == "检查排他性":
            if context.get("合同类型") == "SaaS":
                print(f"{indent}  [动态策略] 使用SaaS专用检查规则")
                print(f"{indent}  SaaS合同排他性检查 -> 发现风险")
                return {"状态": "完成", "发现风险": True, "条款": "第5.2条"}
            else:
                print(f"{indent}  [动态策略] 使用标准检查规则")
                print(f"{indent}  标准排他性检查 -> 无风险")
                return {"状态": "完成", "发现风险": False}
        
        elif task_name == "检查数据条款":
            if context.get("涉及个人数据"):
                print(f"{indent}  [动态策略] 使用GDPR合规检查")
                print(f"{indent}  GDPR数据条款检查 -> 发现风险")
                return {"状态": "完成", "发现风险": True, "条款": "第8.3条"}
            else:
                print(f"{indent}  [动态策略] 使用标准数据检查")
                print(f"{indent}  标准数据条款检查 -> 无风险")
                return {"状态": "完成", "发现风险": False}
        
        elif task_name == "深度风险分析":
            print(f"{indent}  [动态任务] 执行深度风险分析")
            return {"状态": "完成", "分析结果": "建议法务介入"}
        
        elif task_name == "生成报告":
            # 【HTN动态特性4】根据风险情况动态选择报告类型
            if context.get("高风险"):
                print(f"{indent}  [动态策略] 生成高风险详细报告")
                return {"状态": "完成", "报告类型": "详细风险报告"}
            else:
                print(f"{indent}  [动态策略] 生成标准报告")
                return {"状态": "完成", "报告类型": "标准报告"}
        
        else:
            print(f"{indent}  执行 {task_name}")
            return {"状态": "完成", "发现风险": False}

# ================ 演示程序 ================

def demo_htn_decomposition():
    """演示HTN任务分解过程"""
    
    print("=" * 60)
    print("HTN 任务分解演示 - 重点展示动态特性")
    print("=" * 60)
    
    engine = HTNEngine()
    
    print("\n【场景1】SaaS合同审查 - 演示HTN动态适应")
    print("-" * 40)
    context1 = {
        "合同类型": "SaaS",
        "涉及个人数据": True,
        "高风险": False
    }
    
    result1 = engine.execute("审查合同", context1)
    
    print("\n【场景2】标准合同审查 - 对比不同执行路径")
    print("-" * 40)
    # 重置风险计数器以演示不同场景
    engine.risk_count = 0
    context2 = {
        "合同类型": "标准",
        "涉及个人数据": False,
        "高风险": False
    }
    
    result2 = engine.execute("审查合同", context2)
    

    
if __name__ == "__main__":
    demo_htn_decomposition()