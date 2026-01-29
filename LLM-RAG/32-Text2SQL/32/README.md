# DIN-SQL + DAIL-SQL 集成系统

## 系统概述

本系统集成了DIN-SQL的四阶段分解推理和DAIL-SQL的持续学习增强，实现完整的Text-to-SQL解决方案。

## 核心技术

### DIN-SQL四阶段流程
1. **Schema Linking**: 识别查询中的表名和列名
2. **Query Classification**: 将查询分类为EASY/NON_NESTED/NESTED
3. **SQL Generation**: 根据复杂度生成对应SQL
4. **Self Correction**: 对生成的SQL进行校正优化

### DAIL-SQL增强模块
1. **Few-Shot Learning**: 基于少样本学习进行意图预测
2. **Adaptive Reasoning**: 自适应推理和性能监控
3. **History Analysis**: 历史查询分析和相似度匹配

## 文件结构

```
code-32/
├── text_to_sql_system.py    # 核心系统实现
├── config.json              # 系统配置
├── README.md                # 使用说明
└── FINAL_SUMMARY.md         # 项目总结
```

## 快速开始

### 运行演示
```bash
python text_to_sql_system.py
```

### 使用示例
```python
from text_to_sql_system import IntegratedTextToSQLSystem
import asyncio

async def main():
    system = IntegratedTextToSQLSystem()
    result = await system.process_query("显示所有学生信息")
    print(result)

asyncio.run(main())
```

## 系统特性

- **高准确性**: DIN-SQL四阶段分解推理确保SQL生成准确性
- **持续学习**: DAIL-SQL从每次查询中学习，不断优化性能
- **智能预测**: 基于历史数据进行意图预测和相似查询匹配
- **性能监控**: 实时监控系统性能并提供优化建议

## 技术架构

```
用户查询 → DIN-SQL分解推理 → DAIL-SQL学习增强 → 最终结果
         ↓                    ↓
    四阶段处理流程        三模块增强处理
```

## 配置说明

`config.json`支持以下配置项：
- `din_sql.cache_enabled`: 是否启用缓存
- `dail_sql.learning_enabled`: 是否启用学习功能
- `dail_sql.similarity_threshold`: 相似度阈值
