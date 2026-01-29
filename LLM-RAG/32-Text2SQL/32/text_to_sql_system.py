"""
DIN-SQL + DAIL-SQL Integrated System – Concise Demo Version
Core Techniques: Four-Stage Decomposed Reasoning + Continuous Learning Enhancement
"""

import json
import asyncio
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# =========================================================
# Enums & Data Structures
# =========================================================
class QueryComplexity(Enum):
    EASY = "EASY"
    NON_NESTED = "NON_NESTED"
    NESTED = "NESTED"


@dataclass
class QueryResult:
    query: str
    complexity: QueryComplexity
    entities: List[Dict[str, str]]
    sql: str
    success: bool
    execution_time: float


@dataclass
class LearningPattern:
    pattern_id: str
    intent: str
    keywords: List[str]
    success_rate: float
    usage_count: int


# =========================================================
# DIN-SQL Core
# =========================================================
class DINSQLCore:
    """DIN-SQL Core: Four-stage decomposed reasoning"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}

    async def process_query(self, query: str) -> QueryResult:
        """Four-stage processing pipeline"""
        start_time = time.time()

        # Stage 1: Schema Linking
        entities = await self._schema_linking(query)

        # Stage 2: Query Classification
        complexity = await self._classify_query(query, entities)

        # Stage 3: SQL Generation
        sql = await self._generate_sql(query, entities, complexity)

        # Stage 4: Self-Correction
        corrected_sql = await self._self_correction(sql, query)

        execution_time = time.time() - start_time

        return QueryResult(
            query=query,
            complexity=complexity,
            entities=entities,
            sql=corrected_sql,
            success=True,
            execution_time=execution_time,
        )

    async def _schema_linking(self, query: str) -> List[Dict[str, str]]:
        """Stage 1: Schema Linking – identify relevant tables and columns"""
        entities = []
        query_lower = query.lower()

        # Table name mapping (English only)
        table_mapping = {
            "student": "students",
            "course": "courses",
            "professor": "professors",
            "enrollment": "enrollments",
            "grade": "grades",
        }

        # Column name mapping (English only)
        column_mapping = {
            "name": "name",
            "major": "major",
            "gpa": "gpa",
            "credits": "credits",
            "score": "score",
            "semester": "semester",
        }

        for keyword, table in table_mapping.items():
            if keyword in query_lower:
                entities.append({"entity": table, "type": "table"})

        for keyword, column in column_mapping.items():
            if keyword in query_lower:
                entities.append({"entity": column, "type": "column"})

        return entities

    async def _classify_query(
        self, query: str, entities: List[Dict]
    ) -> QueryComplexity:
        """Stage 2: Query Classification – determine query complexity"""
        query_lower = query.lower()

        # NESTED: requires subqueries
        nested_keywords = [
            "highest",
            "lowest",
            "most",
            "least",
            "top",
            "rank",
        ]
        if any(keyword in query_lower for keyword in nested_keywords):
            return QueryComplexity.NESTED

        # NON_NESTED: requires GROUP BY / JOIN but no subquery
        join_keywords = [
            "each",
            "per",
            "group",
            "count",
            "statistics",
            "average",
        ]
        if any(keyword in query_lower for keyword in join_keywords):
            return QueryComplexity.NON_NESTED

        # EASY: simple queries
        return QueryComplexity.EASY

    async def _generate_sql(
        self, query: str, entities: List[Dict], complexity: QueryComplexity
    ) -> str:
        """Stage 3: SQL Generation based on query complexity"""
        if complexity == QueryComplexity.EASY:
            return self._generate_simple_sql(query, entities)
        elif complexity == QueryComplexity.NON_NESTED:
            return self._generate_complex_sql(query, entities)
        else:
            return self._generate_nested_sql(query, entities)

    def _generate_simple_sql(self, query: str, entities: List[Dict]) -> str:
        """Generate simple SQL"""
        tables = [e["entity"] for e in entities if e["type"] == "table"]
        main_table = tables[0] if tables else "students"

        if "all" in query.lower():
            return f"SELECT * FROM {main_table}"
        else:
            return f"SELECT * FROM {main_table} LIMIT 10"

    def _generate_complex_sql(self, query: str, entities: List[Dict]) -> str:
        """Generate complex SQL (GROUP BY, aggregation)"""
        query_lower = query.lower()

        if "count" in query_lower and "major" in query_lower:
            return "SELECT major, COUNT(*) AS count FROM students GROUP BY major"
        elif "average" in query_lower and "gpa" in query_lower:
            return "SELECT major, AVG(gpa) AS avg_gpa FROM students GROUP BY major"
        else:
            return "SELECT major, COUNT(*) FROM students GROUP BY major"

    def _generate_nested_sql(self, query: str, entities: List[Dict]) -> str:
        """Generate nested SQL (subqueries)"""
        query_lower = query.lower()

        if "highest" in query_lower and "gpa" in query_lower:
            return (
                "SELECT * FROM students "
                "WHERE gpa = (SELECT MAX(gpa) FROM students)"
            )
        elif "top" in query_lower:
            return "SELECT * FROM students ORDER BY gpa DESC LIMIT 5"
        else:
            return "SELECT * FROM students ORDER BY gpa DESC LIMIT 1"

    async def _self_correction(self, sql: str, query: str) -> str:
        """Stage 4: Self-correction of SQL"""
        corrected_sql = sql
        query_lower = query.lower()

        if "distinct" not in corrected_sql.lower() and "unique" in query_lower:
            corrected_sql = corrected_sql.replace("SELECT", "SELECT DISTINCT")

        if "descending" in query_lower and "desc" not in corrected_sql.lower():
            if "order by" in corrected_sql.lower():
                corrected_sql += " DESC"

        return corrected_sql


# =========================================================
# DAIL-SQL Core
# =========================================================
class DAILSQLCore:
    """DAIL-SQL Core: Continuous learning enhancement"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patterns: List[LearningPattern] = []
        self.query_history = []
        self.performance_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "avg_response_time": 0.0,
        }

    async def enhance_processing(
        self, query: str, din_result: QueryResult
    ) -> Dict[str, Any]:
        """Three-module enhancement pipeline"""
        prediction = await self._few_shot_learning(query)
        performance_insight = await self._adaptive_reasoning(din_result)
        similar_queries = await self._history_analysis(query)
        await self._update_learning(query, din_result)

        return {
            "prediction": prediction,
            "performance_insight": performance_insight,
            "similar_queries": similar_queries,
            "learning_stats": self._get_learning_stats(),
        }

    async def _few_shot_learning(self, query: str) -> Dict[str, Any]:
        """Few-shot learning for intent prediction"""
        query_lower = query.lower()

        intent_keywords = {
            "SELECT": ["show", "view", "list", "display", "all"],
            "AGGREGATE": ["count", "calculate", "total", "number"],
            "TOP_N": ["highest", "top", "rank"],
            "AVERAGE": ["average", "mean", "avg"],
        }

        predicted_intent = "SELECT"
        confidence = 0.5

        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                predicted_intent = intent
                confidence = 0.8
                break

        matching_patterns = [
            p for p in self.patterns if any(k in query_lower for k in p.keywords)
        ]

        if matching_patterns:
            best_pattern = max(matching_patterns, key=lambda p: p.success_rate)
            confidence = min(1.0, confidence + best_pattern.success_rate * 0.2)

        return {
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "matching_patterns": len(matching_patterns),
        }

    async def _adaptive_reasoning(self, din_result: QueryResult) -> Dict[str, Any]:
        """Adaptive reasoning based on execution feedback"""
        self.performance_metrics["total_queries"] += 1
        if din_result.success:
            self.performance_metrics["successful_queries"] += 1

        success_rate = (
            self.performance_metrics["successful_queries"]
            / self.performance_metrics["total_queries"]
        )

        total = self.performance_metrics["total_queries"]
        avg = self.performance_metrics["avg_response_time"]
        self.performance_metrics["avg_response_time"] = (
            (avg * (total - 1) + din_result.execution_time) / total
        )

        suggestions = []
        if success_rate < 0.8:
            suggestions.append("Improve SQL generation rules")
        if din_result.execution_time > 1.0:
            suggestions.append("Enable query caching")

        return {
            "success_rate": success_rate,
            "avg_response_time": self.performance_metrics["avg_response_time"],
            "optimization_suggestions": suggestions,
        }

    async def _history_analysis(self, query: str) -> List[Dict[str, Any]]:
        """Analyze historical queries for similarity"""
        query_words = set(query.lower().split())
        similar_queries = []

        for h in self.query_history[-10:]:
            hist_words = set(h["query"].lower().split())
            similarity = (
                len(query_words & hist_words) / len(query_words | hist_words)
                if query_words | hist_words
                else 0.0
            )
            if similarity > 0.3:
                similar_queries.append(
                    {
                        "query": h["query"],
                        "similarity": similarity,
                        "success": h["success"],
                    }
                )

        return sorted(similar_queries, key=lambda x: x["similarity"], reverse=True)[:3]

    async def _update_learning(self, query: str, din_result: QueryResult):
        """Update learning patterns and query history"""
        self.query_history.append(
            {
                "query": query,
                "success": din_result.success,
                "complexity": din_result.complexity.value,
                "timestamp": datetime.now(),
            }
        )

        keywords = query.lower().split()
        existing_pattern = None

        for pattern in self.patterns:
            if len(set(keywords) & set(pattern.keywords)) >= 2:
                existing_pattern = pattern
                break

        if existing_pattern:
            existing_pattern.usage_count += 1
            existing_pattern.success_rate = (
                (existing_pattern.success_rate * (existing_pattern.usage_count - 1)
                 + (1.0 if din_result.success else 0.0))
                / existing_pattern.usage_count
            )
        else:
            self.patterns.append(
                LearningPattern(
                    pattern_id=f"pattern_{len(self.patterns) + 1}",
                    intent="unknown",
                    keywords=keywords[:5],
                    success_rate=1.0 if din_result.success else 0.0,
                    usage_count=1,
                )
            )

    def _get_learning_stats(self) -> Dict[str, Any]:
        """Return learning statistics"""
        if not self.patterns:
            return {"total_patterns": 0, "avg_success_rate": 0.0}

        avg_success_rate = sum(p.success_rate for p in self.patterns) / len(self.patterns)

        return {
            "total_patterns": len(self.patterns),
            "avg_success_rate": avg_success_rate,
            "total_queries": len(self.query_history),
        }


# =========================================================
# Integrated System
# =========================================================
class IntegratedTextToSQLSystem:
    """Integrated Text-to-SQL System"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.din_sql = DINSQLCore(self.config)
        self.dail_sql = DAILSQLCore(self.config)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "din_sql": {"cache_enabled": True},
                "dail_sql": {"learning_enabled": True},
            }

    async def process_query(self, query: str) -> Dict[str, Any]:
        din_result = await self.din_sql.process_query(query)
        dail_result = await self.dail_sql.enhance_processing(query, din_result)

        return {
            "query": query,
            "din_sql_result": {
                "complexity": din_result.complexity.value,
                "entities": din_result.entities,
                "sql": din_result.sql,
                "success": din_result.success,
                "execution_time": din_result.execution_time,
            },
            "dail_sql_enhancement": dail_result,
            "overall_success": din_result.success,
        }


# =========================================================
# Demo
# =========================================================
async def run_demo():
    print("DIN-SQL + DAIL-SQL Integrated System Demo")
    print("=" * 50)

    system = IntegratedTextToSQLSystem()

    test_queries = [
        "Show all student information",
        "Count students per major",
        "Find the top 5 students with the highest GPA",
        "Show the average GPA of computer science students",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)

        result = await system.process_query(query)

        print(f"Complexity: {result['din_sql_result']['complexity']}")
        print(f"Generated SQL: {result['din_sql_result']['sql']}")
        print(f"Execution Time: {result['din_sql_result']['execution_time']:.3f}s")
        print(
            f"Predicted Intent: "
            f"{result['dail_sql_enhancement']['prediction']['predicted_intent']}"
        )
        print(
            f"Prediction Confidence: "
            f"{result['dail_sql_enhancement']['prediction']['confidence']:.2f}"
        )

        suggestions = result["dail_sql_enhancement"]["performance_insight"][
            "optimization_suggestions"
        ]
        if suggestions:
            print("Optimization Suggestions:", suggestions)

    stats = system.dail_sql._get_learning_stats()
    print("\nSystem Learning Statistics:")
    print(f"Total Patterns: {stats['total_patterns']}")
    print(f"Total Queries Processed: {stats['total_queries']}")
    print(f"Average Success Rate: {stats['avg_success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(run_demo())


'''
(base)  🐍 base  linghuang@Mac  ~/Git/LLMs   rag-optimization  /Users/linghuang/miniconda3/envs/llm_clean/bin/python /Users/linghuang/Git/LLMs/LLM-RAG/32-Text2SQL/32/t
ext_to_sql_system.py
DIN-SQL + DAIL-SQL Integrated System Demo
==================================================

Query 1: Show all student information
------------------------------
Complexity: EASY
Generated SQL: SELECT * FROM students
Execution Time: 0.000s
Predicted Intent: SELECT
Prediction Confidence: 0.80

Query 2: Count students per major
------------------------------
Complexity: NON_NESTED
Generated SQL: SELECT major, COUNT(*) AS count FROM students GROUP BY major
Execution Time: 0.000s
Predicted Intent: AGGREGATE
Prediction Confidence: 1.00

Query 3: Find the top 5 students with the highest GPA
------------------------------
Complexity: NESTED
Generated SQL: SELECT * FROM students WHERE gpa = (SELECT MAX(gpa) FROM students)
Execution Time: 0.000s
Predicted Intent: TOP_N
Prediction Confidence: 1.00

Query 4: Show the average GPA of computer science students
------------------------------
Complexity: NON_NESTED
Generated SQL: SELECT major, AVG(gpa) AS avg_gpa FROM students GROUP BY major
Execution Time: 0.000s
Predicted Intent: SELECT
Prediction Confidence: 1.00

System Learning Statistics:
Total Patterns: 3
Total Queries Processed: 4
Average Success Rate: 100.0%
'''