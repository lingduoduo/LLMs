"""
Memory Transformer Comprehensive Demo
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import MemoryTransformer

class TraditionalContext:
    """Traditional context window simulation"""
    def __init__(self, max_size=8):
        self.context = []
        self.max_size = max_size

    def add(self, content):
        self.context.append(content)
        if len(self.context) > self.max_size:
            self.context.pop(0)

    def search(self, query):
        results = []
        for item in self.context:
            if any(word in item for word in query.split()):
                results.append(item)
        return results


def run_multi_turn_demo():
    """Multi-turn conversation demo"""
    print("Multi-turn Conversation Memory Retention Demo")
    print("=" * 50)

    model = MemoryTransformer()

    conversations = [
        ("User", "I need to review an agency contract, especially focusing on exclusivity clauses"),
        ("Assistant", "Understood. I will focus on exclusivity clauses. Please provide the contract content"),
        ("User", "The contract states that violating the exclusivity clause requires paying a 50% penalty"),
        ("Assistant", "A 50% penalty is indeed quite high and requires risk assessment"),
        ("User", "There is also an obligation to maintain exclusivity for 2 years after termination"),
        ("Assistant", "A 2-year obligation period is long; negotiation to shorten it is recommended"),
        ("User", "What are the consequences of violating the exclusivity clause?"),
        ("Assistant", "Primarily the 50% penalty and potential legal action"),
        ("User", "Is there a way to reduce the penalty percentage?"),
        ("Assistant", "It can be negotiated or mitigated by adding exemption clauses"),
        ("User", "Summarize the main risks of the exclusivity clause"),
    ]

    print("Simulating 10 conversation turns and observing memory system changes...")

    # Track actual state after each turn
    for i, (role, content) in enumerate(conversations[:-1]):
        model.process(f"{role}: {content}")

        actual_status = model.storage.get_status()
        total_memories = actual_status["short_term"] + actual_status["long_term"]

        # Display status every 3 turns for accuracy
        if (i + 1) % 3 == 0 or i == 0:
            print(
                f"   After turn {i+1} → Total memories: {total_memories} "
                f"(Short-term: {actual_status['short_term']}, "
                f"Long-term: {actual_status['long_term']})"
            )

    # Final state
    final_status = model.storage.get_status()
    final_total = final_status["short_term"] + final_status["long_term"]
    print(
        f"   Final state → Total memories: {final_total} "
        f"(Short-term: {final_status['short_term']}, "
        f"Long-term: {final_status['long_term']})"
    )

    final_query = conversations[-1][1]
    print(f"\nIntelligent retrieval test: {final_query}")

    retrieved = model.query(final_query, top_k=3)
    print(
        f"Retrieved {len(retrieved)} most relevant items "
        f"from {final_total} memories:"
    )

    for i, memory in enumerate(retrieved):
        content_preview = (
            memory.content
            .replace("User: ", "")
            .replace("Assistant: ", "")[:40]
        )
        print(f"   {i+1}. {content_preview}...")
        print(
            f"      [Importance: {memory.importance:.2f}, "
            f"Access count: {memory.access_count}]"
        )

    return model


def run_comparison_demo():
    """Performance comparison demo"""
    print("\nTraditional Approach vs Memory Transformer Comparison")
    print("=" * 50)

    traditional = TraditionalContext(max_size=6)
    memory_transformer = MemoryTransformer()

    test_sequence = [
        "Important: exclusivity clause penalty is 50% of the contract value",
        "General chat: the weather is nice today",
        "Important: exclusivity obligations continue for 2 years after termination",
        "General chat: hello, how can I help you?",
        "General chat: thank you for your help",
        "Important: the agent has exclusive sales rights in the region",
        "General chat: goodbye, see you tomorrow",
        "General chat: wish you success at work",
        "General chat: today’s meeting was successful",
        "General chat: we will continue the discussion next week",
    ]

    print("Inputting 10 messages (3 important, 7 general)...")

    important_items = []
    for i, item in enumerate(test_sequence):
        traditional.add(item)
        memory_transformer.process(item)

        if "Important" in item:
            important_items.append(item)
            print(f"   Item {i+1}: [Key Information] {item[10:50]}...")
        else:
            print(f"   Item {i+1}: [General Chat] {item[13:35]}...")

    print("\nStorage status:")
    print(f"   Traditional approach: keeps last {len(traditional.context)} items")

    mt_status = memory_transformer.storage.get_status()
    print(
        f"   Memory Transformer: {mt_status['short_term'] + mt_status['long_term']} total items"
    )
    print(f"     - Short-term memory: {mt_status['short_term']}")
    print(f"     - Long-term memory: {mt_status['long_term']}")

    query = "exclusivity clause penalty"
    print(f"\nTest query: '{query}'")

    # Traditional retrieval
    traditional_results = traditional.search("exclusivity penalty")
    traditional_important = [r for r in traditional_results if "Important" in r]

    # Memory Transformer retrieval
    mt_results = memory_transformer.query(query, top_k=5)
    mt_important = [r for r in mt_results if "Important" in r.content]

    total_important = len(important_items)

    print("\nRetrieval results comparison:")
    print(f"   Total important items: {total_important}")
    print("   Traditional approach:")
    print(f"     - Retrieved: {len(traditional_results)} relevant items")
    print(
        f"     - Important items: {len(traditional_important)} "
        f"({len(traditional_important) / total_important * 100:.0f}%)"
    )

    print("   Memory Transformer:")
    print(f"     - Retrieved: {len(mt_results)} relevant items")
    print(
        f"     - Important items: {len(mt_important)} "
        f"({len(mt_important) / total_important * 100:.0f}%)"
    )

    traditional_recall = len(traditional_important) / total_important
    mt_recall = len(mt_important) / total_important

    if mt_recall > traditional_recall:
        improvement = (mt_recall - traditional_recall) * 100
        print(f"\nPerformance gain: Memory Transformer improves recall by {improvement:.0f}%")
    elif mt_recall == traditional_recall:
        print("\nBoth approaches perform similarly in this test")
    else:
        decline = (traditional_recall - mt_recall) * 100
        print(f"\nNote: Memory Transformer underperforms by {decline:.0f}% in this test")

    if mt_important:
        print("\nImportant information retrieved by Memory Transformer:")
        for i, memory in enumerate(mt_important):
            content = memory.content.replace("Important:", "")[:40]
            print(
                f"   {i+1}. {content}... "
                f"(Importance: {memory.importance:.2f})"
            )

    return {
        "traditional_recall": traditional_recall,
        "mt_recall": mt_recall,
        "improvement": mt_recall - traditional_recall
    }


def main():
    """Main demo entry point"""
    print("Memory Transformer Comprehensive Demo")
    print("=" * 60)

    run_multi_turn_demo()
    results = run_comparison_demo()

    print("\n" + "=" * 50)
    print("Demo Summary:")
    print(f"   Traditional recall: {results['traditional_recall'] * 100:.0f}%")
    print(f"   Memory Transformer recall: {results['mt_recall'] * 100:.0f}%")

    if results["improvement"] > 0:
        print("   Memory Transformer performs better in this test")
    elif results["improvement"] == 0:
        print("   Both approaches perform similarly")
    else:
        print("   Traditional approach performs better in this test")

    print("\nNote: Actual performance depends on the specific use case and data characteristics")


if __name__ == "__main__":
    main()

'''
(llm_clean)  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs   rag-optimization ±  /Users/linghuang/miniconda3/envs/llm_clean/bin/python /Users/linghuang/Git/LLMs/LLM-RAG/28/d
emo.py
Memory Transformer Comprehensive Demo
============================================================
Multi-turn Conversation Memory Retention Demo
==================================================
Simulating 10 conversation turns and observing memory system changes...
   After turn 1 → Total memories: 1 (Short-term: 1, Long-term: 0)
   After turn 3 → Total memories: 3 (Short-term: 3, Long-term: 0)
   After turn 6 → Total memories: 6 (Short-term: 5, Long-term: 1)
   After turn 9 → Total memories: 9 (Short-term: 8, Long-term: 1)
   Final state → Total memories: 10 (Short-term: 9, Long-term: 1)

Intelligent retrieval test: Summarize the main risks of the exclusivity clause
Retrieved 3 most relevant items from 10 memories:
   1. What are the consequences of violating t...
      [Importance: 0.55, Access count: 1]
   2. The contract states that violating the e...
      [Importance: 0.55, Access count: 1]
   3. It can be negotiated or mitigated by add...
      [Importance: 0.40, Access count: 1]

Traditional Approach vs Memory Transformer Comparison
==================================================
Inputting 10 messages (3 important, 7 general)...
   Item 1: [Key Information]  exclusivity clause penalty is 50% of th...
   Item 2: [General Chat]  the weather is nice t...
   Item 3: [Key Information]  exclusivity obligations continue for 2 ...
   Item 4: [General Chat]  hello, how can I help...
   Item 5: [General Chat]  thank you for your he...
   Item 6: [Key Information]  the agent has exclusive sales rights in...
   Item 7: [General Chat]  goodbye, see you tomo...
   Item 8: [General Chat]  wish you success at w...
   Item 9: [General Chat]  today’s meeting was s...
   Item 10: [General Chat]  we will continue the ...

Storage status:
   Traditional approach: keeps last 6 items
   Memory Transformer: 10 total items
     - Short-term memory: 9
     - Long-term memory: 1

Test query: 'exclusivity clause penalty'

Retrieval results comparison:
   Total important items: 3
   Traditional approach:
     - Retrieved: 0 relevant items
     - Important items: 0 (0%)
   Memory Transformer:
     - Retrieved: 5 relevant items
     - Important items: 3 (100%)

Performance gain: Memory Transformer improves recall by 100%

Important information retrieved by Memory Transformer:
   1.  exclusivity clause penalty is 50% of th... (Importance: 0.55)
   2.  exclusivity obligations continue for 2 ... (Importance: 0.85)
   3.  the agent has exclusive sales rights in... (Importance: 0.40)

==================================================
Demo Summary:
   Traditional recall: 0%
   Memory Transformer recall: 100%
   Memory Transformer performs better in this test

Note: Actual performance depends on the specific use case and data characteristics
'''