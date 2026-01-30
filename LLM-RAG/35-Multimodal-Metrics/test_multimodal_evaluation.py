"""
Multimodal RAG Metrics Test
Tests both the class-based interface and the standalone function interface
"""

import numpy as np

# Optional: HuggingFace Dataset (may be slow / depend on pandas)
try:
    from datasets import Dataset  # type: ignore
except Exception:
    Dataset = None  # type: ignore

from custom_multimodal_metrics import (
    # Class interface
    multimodal_context_recall,
    cross_modal_alignment,
    visual_understanding_accuracy,
    # Standalone function interface
    evaluate_multimodal_retrieval,
    evaluate_visual_understanding,
    evaluate_cross_modal_alignment,
    parse_multimodal_contexts,
)

# -----------------------------------------------------------------------------
# Test Data (FULLY ENGLISH, aligned with the English-only metric code)
# -----------------------------------------------------------------------------
data_samples = {
    "question": [
        "On this Jiuzhaigou map, where is Five-Flower Lake located?",
        "On this Jiuzhaigou map, where is Five-Flower Lake located?",
        "On this Jiuzhaigou map, where is Five-Flower Lake located?",
    ],
    "answer": [
        "Sorry, I cannot determine the exact location of Five-Flower Lake from the provided information. "
        "Please refer to a more detailed map.",
        "Five-Flower Lake is located in the northern area of the Jiuzhaigou scenic region.",
        "According to the map, Five-Flower Lake is in the middle section of Rizhe Valley, about 2 km from Nuorilang Waterfall.",
    ],
    "ground_truth": [
        "Five-Flower Lake is located in the middle section of Rizhe Valley and is one of the most beautiful lakes in Jiuzhaigou.",
        "Five-Flower Lake is located in the middle section of Rizhe Valley and is one of the most beautiful lakes in Jiuzhaigou.",
        "Five-Flower Lake is located in the middle section of Rizhe Valley and is one of the most beautiful lakes in Jiuzhaigou.",
    ],
    "contexts": [
        [
            "Jiuzhaigou is composed of three valleys: Shuzheng Valley, Rizhe Valley, and Zechawa Valley.",
            "IMAGE_DESCRIPTION: An overview map of Jiuzhaigou, but the image is blurry and Five-Flower Lake is not clearly visible.",
        ],
        [
            "Five-Flower Lake is a famous attraction in Jiuzhaigou.",
            "IMAGE_DESCRIPTION: A map of the northern area of Jiuzhaigou showing some attractions, including the approximate area of Five-Flower Lake.",
        ],
        [
            "Rizhe Valley is one of the three main valleys of Jiuzhaigou.",
            "IMAGE_DESCRIPTION: A detailed, clear map that marks Five-Flower Lake in the middle section of Rizhe Valley, about 2 km from Nuorilang Waterfall.",
        ],
    ],
    "multimodal_contexts": [
        [
            {"type": "text", "content": "Jiuzhaigou is composed of three valleys: Shuzheng Valley, Rizhe Valley, and Zechawa Valley."},
            {"type": "image", "description": "An overview map of Jiuzhaigou, but the image is blurry."},
        ],
        [
            {"type": "text", "content": "Five-Flower Lake is a famous attraction in Jiuzhaigou."},
            {"type": "image", "description": "A northern-area map of Jiuzhaigou showing some attractions."},
        ],
        [
            {"type": "text", "content": "Rizhe Valley is one of the three main valleys of Jiuzhaigou."},
            {"type": "image", "description": "A detailed map clearly marking Five-Flower Lake in the middle section of Rizhe Valley."},
        ],
    ],
}


def _iter_samples_fallback():
    """Return iterable samples as a list[dict] if HF Dataset isn't available."""
    n = len(data_samples["question"])
    samples = []
    for i in range(n):
        samples.append({k: data_samples[k][i] for k in data_samples})
    return samples


def _make_dataset_or_list():
    """Prefer HF Dataset if available, otherwise fallback to list[dict]."""
    if Dataset is None:
        return _iter_samples_fallback()
    return Dataset.from_dict(data_samples)


def test_class_interface(dataset_or_list):
    """Test the RAGAS-compatible class interface"""
    print("=== Testing class interface (RAGAS-compatible) ===")

    mm_recall_score = multimodal_context_recall.score(dataset_or_list)
    alignment_score = cross_modal_alignment.score(dataset_or_list)
    visual_accuracy_score = visual_understanding_accuracy.score(dataset_or_list)

    print(f"Multimodal context recall: {mm_recall_score:.3f}")
    print(f"Cross-modal alignment:     {alignment_score:.3f}")
    print(f"Visual understanding:     {visual_accuracy_score:.3f}")

    overall_score = (mm_recall_score + alignment_score + visual_accuracy_score) / 3
    print(f"Overall score:            {overall_score:.3f}")

    return {
        "multimodal_recall": mm_recall_score,
        "cross_modal_alignment": alignment_score,
        "visual_understanding": visual_accuracy_score,
        "overall": overall_score,
    }


def test_function_interface():
    """Test the standalone function interface"""
    print("\n=== Testing standalone function interface ===")

    results = []
    n = len(data_samples["question"])

    for i in range(n):
        question = data_samples["question"][i]
        answer = data_samples["answer"][i]
        ground_truth = data_samples["ground_truth"][i]

        sample_data = {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "multimodal_contexts": data_samples["multimodal_contexts"][i],
            # You can also test the string contexts path by using:
            # "contexts": data_samples["contexts"][i],
        }

        contexts = parse_multimodal_contexts(sample_data)

        retrieval_result = evaluate_multimodal_retrieval(
            query=question,
            retrieved_contexts=contexts,
            ground_truth=ground_truth,
            answer=answer,
        )

        visual_score = evaluate_visual_understanding(
            question=question,
            image_descriptions=contexts["images"],
            answer=answer,
            ground_truth=ground_truth,
        )

        context_list = (
            [{"type": "text", "content": t} for t in contexts["text"]]
            + [{"type": "image", "description": img} for img in contexts["images"]]
        )
        alignment_score = evaluate_cross_modal_alignment(context_list, answer)

        sample_result = {
            "sample_id": i + 1,
            "retrieval": retrieval_result,
            "visual_understanding": visual_score,
            "cross_modal_alignment": alignment_score,
        }
        results.append(sample_result)

        print(f"\nSample {i+1}:")
        print(f"  Q: {question}")
        print(f"  A: {answer[:70]}...")
        print(f"  Retrieval:")
        print(f"    Text relevance:  {retrieval_result['text_relevance']:.3f}")
        print(f"    Image relevance: {retrieval_result['image_relevance']:.3f}")
        print(f"    Fusion score:    {retrieval_result['fusion_score']:.3f}")
        print(f"    Overall:         {retrieval_result['overall_score']:.3f}")
        print(f"  Visual understanding: {visual_score:.3f}")
        print(f"  Cross-modal alignment: {alignment_score:.3f}")

    return results


def compare_interfaces(dataset_or_list):
    """Compare results from class and function interfaces"""
    print("\n=== Comparing interfaces ===")

    class_results = test_class_interface(dataset_or_list)
    function_results = test_function_interface()

    avg_retrieval = float(np.mean([r["retrieval"]["overall_score"] for r in function_results]))
    avg_visual = float(np.mean([r["visual_understanding"] for r in function_results]))
    avg_alignment = float(np.mean([r["cross_modal_alignment"] for r in function_results]))

    print("\nComparison:")
    print(f"Multimodal recall - class: {class_results['multimodal_recall']:.3f}, function avg: {avg_retrieval:.3f}")
    print(f"Visual understanding - class: {class_results['visual_understanding']:.3f}, function avg: {avg_visual:.3f}")
    print(f"Cross-modal alignment - class: {class_results['cross_modal_alignment']:.3f}, function avg: {avg_alignment:.3f}")

    tolerance = 1e-3
    consistent = (
        abs(class_results["multimodal_recall"] - avg_retrieval) < tolerance
        and abs(class_results["visual_understanding"] - avg_visual) < tolerance
        and abs(class_results["cross_modal_alignment"] - avg_alignment) < tolerance
    )
    print(f"\nConsistency: {'✅ consistent' if consistent else '❌ inconsistent'}")


def main():
    print("=== Multimodal RAG Evaluation System Test ===")

    dataset_or_list = _make_dataset_or_list()

    # Basic validation
    if Dataset is not None and hasattr(dataset_or_list, "column_names"):
        print("Dataset validation:")
        print(f"  num_samples: {len(dataset_or_list)}")
        print(f"  columns: {dataset_or_list.column_names}")
    else:
        print("Dataset validation:")
        print(f"  using list[dict], num_samples: {len(list(dataset_or_list))}")

    # Run tests
    compare_interfaces(dataset_or_list)


if __name__ == "__main__":
    main()
