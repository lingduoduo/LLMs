import torch
import logging
import time
from typing import List, Dict
from contextlib import contextmanager
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Logits Processor (HF-native replacement for lm_head hook)
# ------------------------------------------------------------
class AdaptiveKeywordBiasProcessor(LogitsProcessor):
    """
    Dynamically boosts logits of keyword tokens with decay
    to avoid over-repetition.
    """

    def __init__(self, token_ids: torch.Tensor, bias: float = 3.0, adaptive: bool = True):
        self.token_ids = token_ids
        self.base_bias = bias
        self.adaptive = adaptive
        self.step = 0

    def __call__(self, input_ids, scores):
        if self.adaptive:
            decay = max(0.3, 1.0 - self.step * 0.01)
            bias = self.base_bias * decay
        else:
            bias = self.base_bias

        scores[:, self.token_ids] += bias
        self.step += 1
        return scores


# ------------------------------------------------------------
# Keyword-Guided Generator (Hugging Face)
# ------------------------------------------------------------
class KeywordGuidedGenerator:
    """
    Keyword-guided text generator using Hugging Face models.

    Key design principles:
    1. Logits-level intervention (not attention)
    2. Multi-token keyword support
    3. Adaptive bias decay to preserve fluency
    4. Clean, stable HF-native implementation
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = None
        self.model = None
        self.keywords = []
        self.focus_token_ids = None

        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        ).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def set_keywords(self, keywords: List[str]):
        """Encode keywords into token IDs (multi-token safe)."""
        self.keywords = keywords
        token_ids = set()

        for kw in keywords:
            ids = self.tokenizer.encode(kw, add_special_tokens=False)
            token_ids.update(ids)

        self.focus_token_ids = torch.tensor(
            sorted(token_ids),
            device=self.device,
            dtype=torch.long,
        )

        logger.info(f"Focused token IDs: {self.focus_token_ids.tolist()}")

    def _build_prompt(self, user_prompt: str) -> str:
        """Apply chat template if available."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {user_prompt}\nAssistant:"

    def generate_text(
        self,
        prompt: str,
        use_guidance: bool = False,
        bias_strength: float = 3.0,
        adaptive: bool = True,
        **gen_kwargs,
    ) -> str:
        """Generate text with optional keyword guidance."""
        text = self._build_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        logits_processor = None
        if use_guidance:
            logits_processor = LogitsProcessorList([
                AdaptiveKeywordBiasProcessor(
                    token_ids=self.focus_token_ids,
                    bias=bias_strength,
                    adaptive=adaptive,
                )
            ])

        default_kwargs = dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            no_repeat_ngram_size=2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        default_kwargs.update(gen_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                logits_processor=logits_processor,
                **default_kwargs,
            )

        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def analyze_keywords(self, text: str) -> Dict:
        """Analyze keyword coverage and diversity."""
        total = 0
        present = []

        details = {}
        for kw in self.keywords:
            count = text.count(kw)
            if count > 0:
                details[kw] = count
                total += count
                present.append(kw)

        coverage = len(present) / len(self.keywords) if self.keywords else 0.0
        diversity = len(present) / total if total > 0 else 0.0

        return {
            "total_count": total,
            "coverage_rate": coverage,
            "diversity_score": diversity,
            "keyword_details": details,
            "present_keywords": present,
        }

    def benchmark(self, prompt: str, runs: int = 3) -> Dict:
        """Benchmark generation with and without keyword guidance."""
        results = {"baseline": [], "guided": []}

        for _ in range(runs):
            t0 = time.time()
            out = self.generate_text(prompt, use_guidance=False)
            results["baseline"].append(
                (out, self.analyze_keywords(out), time.time() - t0)
            )

        for _ in range(runs):
            t0 = time.time()
            out = self.generate_text(prompt, use_guidance=True)
            results["guided"].append(
                (out, self.analyze_keywords(out), time.time() - t0)
            )

        return results


# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------
def main():
    generator = KeywordGuidedGenerator()
    generator.set_keywords([
        "large language models",
        "artificial intelligence",
        "healthcare",
        "autonomous driving",
        "intelligent customer service",
    ])

    prompt = (
        "Please discuss the future development trends of artificial intelligence "
        "and large language models, and analyze their potential applications and "
        "challenges across industries such as healthcare, autonomous driving, "
        "and intelligent customer service."
    )

    print("\n===== Baseline =====")
    out1 = generator.generate_text(prompt)
    print(out1)
    print(generator.analyze_keywords(out1))

    print("\n===== Keyword-Guided =====")
    out2 = generator.generate_text(prompt, use_guidance=True)
    print(out2)
    print(generator.analyze_keywords(out2))


if __name__ == "__main__":
    main()
