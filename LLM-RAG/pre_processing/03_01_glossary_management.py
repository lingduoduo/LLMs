import re
from typing import List, Dict, Any

# 1) Glossary
GLOSSARY = [
    {
        "term": "Convolutional Neural Network",
        "synonyms": ["CNN", "ConvNet"],
        "definition": "A type of computing model that mimics the structure and function of biological neural networks, especially effective for image processing.",
        "context_tags": ["Image Recognition", "Deep Learning"],
    },
    {
        "term": "Machine Learning",
        "synonyms": ["ML", "AI Modeling", "Machine Learning"],
        "definition": "A field of artificial intelligence that enables computer systems to learn from data without explicit programming.",
        "context_tags": ["Artificial Intelligence", "Data Science"],
    },
    {
        "term": "Natural Language Processing",
        "synonyms": ["NLP", "Natural Language"],
        "definition": "A field that studies interactions between human language and computers.",
        "context_tags": ["Artificial Intelligence", "Linguistics"],
    },
    {
        "term": "Central Processing Unit",
        "synonyms": ["CPU"],
        "definition": "The arithmetic, logic, and control unit of a computer.",
        "context_tags": ["Computer Hardware", "Computing"],
    },
    {
        "term": "Cost Per Unit",
        "synonyms": ["CPU"],  # ambiguous with Central Processing Unit
        "definition": "A metric used in business analysis to measure the cost of each product or service unit.",
        "context_tags": ["Business Analysis", "Financial Management", "Cost"],
    },
]


class TerminologyProcessor:
    def __init__(self, glossary: List[Dict[str, Any]]):
        self.glossary = glossary
        # standard_term_map: "central processing unit" -> "Central Processing Unit"
        self.standard_term_map: Dict[str, str] = {}
        # alias_to_entries_map: "cpu" -> [entry1, entry2] (to handle ambiguity)
        self.alias_to_entries_map: Dict[str, List[Dict[str, Any]]] = {}
        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build mappings for standard terms and aliases."""
        for entry in self.glossary:  # use self.glossary (not global)
            term = entry["term"]
            term_lower = term.lower()
            self.standard_term_map[term_lower] = term

            # include the term itself as an alias as well as its synonyms
            for alias in [term] + entry.get("synonyms", []):
                alias_lower = alias.lower()
                self.alias_to_entries_map.setdefault(alias_lower, []).append(entry)

    def _word_boundary_pattern(self, token: str) -> re.Pattern:
        """
        Compile a case-insensitive regex that matches the token as a whole word
        (letters/digits bounded); for non-ASCII/letter tokens, just escape.
        """
        if re.search(r"[A-Za-z0-9]", token):
            pat = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
        else:
            pat = re.escape(token)
        return re.compile(pat, flags=re.IGNORECASE)

    def standardize_term(self, text: str, context_window: int = 10) -> str:
        """
        Return the standardized term for a single alias/token.
        If ambiguous (e.g., 'CPU'), apply a small heuristic:
          - if financial keywords appear, pick 'Cost Per Unit'
          - otherwise default to 'Central Processing Unit'
        If not found, return the original text.
        """
        token = text.strip()
        key = token.lower()

        entries = self.alias_to_entries_map.get(key)
        if not entries:
            return text  # no mapping

        if len(entries) == 1:
            return entries[0]["term"]

        # tiny disambiguation for 'CPU' style collisions
        finance_cues = {"cost", "unit", "price", "per", "finance", "business"}
        hardware_cues = {"computer", "processor", "hardware", "core", "clock"}

        # look around the token if it's embedded in a sentence (best-effort)
        # (we don't have a larger text here, so we just check the token itself)
        token_context = token.lower()

        if any(w in token_context for w in finance_cues):
            for e in entries:
                if e["term"] == "Cost Per Unit":
                    return e["term"]

        # default to the computing meaning when ambiguous
        for e in entries:
            if e["term"] == "Central Processing Unit":
                return e["term"]

        # fallback to the first entry if nothing matched the heuristic
        return entries[0]["term"]

    def extract_terms(self, text: str) -> List[str]:
        """
        Extract and standardize all terms present in a longer text.
        Returns a sorted list of standardized term names (unique).
        """
        found: set[str] = set()
        # Sort by length (desc) so longer aliases match first (e.g., "ConvNet" before "Net")
        aliases_desc = sorted(self.alias_to_entries_map.keys(), key=len, reverse=True)

        for alias in aliases_desc:
            pattern = self._word_boundary_pattern(alias)
            if pattern.search(text):
                # resolve alias to a standardized term using standardize_term
                std = self.standardize_term(alias)
                found.add(std)

        return sorted(found)


# --- demo ---
if __name__ == "__main__":
    term_processor = TerminologyProcessor(GLOSSARY)
    print(term_processor.standardize_term("CPU"))  # -> "Central Processing Unit"
    sample = "We optimized the CNN on GPU; CPU cost per unit is tracked in finance."
    print(term_processor.extract_terms(sample))
    # e.g., ['Central Processing Unit', 'Convolutional Neural Network']
