import spacy
from glossary_management import GLOSSARY  # (removed to avoid conflict with local GLOSSARY below)


def extract_terms_with_ruler(text, glossary):
    # Ensure you've run: python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    # If called multiple times, avoid duplicate pipes
    if "term_ruler" in nlp.pipe_names:
        nlp.remove_pipe("term_ruler")

    # Case-insensitive phrase matching
    ruler = nlp.add_pipe("entity_ruler", name="term_ruler", before="ner",
                         config={"phrase_matcher_attr": "LOWER"})

    patterns = []
    seen = set()  # de-dupe (label, pattern_lower)

    for entry in glossary:  # GLOSSARY is a list of dicts
        term = (entry.get("term") or "").strip()
        if term:
            key = ("TERM", term.lower())
            if key not in seen:
                patterns.append({"label": "TERM", "pattern": term})
                seen.add(key)

        for synonym in (entry.get("synonyms") or []):
            syn = (synonym or "").strip()
            if syn:
                key = ("TERM", syn.lower())
                if key not in seen:
                    patterns.append({"label": "TERM", "pattern": syn})
                    seen.add(key)

    if patterns:
        ruler.add_patterns(patterns)

    doc = nlp(text)
    candidates = {ent.text for ent in doc.ents if ent.label_ == "TERM"}
    return candidates


# --- Model example ---
# https://modelscope.cn/models/iic/nlp_raner_named-entity-recognition_english-large-ai/summary

text = "CNN is a type of neural network. The CPU cost per unit is high."
candidates = extract_terms_with_ruler(text, GLOSSARY)
print("Extracted Candidates:", candidates)


# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "CNN is a type of neural network. The CPU cost per unit is high."

ner_results = nlp(example)
print(ner_results[0]['word'])