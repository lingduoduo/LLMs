# RAG

## File Extraction

### Dify RAG Extractor

- Unified Interface Design:

All document parsers inherit from the BaseExtractor class and provide a unified extract() method interface.

- Word Document Parsing:

Implemented in word_extractor.py to extract text content, retrieve images and save them to a specified directory, and
handle hyperlinks.

- PDF Document Parsing:

Implemented in pdf_extractor.py to extract text content page by page while preserving page element metadata.

- Unified Processing Workflow:

Through extract_processor.py, the corresponding parser is automatically selected based on the file extension, and all
parsers return a unified Document object format.

- Extensibility Design:

Uses the unstructured library as a backup parsing solution, making it easier to add support for new file formats.

https://github.com/langgenius/dify/blob/main/api/core/rag/extractor/

https://github.com/langgenius/dify/blob/main/api/core/rag/extractor/word_extractor.py

https://github.com/langgenius/dify/blob/main/api/core/rag/extractor/pdf_extractor.py

### Word File Extraction

`python-docx` is a Python library used for creating, modifying, and reading Microsoft Word .docx files.

### PDF Extraction

Dify used pypdfium2 to extract text only. Need to use OCR to parse layout, convert it to structured files, i.e.
markdown.

MinerU is an open-source tool that “converts PDFs into machine-readable formats (e.g., Markdown, JSON)” for document
extraction and layout parsing.

```
conda create -n MinerU python=3.12

pip install uv

uv pip install -U "mineru[core]"

mineru-models-download

mineru -p demo/pdfs/data.pdf -o demo/pdfs/ --source huggingface
```


## **Glossary Management**

### ** 1. Building and Maintaining a Technical Term Database**

#### **1.1 Causes of Terminology Confusion**
- Polysemy (multiple meanings for the same term)  
- Synonyms and near-sonyms  
- Domain-specific variations  
- Company- or product-specific terminology  

#### **1.2 Purpose of a Technical Term Database**
A technical term database forms the core infrastructure of a terminology consistency and optimization system. It provides a single source of truth for standardized terms, preventing ambiguity and miscommunication.

#### **1.3 Construction Process**
1. **Collect terminology sources**  
   Gather terms from documentation, codebases, design specs, academic publications, FAQs, etc.

2. **Standardize terms**  
   Normalize capitalization, formatting, spelling, and abbreviations.

3. **Establish mapping relationships**  
   Define relationships between standard terms, aliases, abbreviations, deprecated names, and cross-lingual equivalents.

4. **Add contextual information**  
   Include definitions, usage examples, domain tags, and common misuses.

5. **Build terminology index**  
   Organize and index terms for efficient querying and system integration.

**Key components of the database:**

| Field | Description |
|-------|-------------|
| Term | Standardized terminology |
| Synonyms / Aliases | Equivalent or similar terms |
| Definition | Clear explanation of the term |
| Context Tags | Usage scenarios or system modules |
| Domain | Field or sub-field where the term applies |
| Usage Example | Example of correct usage |
| External Link | Documentation or reference link |
| Stop Words | Terms ignored in processing |
| Misleading Terms | Common inaccurate or confusing alternatives |

---

## **2. RAG (Retrieval-Augmented Generation) & Terminology Integration**

1. **Data Preprocessing**  
   Term extraction, normalization, and context-aware document chunking.

2. **Terminology System Construction**  
   Build a structured terminology library with names, aliases, definitions, relationships, and version control.

3. **Embedding & Vectorization**  
   Generate embeddings for terms and alias variations. Fine-tune models (e.g., LoRA) to better understand domain terminology.

4. **Retrieval Enhancement**  
   - Expand queries using synonyms or aliases  
   - Implement hybrid search (vector + keyword-based retrieval)  
   - Re-rank results using terminology relevance or metadata  

5. **Generation Control**  
   - Prompt engineering with terminology constraints  
   - Enforce domain-specific term usage during generation  
   - Validate terminology accuracy post-generation  

6. **Evaluation & Feedback**  
   - Define metrics for terminology consistency  
   - Use LLM-as-a-Judge for automated evaluation  
   - Collect user feedback and continuously refine the system  

---

## **3. Maintenance and Continuous Improvement**

1. **Terminology schema design** – Define fields and relationships between terms.  
2. **Automated extraction & normalization** – Use NLP to source terminology from unstructured text.  
3. **Manual review and curation** – Domain experts refine and validate terminology.  
4. **Build relational / hierarchical structures** – Create ontologies or knowledge graphs to represent semantic relationships.  
5. **Versioning and update mechanisms** – Establish change logs, review cycles, and rollback strategies to ensure accuracy and adaptability.  

---
