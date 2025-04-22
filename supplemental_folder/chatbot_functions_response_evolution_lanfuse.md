# Project Overview

This document covers four key areas of the Expanse chatbot project:
1. **Chatbot Functionality**  
2. **Langfuse for LLM Tracing**  
3. **Prompt Engineering Techniques**  
4. **Domain Adaptation via Unsupervised Fine‑Tuning**

---

## 1. Chatbot Functionality Overview

When a user submits a query, the chatbot responds using a two‑stage RAG system:

### 1.1 Standard RAG (Default)
- **Vector Retrieval:** Uses Basic, BM25‑reranked, or Fusion retrievers to pull relevant document chunks.  
- **Initial Answer:** Generates a concise, grounded response based on those chunks.

### 1.2 Interactive Dropdowns (Post‑Answer)
- **Feedback & Scoring:** Users rate answer quality and provide comments.  
- **Sources:** Display retrieved chunks along with relevance scores.  
- **Knowledge Graph Answer:**  
  - Triggers Graph RAG, which leverages a knowledge graph for relationship‑aware reasoning whihc ideal for complex questions that require contextual understanding of referential data  
  - Returns richer, explainable answers by capturing cross/Inter document relationships.( refer kowledge_graph.html file to see the graph visually)
  

**Approach:** This setup ensures fast, grounded responses with deeper, explainable answers when needed. The goal was to emulate deep reasoning that we see in AI chat interfaces;  while the Knowledge Graph RAG takes longer, it often returns richer answers by capturing complex references and relationships across the content(  comparison examples in the rag_document)

---

## 2. Langfuse for LLM Tracing

Langfuse provides detailed tracing, debugging, and evaluation tools for our LLM workflows.

### 2.1 Setup Steps
1. **Start Docker Desktop**  
   Ensure Docker Desktop is running on your machine.  
2. **Launch Langfuse Stack** 
    In the project directory  run :
   ```bash
   docker-compose -f langfuse/docker-compose.yml up --build
   ```  
3. **Access the Interface**  
   Open your browser and go to [http://localhost:3000](http://localhost:3000).  
4. **Create an Account**  
   Sign up with a username and password.  
5. **Configure API Keys**  
   Generate your Langfuse API keys and add them to the `.env` as mentioned in the  README.md

### 2.2 Why Langfuse?
- **Open Source & AWS‑Compatible**  
  Integrates seamlessly with LangChain, LLAMA and the OpenAI SDK.  
- **Detailed Tracing & Debugging**  
  Offers end‑to‑end visibility into prompts, responses, and internal decision flows.  
- **Evaluation & Feedback Tools**  
  Built‑in metrics, feedback loops, and prompt‑testing can help drive continuous improvement at scale

---

## 3. Prompt Engineering Techniques

I tried various prompt engineering techniques such as chain of thought prompting,structred response formatting , self eval and refinement etc.Here are two prompting that I wanted to discuss:

### 3.1 Fixed‑Section Conditioning (Prompt 1)
- **Role Priming:** Single “You are an expert…” line to set tone and persona  
- **Rigid Structure:** Enforces **Summary / Key Details / Conclusion** sections using bold labels which can be useful for long answers realted to polices etc in this use case.  
- **Hallucination Guard:** “Use only the provided context” + per‑section fallback statements.  
- **Pros:** Consistent, easy to parse output.  
- **Cons:** Lacks flexibility for varied query types.

### 3.2 Role/Goal Framing with Dynamic Structuring (Prompt 2)
- **Role / Goal Blocks:** Separate `---Role---` and `---Goal---` sections to clarify persona and objective.  
- **Data Schema:** Defines “Document Chunks (DC)” as the sole knowledge source.  
- **Flexible Headings:** Instructs model to choose logical Markdown headings (Definitions, Analysis, etc.).  
- **Single Fallback:** Centralized “Insufficient Context” rule.  
- **Pros:** Adapts structure to each question, clearer task framing.  
- **Cons:** Slightly less predictable formatting.

**Comparison Examples** Standard RAG ( prompt1 ) VS Knowledge Graph RAG VS Standard RAG( prompt 2) mentioned in the rag_document)
```
-**prompt 1** - """
***You are an expert on escrow account regulations.***
Using only the information provided in the context below, answer the question clearly and concisely. Organize your response into structured sections for clarity.
If the context does not contain sufficient information to answer the question, explicitly state: "The context I gathered does not contain sufficient information to answer this question."
Context: {context}
Question: {question}
Answer:
**Summary:**  
- A concise summary capturing the main themes or regulatory points related to the question, based on the context.  
- If no such information exists, state: "No relevant summary available."
**Key Details:**  
- Bullet points highlighting specific provisions, requirements, or insights from the context.  
- If no relevant details are present, state: "Details not present in the context."
**Conclusion:**  
- A clear and conclusive answer to the question based on the context.  
- If the context was insufficient, state: "The context I gathered does not contain sufficient information to answer this question ,you may continue with: "However, based on my general understanding of escrow regulations and the context I gathered, maybe this could be helpful:"
"""

-**prompt 2** - """
---Role---
You are an expert on escrow account regulations responding to user queries using only the provided Document Chunks .
---Goal---
Generate a concise, accurate answer based solely on the information in the Document Chunks. Summarize and interpret relevant content from the chunks without introducing any unsupported details.
---User Query---
{question}
---Data Sources---
Document Chunks (DC):
{context}
---Response Rules---
- **Formatting:** Use Markdown with clear, descriptive section headings (e.g., Definitions, Regulatory Requirements, Analysis).     
- **Structure:** Organize your answer into logical sections, each focusing on main point.   
- **Insufficient Context:** If the DC do not contain enough information, begin with:  
  > “The provided context does not contain sufficient information to answer this question.”  
  You may then optionally offer a brief paragraph of general guidance grounded in escrow account regulations.  
- Do **not** include any information not present in the Document Chunks.
"""
---
```
## 4. Domain Adaptation via Unsupervised Fine‑Tuning

I performed continued pre‑training( unsupervised Fine tuning) on LLaMA3.2 2B to infuse deep domain knowledge and adaptation

### 4.1 Approach
1. **Corpus Assembly:** Scraped data from all the web pages of 12 CFR Part 1024 (https://www.consumerfinance.gov/rules-policy/regulations/1024/) and collapsed data into clean, continuous sentences.  
2. **Chat‑Style Wrapping:** Wrapped each passage in special tokens (`<|start_header_id|>…`) to align with the assistant format supported for llama fine-tuning  
3. **Response‑Only Training:** Used `unsloth.chat_templates.train_on_responses_only` to ignore user prompts and focus on learning from “assistant” outputs.

### 4.2 Benefits
- **Jargon & Tone Alignment:** Embeds domain‑specific terminology and formal style.  
- **General Language Retained:** Avoids overfitting by not fine‑tuning on narrow Q&A pairs.  
- **Reduced Hallucination:** When used in RAG, the model now favors true regulatory language over invented content.

**Approach:** To get a lightweight, domain‑tuned LLaMA that speaks escrow regulations fluently, perfectly primed for downstream retrieval and Q&A without losing its broad language capabilities.

After fine-tuning, the model outperformed the base LLaMA in answering domain-specific questions—even for very detailed examples (refer `rag_document` for e.g.). While its general instruction-following ability and conversational flow saw slight degradation, the primary goal was to showcase domain adaptation. With further fine-tuning using labeled Q&A data, its performance could be significantly enhanced.

---
