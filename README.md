# 📘 Smart Legal Assistant (Egyptian Penal Code) – README

## **📌 Overview**

The **Smart Legal Assistant** is an AI-powered system designed to support Egyptian lawyers—specifically in the domain of the **Egyptian Penal Code**. The system performs advanced legal reasoning using NLP, LLMs, and Retrieval-Augmented Generation (RAG). It analyzes case documents, retrieves relevant legal articles, finds precedents, and identifies procedural or evidential weaknesses in the case.

The project is designed using **MLOps best practices**, including modular pipelines, reproducibility, model tracking, API deployment, and automated updates.

---

## **🎯 Project Goals**

* Build a real, practical AI assistant for lawyers.
* Focus on one legal domain: **Egyptian Penal Code**.
* Combine **Machine Learning + RAG + LLM reasoning**.
* Detect **weak points (legal loopholes)** in case files.
* Provide **explainable, legally grounded answers**.
* Fully reproducible using an MLOps pipeline.
* Deploy a working MVP without any paid cloud services.

---

## **🧠 Key Features**

### ✅ **1. Case Understanding Engine**

Extracts:

* Facts
* Evidence
* Legal entities (accused, victim, locations)
* Charges & relevant articles

### ✅ **2. RAG Pipeline (Retrieval-Augmented Generation)**

Retrieves:

* Penal Code articles
* Egyptian court precedents
* Similar case patterns
* Defense arguments from real memorandums

### ✅ **3. Weakness & Loophole Detector**

Automatically detects:

* Procedural errors (invalid arrest/search)
* Missing evidence
* Weak witness testimony
* Contradictions in statements
* Violations of criminal procedures

### ✅ **4. LLM Legal Reasoning**

Generates:

* Defense strategies
* Explanation of applicable articles
* Legal interpretation
* Structured case summaries

### ✅ **5. MLOps-Compliant Pipeline**

Includes:

* Model training & evaluation
* Versioning via HuggingFace
* MLflow for experiment tracking
* CI/CD
* API deployment
* Monitoring & rollback

---

## **📁 Project Structure**

```
root
│
├── data/
│   ├── laws/                  # Egyptian Penal Code articles (text)
│   ├── cases/                 # Case documents
│   ├── defenses/              # Defense memorandums
│   ├── precedents/            # Court rulings
│   └── processed/             # Cleaned text files
│
├── notebooks/                 # EDA and training notebooks
│
├── src/
│   ├── preprocessing/         # Text cleaning, parsing, NER
│   ├── rag/                   # Vector DB, retrieval pipeline
│   ├── models/                # LLMs, fine-tuning scripts
│   ├── inference/             # End-to-end pipeline
│   ├── api/                   # FastAPI endpoints
│   └── utils/                 # Helpers
│
├── mlops/
│   ├── mlflow/                # Configs for experiment tracking
│   ├── ci_cd/                 # GitHub Actions workflows
│   ├── registry/              # Model registry structure
│   └── monitoring/            # Logs and evaluation tools
│
├── deployment/
│   ├── docker/                # Dockerfile
│   ├── hf_spaces/             # HuggingFace Spaces deployment
│   └── frontend/              # Simple React/Flutter UI
│
└── README.md
```

---

## **🗂 Data Sources (FREE)**

### **1. Egyptian Penal Code**

* Official Government Portal (public legal texts)

### **2. Court Rulings**

* Published Egyptian Court of Cassation decisions

### **3. Public Defense Memorandums**

* Openly published legal documents

### **4. Self-Labeled Data** (custom)

* Annotated cases with:

  * charges
  * facts
  * procedural issues
  * legal weaknesses

---

## **🤖 Models Used**

### **Embedding Models**

* BGE-M3 (small, multilingual)
* Legal-BERT Arabic

### **LLMs for Reasoning**

* Qwen 2.5 7B
* Llama 3.1 8B
* Gemma 2 9B

### **Retrieval Models**

* BM25
* FAISS Vector Database
* Optional: ColBERT for legal retrieval

---

## **🏗 Architecture**

### **1. Input Layer**

User uploads a case → system extracts key legal elements.

### **2. Preprocessing**

* OCR (if PDF)
* Text chunking
* NER to detect legal entities

### **3. Retrieval Layer**

* Query → embeddings → FAISS
* Pulls articles + precedents + similar cases

### **4. LLM Reasoning**

The LLM receives a structured prompt:

```
Facts:
Relevant Articles:
Similar Cases:
Potential Issues:

Task: provide legal analysis + weaknesses + defense strategies.
```

### **5. Output**

* Case summary
* Relevant law articles
* Precedent rulings
* Weakness detection
* Defense plan

---

## **⚙ MLOps Pipeline**

### **Experiment Tracking**

* MLflow (local or hosted)

### **Model Registry**

* HuggingFace Model Hub

### **CI/CD**

* GitHub Actions:

  * run tests
  * validate model
  * build API
  * auto deploy to HF Spaces

### **Monitoring**

* Prompt quality checker
* Retrieval accuracy
* Model drift alerts

---

## **🚀 Deployment**

### **Backend API**

* FastAPI for:

  * `/summarize`
  * `/retrieve_articles`
  * `/detect_weaknesses`
  * `/legal_reasoning`

### **Frontend**

* Simple chat UI (React or Flutter)
* Upload PDF/Doc

### **Deployment Options (Free)**

* HuggingFace Spaces
* Render Free Tier
* Docker locally

---

## **🔎 Weakness Detection Logic**

The system flags:

* Lack of direct evidence
* Contradictory witness statements
* Invalid search or arrest
* Missing chain of custody
* Procedural violations
* No intent proof

Model is trained on:

* Annotated legal cases
* Defense patterns
* Precedent-based reasoning

---

## **🧪 Fine-tuning Strategy**

### **1. Instruction Tuning**

Train on legal Q/A and reasoning examples.

### **2. LoRA Adapter**

Used to fine‑tune large models on free GPUs.

### **3. Evaluation Metrics**

* Retrieval Accuracy
* Legal correctness
* Reasoning depth
* Hallucination rate

---

## **🧩 MVP Scope**

* Penal Code only
* 10–20 articles
* 50–100 case samples
* Basic RAG
* Weakness detection
* Simple UI

After MVP works → scale to:

* Economic Crimes
* Cybercrime Law
* Civil Law

---

## **📄 License**

This project is for **educational and research purposes only**.
It does **not** provide professional legal advice.

---

## **👩‍💻 Contributors**

* **Machine Learning & MLOps:** Your Name
* **Legal Research:** —
* **Backend & Frontend:** —

You can expand this section later.

---

## **📬 Contact**

For collaboration or support:

> Email: [example@mail.com](mailto:example@mail.com)
