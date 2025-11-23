# 📂 Smart Legal Assistant - Complete Project Folder Structure (MLOps Ready)

## **Project Overview**

This is the final folder structure for the **Smart Legal Assistant** project targeting the Egyptian Penal Code. It supports **online training**, **model saving**, **fine-tuning**, **RAG pipeline**, **Weakness Detection**, **evaluation**, and **deployment**.

---

## **Folder Structure & File Descriptions**

```
smart_legal_assistant_project/
│
├── data/                       # All data sources
│   ├── laws/                    # Raw legal texts (Penal Code articles)
│   ├── precedents/              # Court rulings and previous cases
│   ├── defense_memos/           # Defense memorandums for training LLM
│   ├── cases_raw/               # Original unprocessed case files (PDF/TXT)
│   ├── cases_annotated/         # Manually labeled cases (Weak Points, Facts, Evidence, Articles)
│   ├── weak_points/             # Structured Weak Points examples
│   └── embeddings/              # Precomputed embeddings for RAG
│
├── notebooks/                   # Jupyter / Colab notebooks
│   ├── data_exploration.ipynb   # EDA for understanding dataset distribution
│   ├── preprocessing.ipynb      # Cleaning, OCR, NER, Chunking experiments
│   └── model_training.ipynb     # Fine-tuning LLMs and Weakness Detection
│
├── src/                         # Source code
│   ├── preprocessing/           # Cleaning & NER
│   │   ├── ocr.py
│   │   ├── text_cleaning.py
│   │   └── ner_extraction.py
│   ├── rag/                     # Retrieval pipeline
│   │   ├── vector_db.py
│   │   ├── retriever.py
│   │   └── query_handler.py
│   ├── models/                  # LLMs & Weakness Detection
│   │   ├── fine_tune.py
│   │   ├── lora_adapter.py
│   │   └── weakness_detector.py
│   ├── inference/               # End-to-end prediction pipeline
│   │   └── run_inference.py
│   ├── api/                     # FastAPI backend endpoints
│   │   ├── main.py
│   │   └── utils.py
│   ├── utils/                   # Helper functions
│   │   └── helpers.py
│   ├── evaluation/              # Model evaluation
│   │   ├── legal_benchmark.py   # Automated benchmarks on legal dataset
│   │   └── expert_review.py     # Human expert review interface
│   ├── data_pipeline/           # Data loading and versioning
│   │   ├── data_loader.py       # Unified data loading scripts
│   │   └── data_versioning.py   # Track dataset versions for reproducibility
│   └── config/                  # Configuration files
│       ├── model_config.yaml    # Model hyperparameters & RAG settings
│       └── api_config.yaml      # API host, port, authentication, endpoints
│
├── mlops/                       # MLOps related
│   ├── mlflow/                  # MLflow experiment tracking
│   │   └── mlflow_tracking.yaml
│   ├── ci_cd/                   # GitHub Actions workflows
│   │   └── workflow.yml
│   ├── registry/                # Saved models & versioning
│   │   └── model_v1/
│   └── monitoring/              # Logs & evaluation monitoring
│       └── monitor.py
│
├── deployment/                  # Deployment files
│   ├── docker/                  # Dockerfile, docker-compose.yml
│   ├── hf_spaces/               # HuggingFace Spaces configs
│   └── frontend/                # React / Flutter UI
│       ├── src/
│       └── public/
│
├── tests/                       # Unit tests
│   ├── test_rag.py              # Test RAG pipeline
│   └── test_models.py           # Test LLMs & Weakness Detection
│
├── scripts/                     # Utility scripts
│   ├── setup_env.sh             # Environment setup
│   └── download_data.py         # Download / preprocess dataset
│
├── docs/                        # Documentation
│   ├── api.md                   # API endpoint documentation
│   └── deployment.md            # Deployment instructions
│
├── requirements.txt             # Python dependencies
├── README.md                     # Project documentation
└── setup.py                      # Optional Python package setup
```

---

## **Key Notes**

1. **Evaluation** ensures model quality and credibility (legal_benchmark + expert_review).
2. **Data Pipeline** handles reproducibility and versioning.
3. **Config** separates hyperparameters and API settings from code.
4. **Tests** validate every module to prevent regressions.
5. **Scripts** simplify environment setup and data downloading.
6. **Docs** improve usability and maintainability.
7. Supports **MLOps workflow**: training online, saving models, fine-tuning, and deployment.
8. Fully modular, ready for **RAG + LLM + Weakness Detection + API + Frontend** integration.

