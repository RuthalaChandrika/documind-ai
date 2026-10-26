"""RAG Evaluation Benchmark for DocuMind AI.

Measures real-world system performance across:
1. Retrieval Recall@K
2. Citation Correctness
3. Answer Faithfulness (grounded evidence verification & hallucination prevention)
4. Response Latency (end-to-end inference time per query)

Uses an empirical evaluation dataset with real Ollama llama3.2 & nomic-embed-text execution.
"""

import time
from pathlib import Path
from typing import Any, Dict, List
import pymupdf
from pydantic import BaseModel

from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import DocuMindVectorStore
from src.retriever import DocuMindRetriever
from src.rag_chain import DocuMindRAGChain
from src.chat_history import ConversationalRAGChain


class EvalItem(BaseModel):
    """Evaluation test case specification."""

    question: str
    expected_doc: str
    expected_page: int
    expected_fact: str
    is_answerable: bool


EVAL_DATASET: List[EvalItem] = [
    EvalItem(
        question="How many days in advance should employees submit planned leave requests?",
        expected_doc="employee_handbook.pdf",
        expected_page=1,
        expected_fact="5 working days",
        is_answerable=True,
    ),
    EvalItem(
        question="What is the annual coverage limit for preventive dental and vision care?",
        expected_doc="employee_handbook.pdf",
        expected_page=2,
        expected_fact="$2,000",
        is_answerable=True,
    ),
    EvalItem(
        question="What is the minimum password length and how often must it be rotated?",
        expected_doc="security_policy.pdf",
        expected_page=1,
        expected_fact="14 characters",
        is_answerable=True,
    ),
    EvalItem(
        question="What are the multi-factor authentication requirements for remote VPN access?",
        expected_doc="security_policy.pdf",
        expected_page=2,
        expected_fact="hardware security key",
        is_answerable=True,
    ),
    EvalItem(
        question="What is the reimbursement allowance for home gym equipment?",
        expected_doc="",
        expected_page=0,
        expected_fact="cannot find sufficient information",
        is_answerable=False,
    ),
]


def create_eval_documents(target_dir: Path) -> List[Path]:
    """Generate two synthetic domain PDFs for reproducible evaluation."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Document 1: employee_handbook.pdf
    doc1_path = target_dir / "employee_handbook.pdf"
    doc1 = pymupdf.open()
    p1 = doc1.new_page()
    p1.insert_text(
        (50, 72),
        "Company Handbook - Section 1: Working Hours and Leave Policy.\n\n"
        "Standard working hours are 9:00 AM to 5:00 PM EST from Monday through Friday.\n"
        "All planned annual leave requests must be submitted at least 5 working days in advance.\n"
        "Emergency sick leave should be communicated to the team lead before 8:30 AM.",
    )
    p2 = doc1.new_page()
    p2.insert_text(
        (50, 72),
        "Company Handbook - Section 2: Health and Wellness Benefits.\n\n"
        "Comprehensive health insurance covers major medical, prescription, and emergency care.\n"
        "Preventive dental and vision care is covered up to an annual limit of $2,000 per employee.\n"
        "Employees also receive an annual wellness stipend of $300 for ergonomic accessories.",
    )
    doc1.save(str(doc1_path))
    doc1.close()

    # Document 2: security_policy.pdf
    doc2_path = target_dir / "security_policy.pdf"
    doc2 = pymupdf.open()
    p1 = doc2.new_page()
    p1.insert_text(
        (50, 72),
        "Information Security Policy - Identity & Access Management.\n\n"
        "All employee account passwords must be a minimum of 14 characters in length.\n"
        "Passwords must include uppercase letters, lowercase letters, numbers, and symbols.\n"
        "Passwords must be rotated every 90 days, and past 5 passwords cannot be reused.",
    )
    p2 = doc2.new_page()
    p2.insert_text(
        (50, 72),
        "Information Security Policy - Remote Access & Network Infrastructure.\n\n"
        "Remote employees accessing internal servers must connect through the corporate VPN.\n"
        "VPN connections require mandatory multi-factor authentication (MFA) using a hardware security key.\n"
        "Unapproved third-party VPN software is strictly prohibited on company-managed devices.",
    )
    doc2.save(str(doc2_path))
    doc2.close()

    return [doc1_path, doc2_path]


def run_evaluation(temp_dir: Path) -> Dict[str, Any]:
    """Execute complete RAG benchmark against local Ollama models and calculate metrics.

    Args:
        temp_dir: Directory for storing evaluation PDFs and Chroma DB collection.

    Returns:
        Dictionary containing measured evaluation metrics.
    """
    # 1. Setup isolated database and index evaluation documents
    pdf_paths = create_eval_documents(temp_dir / "docs")
    store = DocuMindVectorStore(
        persist_directory=temp_dir / "eval_chroma",
        collection_name="eval_collection",
    )

    all_chunks = []
    for pdf_path in pdf_paths:
        docs = load_pdf(pdf_path)
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
        all_chunks.extend(chunks)

    store.add_documents(all_chunks, check_duplicates=False)

    retriever = DocuMindRetriever(vector_store=store, default_top_k=3)
    rag_chain = DocuMindRAGChain(retriever=retriever)
    conv_chain = ConversationalRAGChain(rag_chain=rag_chain)

    # 2. Run benchmark evaluation
    total_queries = len(EVAL_DATASET)
    retrieval_hits = 0
    citation_correct_count = 0
    faithful_count = 0
    latencies: List[float] = []

    case_results: List[Dict[str, Any]] = []

    for item in EVAL_DATASET:
        t_start = time.perf_counter()
        response = conv_chain.ask(item.question)
        latency = time.perf_counter() - t_start
        latencies.append(latency)

        answer_lower = response.answer.lower()

        # Check retrieval recall
        retrieved_file_names = [
            doc.metadata.get("file_name", "") for doc in response.retrieved_documents
        ]
        retrieval_hit = (
            item.expected_doc in retrieved_file_names if item.is_answerable else True
        )
        if retrieval_hit:
            retrieval_hits += 1

        # Check citation correctness
        if item.is_answerable:
            has_correct_citation = any(
                cit.file_name == item.expected_doc and cit.page == item.expected_page
                for cit in response.sources
            )
            if has_correct_citation:
                citation_correct_count += 1
        else:
            # For unanswerable questions, correct citation means NO fabricated citations
            has_correct_citation = len(response.sources) == 0
            if has_correct_citation:
                citation_correct_count += 1

        # Check faithfulness (grounded fact match or proper refusal)
        if item.is_answerable:
            is_faithful = (
                item.expected_fact.lower() in answer_lower and response.is_grounded
            )
        else:
            is_faithful = (
                "cannot find sufficient information" in answer_lower
                and not response.is_grounded
            )

        if is_faithful:
            faithful_count += 1

        case_results.append(
            {
                "question": item.question,
                "is_answerable": item.is_answerable,
                "expected_fact": item.expected_fact,
                "latency_sec": round(latency, 2),
                "retrieval_hit": retrieval_hit,
                "citation_correct": has_correct_citation,
                "faithful": is_faithful,
                "answer_snippet": response.answer[:120] + "...",
            }
        )

    recall_at_k = retrieval_hits / total_queries
    citation_accuracy = citation_correct_count / total_queries
    faithfulness_rate = faithful_count / total_queries
    avg_latency = sum(latencies) / len(latencies)

    summary = {
        "total_test_cases": total_queries,
        "retrieval_recall_at_k": round(recall_at_k * 100, 1),
        "citation_accuracy": round(citation_accuracy * 100, 1),
        "faithfulness_rate": round(faithfulness_rate * 100, 1),
        "average_latency_sec": round(avg_latency, 2),
        "min_latency_sec": round(min(latencies), 2),
        "max_latency_sec": round(max(latencies), 2),
        "case_results": case_results,
    }
    return summary


if __name__ == "__main__":
    import tempfile
    import json

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        print("Starting DocuMind AI empirical evaluation...")
        results = run_evaluation(Path(tmp_dir))
        print("\n================== BENCHMARK RESULTS ==================")
        print(f"Total Test Cases:       {results['total_test_cases']}")
        print(f"Retrieval Recall@K:     {results['retrieval_recall_at_k']}%")
        print(f"Citation Accuracy:      {results['citation_accuracy']}%")
        print(f"Answer Faithfulness:    {results['faithfulness_rate']}%")
        print(f"Average Latency:        {results['average_latency_sec']}s")
        print(f"Latency Range:          {results['min_latency_sec']}s - {results['max_latency_sec']}s")
        print("=======================================================")
        print(json.dumps(results["case_results"], indent=2))
