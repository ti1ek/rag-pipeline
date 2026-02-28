import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from src.pipeline import RAGPipeline


def load_golden_dataset(path: str = "data/golden_dataset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_dataset(pipeline: RAGPipeline, dataset: list[dict]) -> Dataset:
    """Run pipeline on all questions and format for RAGAS evaluation."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in dataset:
        result = pipeline.query(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def evaluate_ragas(dataset: Dataset) -> dict:
    """Run RAGAS evaluation and return scores."""
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return dict(result)


def run_experiment(pipeline: RAGPipeline, golden: list[dict], experiment_name: str) -> dict:
    """Run a full experiment: pipeline on dataset + RAGAS evaluation."""
    dataset = run_pipeline_on_dataset(pipeline, golden)
    scores = evaluate_ragas(dataset)
    return {
        "experiment": experiment_name,
        "config": dict(pipeline.config),
        **scores,
    }


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert experiment results to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Experiment": r["experiment"],
            "Faithfulness": r.get("faithfulness", None),
            "Answer Relevancy": r.get("answer_relevancy", None),
            "Context Recall": r.get("context_recall", None),
            "Context Precision": r.get("context_precision", None),
        })
    return pd.DataFrame(rows)
