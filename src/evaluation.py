import json
import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import OPENAI_API_KEY
from src.pipeline import RAGPipeline


def load_golden_dataset(path: str = "data/golden_dataset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_dataset(pipeline: RAGPipeline, dataset: list[dict]) -> EvaluationDataset:
    """Run pipeline on all questions and format for RAGAS evaluation."""
    samples = []
    for item in dataset:
        result = pipeline.query(item["question"])
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=item["ground_truth"],
        ))
    return EvaluationDataset(samples=samples)


def evaluate_ragas(dataset: EvaluationDataset) -> dict:
    """Run RAGAS evaluation and return scores."""
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=OPENAI_API_KEY))

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    return {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_recall": df["context_recall"].mean(),
        "context_precision": df["context_precision"].mean(),
    }


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
