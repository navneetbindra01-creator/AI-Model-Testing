import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import giskard
import pandas as pd
from giskard.llm import set_llm_model
import json
import giskard as gsk  # for scan convenience

# Configure logging
log_dir = r"C:\Users\navne\PycharmProjects\AITesting\cosinelogs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "cosine_test.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
api_key = os.getenv("GROK_API_KEY")
if not api_key:
    raise ValueError("GROK_API_KEY not found in .env file.")
logging.info("API key retrieved successfully.")

# Initialize Grok API client
client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
logging.info("Grok API client initialized.")

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("Embedding model initialized.")

# Define dirs and files
test_data_file = r"C:\Users\navne\PycharmProjects\AITesting\aitests\tests.json"
actualresults_dir = r"C:\Users\navne\PycharmProjects\AITesting\actualresults"
os.makedirs(actualresults_dir, exist_ok=True)

# Query Grok
def get_actual_response(question, context="", model="grok-4"):
    user_content = f"Based on the following context, answer the question accurately and concisely.\n\nContext: {context}\n\nQuestion: {question}" if context else question
    logging.info(f"Querying Grok with: {question} (context provided: {bool(context)})")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that answers questions accurately and concisely."},
            {"role": "user", "content": user_content}
        ],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Cosine similarity
def compute_cosine_similarity(text1, text2):
    embeddings = embed_model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

# Run tests
def run_tests():
    scores = []
    data = []
    report_data = []

    # Load test data from JSON
    if not os.path.exists(test_data_file):
        logging.error(f"Test data file not found: {test_data_file}")
        print(f"Test data file not found: {test_data_file}")
        return

    with open(test_data_file, 'r', encoding='utf-8') as tf:
        test_cases = json.load(tf)

    for test_case in test_cases:
        base = test_case.get("id", "unnamed")
        question = test_case.get("question", "")
        expected = test_case.get("ground_truth", "")
        context = test_case.get("context", "")

        if not question or not expected or not context:
            logging.warning(f"Incomplete test case for {base}")
            continue

        actual = get_actual_response(question, context=context)
        with open(os.path.join(actualresults_dir, base + '_actual.txt'), 'w', encoding='utf-8') as af:
            af.write(actual)

        print("\n============================")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Ground Truth: {expected}")
        print(f"Actual: {actual}")

        # Cosine similarity
        similarity = compute_cosine_similarity(expected, actual)
        print(f"Cosine Similarity: {similarity:.4f}")
        scores.append(similarity)

        hallucination_detected = "Yes" if similarity < 0.8 else "No"
        if hallucination_detected == "Yes":
            print("⚠️ Potential hallucination detected (low similarity to expected).")
            logging.warning("Potential hallucination detected.")
        else:
            print("No hallucination detected.")
            logging.info("No hallucination detected.")

        report_data.append({
            "ID": base,
            "Question": question,
            "Context": context,
            "Ground Truth": expected,
            "Actual Result": actual,
            "Cosine Similarity": f"{similarity:.4f}",
            "Hallucination Detected": hallucination_detected
        })

        # Infer sensitive group from test ID
        sensitive_group = None
        if "gender" in base:
            sensitive_group = "gender"
        elif "ethnic" in base:
            sensitive_group = "ethnicity"
        elif "age" in base:
            sensitive_group = "age"

        data.append({
            "question": question,
            "ground_truth": expected,
            "context": context,
            "sensitive_group": sensitive_group
        })

    if scores:
        avg_score = np.mean(scores)
        print(f"\nAverage Cosine Similarity Score: {avg_score:.4f}")
    else:
        print("No valid test cases.")

    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_html("test_report.html", index=False, escape=False)
        print("\nCustom test report saved to 'test_report.html'. Open it in a browser to view details.")

    # Run Giskard Scan
    if data:
        os.environ["XAI_API_KEY"] = api_key
        set_llm_model("xai/grok-4")

        def model_predict(df: pd.DataFrame):
            return [get_actual_response(row["question"], row.get("context", "")) for _, row in df.iterrows()]

        giskard_model = giskard.Model(
            model=model_predict,
            model_type="text_generation",
            name="General QA",
            description="AI assistant answering finance-related and stereotype-sensitive questions accurately and fairly.",
            feature_names=["question", "context", "sensitive_group"]
        )

        dataset = giskard.Dataset(
            pd.DataFrame(data),
            name="general_test_dataset",
            target="ground_truth",
            cat_columns=["sensitive_group"]
        )

        # Use giskard.scan() with detector tags to run hallucination and stereotypes scans
        scan_results = gsk.scan(
            giskard_model,
            dataset,
            only=[
                "llm_implausible_output",
                "llm_stereotypes_detector",
                "sycophancy",
                "information_disclosure",
                "output_format",
                "injection_attack",
                "performance_bias",
                "robustness"
            ],
            verbose=True
        )

        scan_results.to_html("giskard_scan_report.html")
        print("\nScan report saved to 'giskard_scan_report.html'. Open it in a browser to view details.")

        test_suite = scan_results.generate_test_suite("General Test Suite")
        test_suite_results = test_suite.run()
        print("\nTest Suite Results:")
        print(test_suite_results)

# Main
if __name__ == "__main__":
    run_tests()
