import csv
from retrieve import search_assessments

test_queries = [
    "Assessment for graduate roles requiring numerical reasoning",
    "Cognitive tests for problem solving and logical reasoning",
    "Assessments for entry-level analytical and finance roles"
]

output_file = "dhatri_dwivedi.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query", "rank", "assessment_name", "score"])

    for query in test_queries:
        results = search_assessments(query, top_k=3)

        for rank, res in enumerate(results, start=1):
            name_line = res["document"].splitlines()[0]
            name = name_line.replace("Name:", "").strip()

            writer.writerow([
                query,
                rank,
                name,
                round(res["score"], 3)
            ])

print(f"Predictions saved to {output_file}")
