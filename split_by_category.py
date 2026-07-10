import csv
import os
from collections import defaultdict

import teanga


def load_doc_genres(csv_path: str) -> dict[str, str]:
    with open(csv_path) as f:
        return {row["doc_id"]: row["genre"] for row in csv.DictReader(f)}


def main():
    doc_genres = load_doc_genres("data/doc_categories.csv")
    corpus = teanga.read_yaml("data/semcor_oewn2026.yaml")

    doc_ids_by_genre = defaultdict(list)
    for doc_id in corpus.doc_ids:
        doc_ids_by_genre[doc_genres[doc_id]].append(doc_id)

    os.makedirs("data/by_category", exist_ok=True)
    for genre, doc_ids in sorted(doc_ids_by_genre.items()):
        out_path = f"data/by_category/{genre}.yaml"
        corpus.subset(doc_ids).to_yaml(out_path)
        print(f"{genre}: {len(doc_ids)} docs -> {out_path}")


if __name__ == "__main__":
    main()
