import csv
import re
import xml.etree.ElementTree as ET

import teanga

# Standard Brown Corpus genre codes, as used in SemCor's br-<letter><nn> document ids.
BROWN_GENRES = {
    "a": "press_reportage",
    "b": "press_editorial",
    "c": "press_reviews",
    "d": "religion",
    "e": "skill_and_hobbies",
    "f": "popular_lore",
    "g": "belles_lettres",
    "h": "miscellaneous",
    "j": "learned",
    "k": "fiction_general",
    "l": "fiction_mystery",
    "m": "fiction_science",
    "n": "fiction_adventure",
    "p": "fiction_romance",
    "r": "humor",
}


def strip_for_match(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def brown_doc_order(xml_path: str) -> list[tuple[str, str]]:
    """Return [(brown_id, first-20-words-stripped), ...] in document order."""
    order = []
    words: list[str] = []
    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        if event == "start" and elem.tag == "document":
            words = []
        elif event == "end" and elem.tag in ("word", "wf"):
            surface = elem.attrib.get("surface_form") or elem.text
            if surface:
                words.append(surface)
        elif event == "end" and elem.tag == "document":
            order.append((elem.attrib["id"], strip_for_match(" ".join(words[:20]))))
            elem.clear()
    return order


def main():
    # SemCor's YAML conversion drops the original Brown document ids, so we
    # recover them positionally from the UFSAC source XML: both list documents
    # in the same order, which we verify by comparing each document's opening
    # words before trusting the alignment.
    xml_path = "/home/john-mccrae/p/ufsac-public-2.1/semcor.xml"
    brown_order = brown_doc_order(xml_path)

    corpus = teanga.read_yaml("data/semcor_oewn2026.yaml")
    doc_ids = list(corpus.doc_ids)

    if len(doc_ids) != len(brown_order):
        raise SystemExit(
            f"Document count mismatch: corpus has {len(doc_ids)}, "
            f"XML has {len(brown_order)}"
        )

    rows = []
    for doc_id, (brown_id, xml_snippet) in zip(doc_ids, brown_order):
        doc = corpus.doc_by_id(doc_id)
        corpus_snippet = strip_for_match(str(doc.text)[:80])
        if not (
            corpus_snippet.startswith(xml_snippet[:30])
            or xml_snippet.startswith(corpus_snippet[:30])
        ):
            raise SystemExit(
                f"Alignment check failed for {doc_id} / {brown_id}: "
                f"{corpus_snippet[:30]!r} vs {xml_snippet[:30]!r}"
            )
        letter = brown_id.split("-")[1][0]
        rows.append([doc_id, brown_id, letter, BROWN_GENRES[letter]])

    with open("data/doc_categories.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "brown_id", "category_letter", "genre"])
        writer.writerows(rows)

    print(f"Wrote data/doc_categories.csv with {len(rows)} documents")


if __name__ == "__main__":
    main()
