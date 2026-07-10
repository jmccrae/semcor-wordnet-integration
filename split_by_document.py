import csv
import os
from bisect import bisect_left, bisect_right

import teanga


def load_doc_info(csv_path: str) -> dict[str, tuple[str, str]]:
    with open(csv_path) as f:
        return {
            row["doc_id"]: (row["brown_id"], row["genre"])
            for row in csv.DictReader(f)
        }


def new_sentence_corpus() -> teanga.Corpus:
    corpus = teanga.Corpus()
    corpus.add_layer_meta("text", layer_type="characters")
    corpus.add_layer_meta("paragraph", layer_type="characters")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta("lemmas", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta("pos", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta("wn16_key", layer_type="element", base="tokens", data="string")
    corpus.add_layer_meta("wn30_key", layer_type="element", base="tokens", data="string")
    corpus.add_layer_meta(
        "oewn2026_key", layer_type="element", base="tokens", data="string"
    )
    return corpus


def normalize_seq(values: list) -> list:
    # A "seq" layer is one-to-one with its base (tokens), so it never needs
    # explicit indices, but teanga sometimes round-trips it as (idx, value)
    # pairs carrying the layer's original absolute token indices rather than
    # plain values (see the same workaround in teanga_split_token.py). Strip
    # those back down to plain values so slicing produces a plain list with
    # no leftover absolute indices.
    return [
        v[1] if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], int)
        else v
        for v in values
    ]


def split_doc_into_sentences(doc, out_corpus: teanga.Corpus) -> None:
    text = str(doc.text)
    tokens = list(doc.tokens)
    lemmas = normalize_seq(list(doc.lemmas))
    pos = normalize_seq(list(doc.pos))
    wn16_key = list(doc.wn16_key)
    wn30_key = list(doc.wn30_key)
    oewn2026_key = list(doc.oewn2026_key)
    sentence_offsets = list(doc.sentence)
    paragraph_offsets = list(doc.paragraph)
    token_starts = [t[0] for t in tokens]

    def remap(layer, i0, i1):
        return [[idx - i0, val] for idx, val in layer if i0 <= idx < i1]

    for si, start in enumerate(sentence_offsets):
        end = (
            sentence_offsets[si + 1]
            if si + 1 < len(sentence_offsets)
            else len(text)
        )
        para_idx = bisect_right(paragraph_offsets, start) - 1

        i0 = bisect_left(token_starts, start)
        i1 = bisect_left(token_starts, end)

        sent_doc = out_corpus.add_doc(text=text[start:end], paragraph=str(para_idx))
        sent_doc.tokens = [(s - start, e - start) for s, e in tokens[i0:i1]]
        sent_doc.lemmas = lemmas[i0:i1]
        sent_doc.pos = pos[i0:i1]
        sent_doc.wn16_key = remap(wn16_key, i0, i1)
        sent_doc.wn30_key = remap(wn30_key, i0, i1)
        sent_doc.oewn2026_key = remap(oewn2026_key, i0, i1)


def main():
    doc_info = load_doc_info("data/doc_categories.csv")
    corpus = teanga.read_yaml("data/semcor_oewn2026.yaml")

    for doc_id in corpus.doc_ids:
        brown_id, genre = doc_info[doc_id]
        out_dir = f"data/by_document/{genre}"
        os.makedirs(out_dir, exist_ok=True)

        out_corpus = new_sentence_corpus()
        split_doc_into_sentences(corpus.doc_by_id(doc_id), out_corpus)
        out_corpus.to_yaml(f"{out_dir}/{brown_id}.yaml")

    print(f"Wrote {len(list(corpus.doc_ids))} files to data/by_document/<genre>/")


if __name__ == "__main__":
    main()
