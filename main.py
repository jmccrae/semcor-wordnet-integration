import csv
import re
from bisect import bisect_right
from collections import defaultdict

import teanga
import wn

from teanga_split_token import split_token


def unescape_sense_key(s: str) -> str:
    """
    Unescape a sense key from OEWN
    """
    return (
        s.replace("-apos-", "'")
        .replace("-colon-", ":")
        .replace("-excl-", "!")
        .replace("-num-", "#")
        .replace("-dollar-", "$")
        .replace("-percnt-", "%")
        .replace("-amp-", "&")
        .replace("-lpar-", "(")
        .replace("-rpar-", ")")
        .replace("-ast-", "*")
        .replace("-plus-", "+")
        .replace("-comma-", ",")
        .replace("-sol-", "/")
        .replace("-lbrace-", "{")
        .replace("-vert-", "|")
        .replace("-rbrace-", "}")
        .replace("-tilde-", "~")
        .replace("-cent-", "¢")
        .replace("-pound-", "£")
        .replace("-sect-", "§")
        .replace("-copy-", "©")
        .replace("-reg-", "®")
        .replace("-deg-", "°")
        .replace("-acute-", "´")
        .replace("-para-", "¶")
        .replace("-ordm-", "º")
        .replace("--", "-")
    )


def build_ssid_index(wordnet):
    senseid2ssid = {}
    for sense in wordnet.senses():
        sense_id = sense.id[5:]
        lemma, key = sense_id.split("__")
        key = key.replace(".", ":")
        senseid2ssid[unescape_sense_key(lemma) + "%" + key] = sense.synset().id
    return senseid2ssid


def load_corpus(path: str) -> teanga.Corpus:
    return teanga.read_yaml(path)


def strip_for_match(text: str) -> str:
    # Discard all non-alphanumeric characters so SemCor tokenisation differences
    # (spaces around punctuation, split contractions, bracket spacing, etc.) are ignored.
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def expand_sense_key(raw_key: str) -> list[str]:
    # Corpus may store compound annotations ("key1;key2") and adjective marker "(a)"
    return [re.sub(r"\(a\)", "", part) for part in raw_key.split(";")]


def build_correction_map(csv_path: str) -> dict:
    corrections = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.reader(f):
            old_key = row[0]
            context = row[1]
            # Replacement may be in any column from 2 onwards; find the first non-empty one
            new_key_with_desc = next(
                (val.strip() for val in row[2:] if val.strip()), None
            )
            if new_key_with_desc:
                # Strip description (format is "sense_key : description text")
                new_key = new_key_with_desc.split(" : ")[0].strip()
                # Fix malformed keys that use dots instead of colons after the %
                # e.g. "simplify%2.30.00.." -> "simplify%2:30:00::"
                if "%" in new_key and ":" not in new_key.split("%", 1)[1]:
                    lemma, rest = new_key.split("%", 1)
                    new_key = lemma + "%" + rest.replace(".", ":")
                if (
                    context in corrections[old_key]
                    and new_key != corrections[old_key][context]
                ):
                    print(
                        f"Warning duplicate but different keys: {new_key} vs {corrections[old_key][context]}"
                    )
                corrections[old_key][context] = new_key
    return corrections


def fix_dotted_key(key: str) -> str:
    # Some replacement keys use dots instead of colons after the %, e.g.
    # "then%4.02.00.." -> "then%4:02:00::"
    if "%" in key and ":" not in key.split("%", 1)[1]:
        lemma, rest = key.split("%", 1)
        key = lemma + "%" + rest.replace(".", ":")
    return key


def wn30_key_forms(key: str) -> set:
    # Adjective satellites are stored in the corpus with ss_type 3, but this
    # CSV (like the sense-key corrections CSV) may give the ss_type 5 form of
    # the same key.
    forms = {key}
    if "%3" in key and "::" not in key:
        forms.add(key.replace("%3", "%5"))
    elif "%5" in key and "::" not in key:
        forms.add(key.replace("%5", "%3"))
    return forms


def apply_mwe_splits(corpus: teanga.Corpus, csv_path: str) -> teanga.Corpus:
    """
    Split single tokens that represent multi-word expressions (e.g. "and_then")
    into one token per word, distributing the per-word sense keys given in
    csv_path. Some corpus occurrences tag a multi-word sense onto a single,
    non-adjacent word (a discontinuous MWE, e.g. a lone "brought" tagged with
    "bring_together"); those can't be split, so we just correct the one key
    that applies to that word instead.
    """
    with open(csv_path) as f:
        rows = list(csv.reader(f))

    rows_by_key = defaultdict(list)
    for row in rows:
        rows_by_key[row[0]].append(row)

    docs = list(corpus.docs)

    # Phase 1: find which (doc, token index) each CSV row refers to, using the
    # original (unsplit) corpus so token indices stay stable while matching.
    planned = defaultdict(list)
    consumed = set()
    not_found = []

    for old_key, csv_rows in rows_by_key.items():
        forms = wn30_key_forms(old_key)
        for row in csv_rows:
            context = row[1]
            cols = row[2:6]
            last = max((i for i, v in enumerate(cols) if v != ""), default=-1)
            num_words = last + 1
            sub_keys = [
                "-" if v in ("-", "new sense") else fix_dotted_key(v)
                for v in cols[:num_words]
            ]
            stripped_ctx = strip_for_match(context)

            candidates = []
            for di, doc in enumerate(docs):
                txt = str(doc.text)
                tokens = list(doc.tokens)
                sent_offsets = list(doc.sentence)
                for idx, raw in doc.wn30_key:
                    if raw not in forms or (di, idx) in consumed:
                        continue
                    tok_start = tokens[idx][0]
                    sent_idx = bisect_right(sent_offsets, tok_start) - 1
                    sent_start = sent_offsets[sent_idx]
                    sent_end = (
                        sent_offsets[sent_idx + 1]
                        if sent_idx + 1 < len(sent_offsets)
                        else len(txt)
                    )
                    stripped_sent = strip_for_match(txt[sent_start:sent_end])
                    if stripped_ctx in stripped_sent:
                        candidates.append((di, idx, stripped_sent))

            exact = [c for c in candidates if c[2] == stripped_ctx]
            chosen = exact[0] if exact else (candidates[0] if candidates else None)

            if chosen is None:
                not_found.append((old_key, context))
                continue

            di, idx, _ = chosen
            consumed.add((di, idx))
            planned[di].append((idx, num_words, sub_keys))

    # Phase 2: apply the splits, highest token index first within each doc so
    # earlier (lower-index) splits aren't thrown off by index shifts caused by
    # later ones.
    split_count = 0
    substituted_count = 0
    for di, actions in planned.items():
        doc = docs[di]
        for idx, num_words, sub_keys in sorted(actions, key=lambda a: -a[0]):
            tokens = list(doc.tokens)
            txt = str(doc.text)
            tok_text = txt[tokens[idx][0] : tokens[idx][1]]

            parts = None
            for sep in ("_", "-"):
                if tok_text.count(sep) == num_words - 1:
                    parts = tok_text.split(sep)
                    break

            if parts is not None and len(parts) == num_words:
                split_token(idx, parts, sub_keys, doc)
                split_count += 1
            else:
                # A discontinuous MWE annotation on a single word: nothing to
                # split, so just correct the key for the word that's here.
                if sub_keys[0] != "-":
                    doc.wn30_key = [
                        (i, sub_keys[0] if i == idx else k) for i, k in doc.wn30_key
                    ]
                substituted_count += 1

    print(
        f"Split {split_count} tokens and corrected {substituted_count} "
        f"single-token MWE annotations out of {len(rows)} entries in "
        f"{csv_path} ({len(not_found)} not found)"
    )
    for old_key, context in not_found[:5]:
        print(f"Warning: no matching occurrence found for {old_key}: {context[:80]}")

    return corpus


def add_oewn2026_keys(corpus: teanga.Corpus, csv_path: str) -> teanga.Corpus:
    wordnet = wn.Wordnet("oewn:2025+")
    ssid_index = build_ssid_index(wordnet)
    corrections = build_correction_map(csv_path)
    corrections_set = set(
        (key, context) for key, contexts in corrections.items() for context in contexts
    )

    # Precompute stripped versions of every context string (keyed by sense key)
    stripped_corrections: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for key, ctx_map in corrections.items():
        for ctx, repl in ctx_map.items():
            if repl == "occur":
                repl = "be%2:30:14::"
            stripped_corrections[key][strip_for_match(ctx)] = (ctx, repl)

    corpus.add_layer_meta(
        "oewn2026_key", layer_type="element", base="tokens", data="string"
    )
    corrections_made = set()

    for doc in corpus.docs:
        txt = str(doc.text)
        tokens = list(doc.tokens)
        sent_offsets = list(doc.sentence)

        oewn2026 = []
        for idx, raw_key in doc.wn30_key:
            # Locate the sentence that contains this token
            tok_start = tokens[idx][0]
            sent_idx = bisect_right(sent_offsets, tok_start) - 1
            sent_start = sent_offsets[sent_idx]
            sent_end = (
                sent_offsets[sent_idx + 1]
                if sent_idx + 1 < len(sent_offsets)
                else len(txt)
            )
            stripped_sent = strip_for_match(txt[sent_start:sent_end])

            replaced = False
            fallback_key = None
            for subkey in expand_sense_key(raw_key):
                key_changed = False
                # Some keys in the corpus are stored with ss_type 3 (adjective satellite) but the CSV uses ss_type 5 (adjective).
                # we should fix this
                if "%3" in subkey and "::" not in subkey:
                    subkey = subkey.replace("%3", "%5")
                    key_changed = True

                lookup_key = subkey
                candidates = stripped_corrections[lookup_key]
                # Prefer an exact match (the CSV context is the whole sentence).
                # Only fall back to substring containment when no exact match
                # exists, and among substring matches pick the longest one, so
                # a short/generic context (e.g. "I asked .") can't hijack a
                # match ahead of a more specific, correct context for the
                # same sense key.
                match = candidates.get(stripped_sent)
                if match is None:
                    best_len = -1
                    for stripped_ctx, candidate in candidates.items():
                        if stripped_ctx in stripped_sent and len(stripped_ctx) > best_len:
                            match = candidate
                            best_len = len(stripped_ctx)
                if match is not None:
                    orig_ctx, repl = match
                    oewn2026.append([idx, repl])
                    corrections_made.add((lookup_key, orig_ctx))
                    replaced = True
                    break
                # Only use the raw %3->%5 conversion as a last resort, after every
                # subkey in a compound key has had a chance to find a real match.
                if key_changed and fallback_key is None:
                    fallback_key = subkey
            if not replaced and fallback_key is not None:
                oewn2026.append([idx, fallback_key])
                replaced = True
            if not replaced:
                oewn2026.append([idx, raw_key])

        ssid = []
        for idx, raw_key in oewn2026:
            for key in raw_key.split(";"):
                if key in ssid_index:
                    ssid.append([idx, ssid_index.get(key, None)])
                else:
                    print(f"Warning: sense key {key} not found in WordNet 2026 index")
        doc.oewn2026_key = oewn2026

    print(f"Made {len(corrections_made)} out of {len(corrections_set)} corrections")
    print(list(corrections_set.difference(corrections_made))[:5])

    return corpus


def main():
    corpus = load_corpus("data/semcor.yaml")
    print(f"Loaded corpus with {len(list(corpus.docs))} documents")

    corpus = apply_mwe_splits(corpus, "data/mwe2single.csv")

    corpus = add_oewn2026_keys(corpus, "data/semcor_updated_sense_keys.csv")

    total = sum(len(list(doc["wn30_key"])) for doc in corpus.docs)
    updated = 0
    for doc in corpus.docs:
        for (_, old_key), (_, new_key) in zip(doc["wn30_key"], doc["oewn2026_key"]):
            if old_key != new_key:
                updated += 1

    print(f"Updated {updated}/{total} sense key annotations")
    corpus.to_yaml("data/semcor_oewn2026.yaml")
    print("Written to data/semcor_oewn2026.yaml")


if __name__ == "__main__":
    main()
