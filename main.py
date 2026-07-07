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
    # A (old_key, context) pair can legitimately appear more than once: the
    # same sentence can contain two occurrences of the same old sense key
    # that need different replacements (e.g. "asked" tagged ask%2:32:00::
    # twice in one sentence, meaning two different senses). Rows are stored
    # in CSV order and handed out one-per-occurrence, in that order, as
    # matching occurrences are found in the corpus.
    corrections = defaultdict(lambda: defaultdict(list))
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
                corrections[old_key][context].append(new_key)
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


def resolve_fix_key(value: str) -> str:
    # "-" / "new sense" mark a word that gets no key; "occur" is shorthand
    # (also used in semcor_updated_sense_keys.csv) for the "happen" sense of "be".
    value = value.strip()
    if value in ("-", "new sense"):
        return "-"
    if value == "occur":
        return "be%2:30:14::"
    return fix_dotted_key(value)


def apply_indexed_fixes(corpus: teanga.Corpus, csv_path: str) -> teanga.Corpus:
    """
    Apply sense-key fixes located by exact (doc_id, annotation_index) instead
    of the fuzzy context matching apply_mwe_splits/add_oewn2026_keys use.
    annotation_index is a token index in the corpus as it stands *after*
    apply_mwe_splits has run, so this must be called afterwards. Rows with a
    single fix column correct (or add) the key on that one token; rows with
    more than one split the token the same way apply_mwe_splits does.
    """
    with open(csv_path) as f:
        rows = list(csv.reader(f))[1:]  # skip header

    docs_by_id = {doc_id: doc for doc_id, doc in zip(corpus.doc_ids, corpus.docs)}

    actions_by_doc = defaultdict(list)
    skipped = 0
    for row in rows:
        if len(row) < 4 or not row[2] or not row[3]:
            continue
        doc_id, idx = row[2], int(row[3])
        if doc_id not in docs_by_id:
            skipped += 1
            continue
        cols = row[4:6]
        last = max((i for i, v in enumerate(cols) if v.strip()), default=-1)
        if last == -1:
            continue
        num_words = last + 1
        sub_keys = [resolve_fix_key(v) for v in cols[:num_words]]
        actions_by_doc[doc_id].append((idx, num_words, sub_keys))

    split_count = 0
    substituted_count = 0
    for doc_id, actions in actions_by_doc.items():
        doc = docs_by_id[doc_id]
        # Highest token index first so earlier splits in the same doc aren't
        # thrown off by index shifts caused by later ones.
        for idx, num_words, sub_keys in sorted(actions, key=lambda a: -a[0]):
            tokens = list(doc.tokens)
            txt = str(doc.text)
            tok_text = txt[tokens[idx][0] : tokens[idx][1]]

            parts = None
            if num_words > 1:
                for sep in ("_", "-"):
                    if tok_text.count(sep) == num_words - 1:
                        parts = tok_text.split(sep)
                        break

            if parts is not None and len(parts) == num_words:
                split_token(idx, parts, sub_keys, doc)
                split_count += 1
            else:
                if sub_keys[0] != "-":
                    keys = dict(doc.wn30_key)
                    keys[idx] = sub_keys[0]
                    doc.wn30_key = sorted(keys.items())
                substituted_count += 1

    print(
        f"Applied indexed fixes: split {split_count} tokens and corrected "
        f"{substituted_count} single-token annotations out of {len(rows)} "
        f"entries in {csv_path} ({skipped} skipped, doc not found)"
    )

    return corpus


def load_new_key_ssids(csv_path: str) -> dict:
    # Sense keys introduced in OEWN 2026 that don't exist in oewn:2025+, so
    # build_ssid_index can't find them. Format: sense_key,synset_id
    new_keys = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if row and row[0]:
                new_keys[row[0]] = row[1]
    return new_keys


def add_oewn2026_keys(
    corpus: teanga.Corpus, csv_path: str, new_keys_csv_path: str
) -> teanga.Corpus:
    wordnet = wn.Wordnet("oewn:2025+")
    ssid_index = build_ssid_index(wordnet)
    ssid_index.update(load_new_key_ssids(new_keys_csv_path))
    corrections = build_correction_map(csv_path)
    corrections_set = set(
        (key, context) for key, contexts in corrections.items() for context in contexts
    )

    # Precompute stripped versions of every context string (keyed by sense key).
    # Each context maps to a *list* of replacements, one per occurrence of the
    # sense key within that sentence, consumed in order (see build_correction_map).
    stripped_corrections: dict[str, dict[str, tuple[str, list[str]]]] = defaultdict(dict)
    for key, ctx_map in corrections.items():
        for ctx, repls in ctx_map.items():
            resolved = ["be%2:30:14::" if r == "occur" else r for r in repls]
            stripped_corrections[key][strip_for_match(ctx)] = (ctx, resolved)

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
                exact = candidates.get(stripped_sent)
                match = exact if exact is not None and exact[1] else None
                if match is None:
                    best_len = -1
                    for stripped_ctx, candidate in candidates.items():
                        if not candidate[1]:
                            continue
                        if stripped_ctx in stripped_sent and len(stripped_ctx) > best_len:
                            match = candidate
                            best_len = len(stripped_ctx)
                if match is not None:
                    orig_ctx, repls = match
                    # Consume one replacement per occurrence, in CSV row
                    # order, so repeated occurrences of the same sense key
                    # within one sentence get their own distinct replacement.
                    repl = repls.pop(0)
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
            resolved_ssids = []
            for key in raw_key.split(";"):
                if key in ssid_index:
                    resolved_ssids.append(ssid_index[key])
                else:
                    print(f"Warning: sense key {key} not found in WordNet 2026 index")
            if resolved_ssids:
                ssid.append([idx, ";".join(resolved_ssids)])
        doc.oewn2026_key = ssid

    print(f"Made {len(corrections_made)} out of {len(corrections_set)} corrections")
    print(list(corrections_set.difference(corrections_made))[:5])

    return corpus


def main():
    corpus = load_corpus("data/semcor.yaml")
    print(f"Loaded corpus with {len(list(corpus.docs))} documents")

    corpus = apply_mwe_splits(corpus, "data/mwe2single.csv")

    corpus = apply_indexed_fixes(corpus, "data/missing_sense_keys.csv")

    corpus = add_oewn2026_keys(
        corpus, "data/semcor_updated_sense_keys.csv", "data/2026_new_keys.csv"
    )

    # oewn2026_key is sparse (unresolvable keys get no entry), so compare by
    # token index rather than assuming the two layers line up positionally.
    total = sum(len(list(doc["wn30_key"])) for doc in corpus.docs)
    resolved = sum(len(list(doc["oewn2026_key"])) for doc in corpus.docs)

    print(f"Resolved {resolved}/{total} sense key annotations to OEWN 2026 synsets")
    corpus.to_yaml("data/semcor_oewn2026.yaml")
    print("Written to data/semcor_oewn2026.yaml")


if __name__ == "__main__":
    main()
