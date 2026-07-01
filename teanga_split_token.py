import teanga


def split_token(
    idx: int, lemmas: list[str], wn30_keys: list[str], doc: teanga.Document
):
    """
    Split a token in a teanga document into multiple tokens.

    Args:
        idx (int): The index of the token to split.
        lemmas: A list of lemmas corresponding to the new tokens.
        doc (teanga.Document): The teanga document containing the token to split.
    """
    assert len(lemmas) == len(wn30_keys), (
        "The number of lemmas must match the number of wn30_keys."
    )
    # Get the original token's start and end offsets
    tokens = [tk for tk in doc.tokens]

    new_tokens = []
    lemma_idx = 0
    for lemma in lemmas:
        # Create a new token with the same start and end offsets as the original token
        new_tokens.append((lemma_idx, lemma_idx + len(lemma)))
        lemma_idx += len(lemma) + 1  # Assuming a space between tokens

    # Remove this token and splice it into the list
    tk_start, tk_end = tokens.pop(idx)
    for offset, (start, end) in enumerate(new_tokens):
        tokens.insert(idx + offset, (tk_start + start, tk_start + end))

    doc.tokens = tokens

    # Update the lemmas
    new_lemmas = []
    for offset, lemma in enumerate(doc.lemmas):
        if offset < idx:
            new_lemmas.append(lemma)
        elif offset == idx:
            new_lemmas.extend(lemmas)
        else:
            new_lemmas.append(lemma)
    doc.lemmas = new_lemmas

    # Update the keys. wn30_key is a sparse layer (not every token has an
    # entry), so entries must be matched by their own stored index, not by
    # their position in the list.
    new_keys = []
    for key_idx, key_value in doc.wn30_key:
        if key_idx < idx:
            new_keys.append((key_idx, key_value))
        elif key_idx == idx:
            # Skip annotations that are not relevant to the new tokens (e.g., "-")
            new_keys.extend(
                (idx + i, wn30_key)
                for i, wn30_key in enumerate(wn30_keys)
                if wn30_key != "-"
            )
        else:
            new_keys.append((key_idx + len(lemmas) - 1, key_value))
    doc.wn30_key = new_keys

    # Other fields dependent on this will need to have their offsets updated.
    for field in ["pos", "wn16_key"]:
        if hasattr(doc, field):
            field_values = list(getattr(doc, field))
            if doc._meta[field].layer_type == "seq":
                # A "seq" layer needs exactly one entry per token, so the
                # original value is duplicated across the new tokens (there
                # is no per-word data available to split it properly).
                # Entries may come back as plain values or as (index, value)
                # pairs depending on how the layer was last stored, so
                # normalise to plain values first.
                plain_values = [
                    v[1] if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], int)
                    else v
                    for v in field_values
                ]
                new_values = []
                for offset, value in enumerate(plain_values):
                    if offset < idx:
                        new_values.append(value)
                    elif offset == idx:
                        new_values.extend([value] * len(lemmas))
                    else:
                        new_values.append(value)
                setattr(doc, field, new_values)
            else:
                # A sparse "element" layer: drop the annotation at the split
                # position and shift indices for everything after it.
                new_values = []
                for i, value in field_values:
                    if i < idx:
                        new_values.append((i, value))
                    elif i > idx:
                        new_values.append((i + len(lemmas) - 1, value))
                setattr(doc, field, new_values)


if __name__ == "__main__":
    # Example usage
    corpus = teanga.Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta("lemmas", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta(
        "wn30_key", layer_type="element", base="tokens", data="string"
    )
    doc = corpus.add_doc("Watson snorted and_then laughed aloud.")
    doc.tokens = [(0, 6), (7, 14), (15, 23), (24, 30), (31, 36)]
    doc.lemmas = ["Watson", "snort", "and_then", "laugh", "aloud"]
    doc.wn30_key = [
        (0, "Watson%1:18:00::"),
        (1, "snort%2:38:00::"),
        (2, "and_then%4:02:00::"),
        (3, "laugh%2:38:00::"),
        (4, "aloud%4:02:00::"),
    ]

    print("Before split:")
    print("Tokens:", doc.tokens)
    print("Lemmas:", doc.lemmas)
    print("WN30 Keys:", doc.wn30_key)

    # Split the token at index 4 ("jumps") into two tokens ("jump" and "s")
    split_token(2, ["and", "then"], ["-", "then%4:02:00::"], doc)

    print("\nAfter split:")
    print("Tokens:", doc.tokens)
    print("Lemmas:", doc.lemmas)
    print("WN30 Keys:", doc.wn30_key)
