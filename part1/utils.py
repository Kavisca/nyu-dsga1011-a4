import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    try:
        tokens = word_tokenize(example["text"])
    except LookupError:
        # The NLTK punkt resource is optional in this assignment; leave the text
        # unchanged rather than making evaluation fail if it was not downloaded.
        return example

    protected = {
        "a", "an", "the", "and", "or", "but", "if", "because", "while", "when",
        "not", "no", "never", "nothing", "neither", "nor", "hardly", "barely",
        "scarcely", "very", "more", "less", "most", "least", "did", "does", "do",
        "is", "are", "was", "were", "be", "been", "being",
    }

    qwerty_neighbors = {
        "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "erfcxs",
        "e": "wrds", "f": "rtgvcd", "g": "tyhbvf", "h": "yujnbg",
        "i": "uojk", "j": "uikhmn", "k": "iolmj", "l": "opk",
        "m": "njk", "n": "bhjm", "o": "ipkl", "p": "ol",
        "q": "wa", "r": "etfd", "s": "wedxz", "t": "ryfg",
        "u": "yhji", "v": "cfgb", "w": "qse", "x": "zsdc",
        "y": "tugh", "z": "asx",
    }

    def introduce_typo(word):
        letters = list(word.lower())
        positions = [i for i, char in enumerate(letters) if char in qwerty_neighbors]
        if not positions:
            return word
        pos = random.choice(positions)
        letters[pos] = random.choice(qwerty_neighbors[letters[pos]])
        result = "".join(letters)
        if word.isupper():
            return result.upper()
        if word and word[0].isupper():
            return result.capitalize()
        return result

    transformed = []
    for token in tokens:
        lower = token.lower()

        if (lower not in protected and lower.isalpha() and len(lower) >= 4
                and random.random() < 0.25):
            transformed.append(introduce_typo(token))
            continue

        if lower in protected or len(lower) < 4 or not lower.isalpha():
            transformed.append(token)
            continue

        if random.random() >= 0.35:
            transformed.append(token)
            continue

        try:
            synsets = wordnet.synsets(lower)
        except LookupError:
            # This can happen before wordnet.download('wordnet') is run.
            return example

        if not synsets:
            transformed.append(token)
            continue

        # Prefer the word's dominant WordNet part of speech. This prevents
        # replacements such as turning a noun into an incompatible verb/adjective.
        pos_counts = {}
        for synset in synsets:
            pos_counts[synset.pos()] = pos_counts.get(synset.pos(), 0) + 1
        dominant_pos = max(pos_counts, key=pos_counts.get)

        candidates = []
        for synset in synsets:
            if synset.pos() != dominant_pos:
                continue
            for lemma in synset.lemmas()[:3]:
                candidate = lemma.name().replace("_", " ").lower()
                if candidate.isalpha() and candidate != lower:
                    candidates.append(candidate)

        if candidates:
            replacement = random.choice(list(dict.fromkeys(candidates)))
            if token.isupper():
                replacement = replacement.upper()
            elif token[0].isupper():
                replacement = replacement.capitalize()
            transformed.append(replacement)
        else:
            transformed.append(token)

    example["text"] = TreebankWordDetokenizer().detokenize(transformed)

    ##### YOUR CODE ENDS HERE ######

    return example
