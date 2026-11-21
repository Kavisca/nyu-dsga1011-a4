# DS-GA 1011: Homework 4 Part 1

> ### Please start early.

## Set up environment
We recommend you set up a conda environment for the packages used in this part of the homework.
```bash
conda create -n dsga1011-hw4-part1 python=3.11
conda activate dsga1011-hw4-part1
pip install -r requirements.txt
```

After this, you will need to install certain packages in nltk.
```bash
python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('punkt')
>>> exit()
```

## Data transformation

`custom_transform` applies two moderate transformations to each review:

1. It tokenizes the text with NLTK `word_tokenize`.
2. A word is eligible if it is purely alphabetic, has at least four characters, and is not a protected word.
3. Each eligible word receives one adjacent-keyboard typo with probability 25%: one character is replaced by a random QWERTY neighbor, preserving capitalization.
4. If no typo was introduced, the word is replaced by a WordNet synonym with probability 35%. Candidates come from the word's dominant part of speech (first three lemmas per synset); one is chosen at random and capitalization is preserved.
5. Protected words (articles, conjunctions, negations such as not/no/never, forms of be and do, and intensifiers such as very/more/most) are never changed. The text is reassembled with NLTK's `TreebankWordDetokenizer`.

Both operations preserve the original sentiment label. The augmented training loader deterministically shuffles the original training split with seed 42, selects 5,000 reviews, applies the same transformation, and concatenates them with the original 25,000 training examples.

This is reasonable because people naturally vary word choice and occasionally make small typing mistakes. The synonym operation is randomized and limited to one character change in the typo operation, while negations and structural words remain unchanged, so the intended sentiment should normally remain the same.
