
# autocorrector.py
# A simple autocorrector using NLP-style edits and word frequencies from a text corpus.
# Mirrors the "exact project" structure (steps: load, count frequency, probabilities, edit functions, candidates, best correction).
#
# Usage (local):
#   python autocorrector.py --dataset final.txt --k 3
#
# If you're using Google Colab, you can copy-paste the functions cell-by-cell into a notebook
# and use the upload widget to provide final.txt.

import argparse
import re
import string
import sys
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# -----------------------------
# Step 1: Optional Lemmatizer
# -----------------------------
lemmatizer = WordNetLemmatizer()

def lemmatize_word(word: str) -> str:
    """Lemmatize a given word using NLTK WordNet Lemmatizer."""
    return lemmatizer.lemmatize(word)

# -----------------------------
# Step 2: Load and process text dataset
# -----------------------------
WORD_RE = re.compile(r"\w+")  # sequences of alphanumeric characters

def words_from_text(text: str):
    # Lowercase, then extract tokens
    return WORD_RE.findall(text.lower())

def load_words_from_file(path: str):
    with open(path, 'r', encoding='utf8') as f:
        txt = f.read()
    return words_from_text(txt)

# -----------------------------
# Step 3: Count word frequency
# -----------------------------
def count_word_frequency(words):
    return Counter(words)

# -----------------------------
# Step 4: Calculate word probability
# -----------------------------
def calculate_probability(word_count: Counter):
    total = sum(word_count.values())
    # Avoid division by zero – if dataset is empty, return empty dict
    return {w: c / total for w, c in word_count.items()} if total else {}

# -----------------------------
# Step 5: Define edit functions
# -----------------------------
letters = string.ascii_lowercase

def delete_letter(word: str):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def swap_letters(word: str):
    return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word) - 1)]

def replace_letter(word: str):
    return [word[:i] + l + word[i+1:] for i in range(len(word)) for l in letters]

def insert_letter(word: str):
    return [word[:i] + l + word[i:] for i in range(len(word) + 1) for l in letters]

# -----------------------------
# Step 6: Generate candidate corrections
# -----------------------------
def generate_candidates(word: str):
    # One-edit variations
    cands = set()
    cands.update(delete_letter(word))
    cands.update(swap_letters(word))
    cands.update(replace_letter(word))
    cands.update(insert_letter(word))
    return cands

def generate_candidates_level2(word: str):
    # Two-edit variations (apply one edit to each level-1 candidate)
    level1 = generate_candidates(word)
    level2 = set()
    for w in level1:
        level2.update(generate_candidates(w))
    return level2

# -----------------------------
# Step 7: Get best corrections
# -----------------------------
def get_best_correction(word: str, probs: dict, vocab: set, max_suggestions: int = 3):
    # Prefer exact match; else level-1; else level-2
    if word in vocab:
        candidates = [word]
    else:
        level1 = generate_candidates(word).intersection(vocab)
        candidates = list(level1) if level1 else list(generate_candidates_level2(word).intersection(vocab))

    # Rank by probability (frequency in corpus)
    ranked = sorted(((w, probs.get(w, 0.0)) for w in candidates), key=lambda x: x[1], reverse=True)
    return [w for (w, _) in ranked[:max_suggestions]]

# -----------------------------
# CLI / Interactive loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Simple NLP autocorrector (edit-based) using a text corpus.")
    parser.add_argument('--dataset', '-d', default='final.txt', help='Path to the text dataset (e.g., final.txt)')
    parser.add_argument('--k', type=int, default=3, help='Number of suggestions to return')
    parser.add_argument('--lemmatize', action='store_true', help='Lemmatize input word before suggesting (optional)')
    args = parser.parse_args()

    try:
        words = load_words_from_file(args.dataset)
    except FileNotFoundError:
        print(f"[Error] Dataset not found: {args.dataset}")
        print("Create a 'final.txt' with lots of correctly spelled words (news articles, books, etc.) and rerun.")
        sys.exit(1)

    word_count = count_word_frequency(words)
    probabilities = calculate_probability(word_count)
    vocab = set(word_count.keys())

    if not vocab:
        print("[Error] Dataset appears to be empty after processing.")
        sys.exit(1)

    print("\nAutocorrector ready. Type a word to get suggestions. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input('Enter a word: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break

        if user_input in {'exit', 'quit'}:
            break

        if not re.fullmatch(r"[a-z]+", user_input):
            print("Please enter only alphabetic characters (a-z).\n")
            continue

        query = lemmatize_word(user_input) if args.lemmatize else user_input
        suggestions = get_best_correction(query, probabilities, vocab, max_suggestions=args.k)

        if suggestions:
            print("Top suggestions:", ", ".join(suggestions))
        else:
            print("(No suggestions found)")

    print("Goodbye!")

if __name__ == '__main__':
    main()

