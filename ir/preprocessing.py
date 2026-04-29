import re

# Common English stopwords plus programming-specific noise terms
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "this", "that", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "i", "you", "he", "she", "we", "they", "my", "your", "our", "their",
    "as", "not", "no", "so", "if", "then", "than", "also", "into", "about",
    "use", "used", "using", "new", "make", "makes", "made", "get", "just",
    "more", "one", "all", "any", "some", "when", "what", "how", "which",
    "here", "there", "like", "well", "need", "want", "great", "good",
    "simple", "easy", "fast", "based", "built", "build", "work", "works",
    "project", "projects", "repo", "repository", "code", "github",
    "support", "supports", "include", "includes", "including", "other",
}


def preprocess(text):
    """
    Normalize text for TF-IDF:
      1. Lowercase
      2. Replace non-alphanumeric chars with spaces
      3. Tokenize on whitespace
      4. Drop stopwords and single-char tokens
    Returns a single whitespace-joined string (scikit-learn input format).
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
