# imports
import json
import re
import math

# functions
# data
def load_json(path: str) -> list[dict]:
    with open(path, 'r') as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def load_text_lines(path: str) -> list:
    lines = []
    with open(path, 'r') as f:
        for l in f:
            lines.append(l)
    return lines

# utils
def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', " ", text).strip()

def dedupe_chunks(chunks: list[str]) -> list[str]:
    seen = set()
    result = []
    for c in chunks:
        normalized = normalize_whitespace(c.lower())
        if normalized not in seen:
            seen.add(normalized)
            result.append(c)
    return result


# business logic
# input "the quick brown fox jumps over the lazy dog the fox"
# output [ "the quick", "brown fox", ]
def chunk_text(text: str, chunk_size: int = 2) -> list:
    tokens = text.split()
    token_list = []
    for i in range(0, len(tokens), chunk_size):
        token_list.append(tokens[i:i+chunk_size])
    print(token_list)
    chunk_list = [' '.join(tl) for tl in token_list]
    print(chunk_list)

# input "the quick brown fox jumps over the lazy dog the fox"
# output [ "the quick", "brown fox", ]
def chunk_text_sliding(text: str, chunk_size: int = 2, overlap: int = 1) -> list:
    tokens = text.split()
    step = chunk_size - overlap
    token_list = []
    for i in range(0, len(tokens), step):
        token_list.append(tokens[i:i+chunk_size])
    #print(token_list)
    chunk_list = [' '.join(tl) for tl in token_list]
    #print(chunk_list)
    return chunk_list


# return {k -> list of words}, example: {a -> [animal, ant]}
def group_words(words: list) -> dict:
    groups = {}
    for w in words:
        key = w[0]
        groups.setdefault(key, []).append(w)
    return groups


# input: the quick brown fox jumps over the lazy dog the fox
# output: [('the', 3), ('fox', 2), ('brown', 1), ('dog', 1), ('jumps', 1), ('lazy', 1), ('over', 1), ('quick', 1)
def sorted_count(text: str) -> list:
    groups = {}
    tokens = text.split()
    for token in tokens:
        key = token
        count = groups.get(key, 0)
        groups[key] = count + 1
    print(groups)
    sorted_list = sorted(groups.items(), key = lambda kv: -kv[1])
    print(sorted_list)


def search_documents(docs: list[str], query: str) -> list[str]:
    q = query.lower()
    return [d for d in docs if q in d.lower()]

# llm business functioins
def trim_messages(
    messages: list[dict],
    n: int = 2,
) -> list[dict]:
    """Keep all system messages plus the last n non-system messages,
    preserving original relative order."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    kept_others = other_msgs[-n:]
    return system_msgs + kept_others


def get_sample_messages():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(1, 13):
        role = "user" if i % 2 == 1 else "assistant"
        messages.append({"role": role, "content": f"msg {i}"})
    return messages


def main():
    # group words
    #words = ["apple", "banana", "avocado", "cherry", "blueberry"]
    #word_groups = group_words(words)
    #print(word_groups)

    # sort and count words
    #text = 'the quick brown fox jumps over the lazy dog the fox'
    #sorted_count(text)
    #jdata = load_json("data.json")
    #print(jdata)
    text = load_text('data.txt')
    print(text)
    text = normalize_whitespace(text)
    print(text)
    chunk_list = chunk_text_sliding(text, chunk_size = 4, overlap = 2)
    print(chunk_list)
    chunk_list = dedupe_chunks(chunk_list)
    print(chunk_list)
    docs = search_documents(chunk_list, 'fox jumps')
    print(docs)
    #lines = load_text_lines('data.txt')
    #print(lines)
    messages = get_sample_messages()
    print(messages)
    messages = trim_messages(messages)
    print(messages)

if __name__ == "__main__":
    main()
