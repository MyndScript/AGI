import json
from transformers import pipeline
from sentence_transformers import SentenceTransformer

INPUT_PATH = "extracted_posts.json"
OUTPUT_PATH = "digested_posts.json"

# Load custom or best available models
# Upgraded models for better performance and quality
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
intent_classifier = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", top_k=1)

# Upgraded embedding model: Better quality (768 dims vs 384)
# For production, consider: BAAI/bge-large-en-v1.5 (1024 dims, state-of-the-art)
semantic_model = SentenceTransformer("all-mpnet-base-v2")  # Upgraded from all-MiniLM-L6-v2

def digest_post(content):
    # Truncate content for classifier models (max 512 chars)
    truncated_content = content[:512]
    # Semantic embedding (use full content)
    embedding = semantic_model.encode(content).tolist()
    # Emotion
    emotion_result = emotion_classifier(truncated_content)
    intent_result = intent_classifier(truncated_content)
    def extract_first_dict(result):
        # Handles [dict], [[dict]], or [] cases
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return first
            elif isinstance(first, list) and first:
                return first[0]
        return {"label": None, "score": None}

    emotion = extract_first_dict(emotion_result)
    intent = extract_first_dict(intent_result)
    return {
        "content": content,
        "embedding": embedding,
        "emotion": emotion.get("label"),
        "emotion_score": emotion.get("score"),
        "intent": intent.get("label"),
        "intent_score": intent.get("score")
    }

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    digested = []
    for item in data:
        # Only process dicts with a non-empty 'content' field
        if isinstance(item, dict) and item.get("content"):
            content = item["content"].strip()
            if content:
                result = digest_post(content)
                # Validate result has all expected fields
                if all(k in result for k in ["content", "embedding", "emotion", "emotion_score", "intent", "intent_score"]):
                    digested.append(result)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(digested, out, ensure_ascii=False, indent=2)
    print(f"Digested {len(digested)} posts written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
