import json
import re

INPUT_PATH = "extracted_posts.json"
OUTPUT_PATH = "filtered_posts.json"

# Heuristics for fluff removal
URL_PATTERN = re.compile(r"^https?://")
DATE_PATTERN = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4}|am|pm|\d{1,2}:\d{2})\b", re.I)
GENERIC_PHRASES = ["timeline photos", "updated", "photo", "video", "check-in", "joined facebook"]


def is_fluff(post):
    content = post.get("content", "").strip().lower()
    # Remove if only a URL
    if URL_PATTERN.match(content):
        return True
    # Remove if only a date/time
    if len(content) < 40 and DATE_PATTERN.search(content):
        return True
    # Remove if generic phrase
    for phrase in GENERIC_PHRASES:
        if phrase in content:
            return True
    # Remove if very short
    if len(content) < 10:
        return True
    return False


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [post for post in data if not is_fluff(post)]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(filtered, out, ensure_ascii=False, indent=2)
    print(f"Filtered {len(filtered)} posts written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
