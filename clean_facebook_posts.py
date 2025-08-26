import json
import re

INPUT_PATH = "extracted_posts.json"
OUTPUT_PATH = "cleaned_posts.json"

def is_meaningful_post(item):
    if item.get("type") != "post":
        return False
    content = item.get("content", "").strip()
    # Filter out empty, very short, or non-informative posts
    return bool(content) and len(content) > 10 and not content.lower().startswith("checked in at")

def is_valid_checkin(item):
    if item.get("type") != "check-in":
        return False
    content = item.get("content", "").strip()
    # Look for location and date info heuristically
    has_location = bool(re.search(r"checked in at [A-Za-z0-9, .]+", content))
    has_date = bool(re.search(r"\d{4}|January|February|March|April|May|June|July|August|September|October|November|December", content))
    return has_location and has_date

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = []
    for item in data:
        if is_meaningful_post(item):
            cleaned.append({"type": "post", "content": item["content"]})
        elif is_valid_checkin(item):
            cleaned.append({"type": "check-in", "content": item["content"]})
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(cleaned, out, ensure_ascii=False, indent=2)
    print(f"Cleaned {len(cleaned)} items written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
