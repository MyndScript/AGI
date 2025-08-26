import json
import sys

def dedupe_posts(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    seen = set()
    unique_posts = []
    for post in posts:
        content = post.get('content') if isinstance(post, dict) else None
        if content and content not in seen:
            seen.add(content)
            unique_posts.append(post)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_posts, f, ensure_ascii=False, indent=2)
    print(f"Deduped {len(posts)} -> {len(unique_posts)} posts. Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dedupe_posts.py <input_json> <output_json>")
    else:
        dedupe_posts(sys.argv[1], sys.argv[2])