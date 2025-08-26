import json
from bs4 import BeautifulSoup

input_path = r"C:\Users\Ommi\Desktop\Data\your_facebook_activity\posts\your_posts__check_ins__photos_and_videos_1.html"
output_path = r"C:\Users\Ommi\Desktop\Data\your_facebook_activity\posts\extracted_posts.json"

def extract_posts_and_checkins(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    results = []
    # Facebook HTML exports often use <div> or <li> for each post/check-in
    for post in soup.find_all(["div", "li"]):
        text = post.get_text(strip=True)
        # Heuristic: look for keywords to identify posts/check-ins
        if "checked in at" in text or "Check-in" in text:
            results.append({"type": "check-in", "content": text})
        elif text:
            results.append({"type": "post", "content": text})

    return results

if __name__ == "__main__":
    data = extract_posts_and_checkins(input_path)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
    print(f"Extracted {len(data)} items to {output_path}")