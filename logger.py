import json
from datetime import datetime

LOG_PATH = "data/review_logs.jsonl"
BANLIST_PATH = "data/banned_users.json"

def log_review(user_id, review_text, score, label):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "review": review_text,
        "score": score,
        "label": label
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_logs():
    try:
        with open(LOG_PATH, "r") as f:
            return [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        return []

def get_banned_users():
    try:
        with open(BANLIST_PATH, "r") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def ban_user(user_id):
    banned = get_banned_users()
    banned.add(user_id)
    with open(BANLIST_PATH, "w") as f:
        json.dump(list(banned), f)
