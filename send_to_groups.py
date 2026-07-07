import sys
import io
import requests

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BOT_TOKEN = "8504828186:AAHcEo1ABoRmWukO4YXWCIjlE8jhpaVEs8Y"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

MESSAGE = (
    "WARNING: If your Telegram display name contains "
    "< or > symbols, please reply here immediately. "
    "It is causing an issue with the auction bot."
)

# --- ADD YOUR GROUP CHAT IDs HERE ---
# To find a group chat ID: forward any group message to @userinfobot
# Group IDs are negative numbers like -1001234567890
GROUP_IDS = [
    -1004499270065,  # CWL [DEN] (found earlier)
    # Add more group IDs below:
    # -1009999999999,
]

print(f"Sending to {len(GROUP_IDS)} group(s)...\n")

for chat_id in GROUP_IDS:
    resp = requests.post(f"{BASE_URL}/sendMessage", data={
        "chat_id": chat_id,
        "text": MESSAGE
    })
    result = resp.json()
    if result.get("ok"):
        print(f"OK - Sent to {chat_id}")
    else:
        print(f"FAILED - {chat_id}: {result.get('description')}")

print("\nDone!")
