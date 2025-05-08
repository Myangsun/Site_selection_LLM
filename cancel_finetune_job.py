# cancel_finetune_job.py

import requests
import os

# Replace with your actual OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")

# Replace with your actual fine-tuning job ID
FINE_TUNE_JOB_ID = "ftjob-WrUBw7WmwFniQW5hZG7liJTS"

# API endpoint
url = f"https://api.openai.com/v1/fine_tuning/jobs/{FINE_TUNE_JOB_ID}/cancel"

# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Send the cancel request
response = requests.post(url, headers=headers)

# Output
if response.status_code == 200:
    print("✅ Fine-tune job successfully canceled:")
    print(response.json())
else:
    print(
        f"❌ Failed to cancel. Status {response.status_code}: {response.text}")
