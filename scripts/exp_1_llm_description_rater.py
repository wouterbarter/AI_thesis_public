import json
import time
import random
from typing import Dict
from pydantic import BaseModel, ValidationError
from ollama import chat

import os

# Always work relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


class Rating(BaseModel):
    Rating: int


def rate_online(descriptions: Dict[str, str], prompt: str, model_name='llama3.2', sleep=1):
    """
    Continuously rates descriptions using the LLM and stores the output.
    """
    results = []  # Store all ratings over time

    while True:
        company_id = random.choice(list(descriptions.keys()))
        desc = descriptions[company_id]

        try:
            response = chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': f"{prompt}\nDescription: {desc}\nReturn the rating in JSON format."
                    }
                ],
                format=Rating.model_json_schema(),
                options={'temperature': 0.5}  # Tuneable
            )

            rating_obj = Rating.model_validate_json(response.message.content)

            result = {
                "company_id": company_id,
                "description": desc,
                "rating": rating_obj.Rating,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }

            results.append(result)
            print(
                f"[✓] {result['timestamp']} | {company_id} → {rating_obj.Rating}")

            # Optional: Append to file immediately (online logging)
            with open("ratings_log.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

        except (ValidationError, Exception) as e:
            print(f"[!] Error on {company_id}: {e}")

        time.sleep(sleep)  # Control the pace to avoid flooding or throttling


prompt = """You are a rating assistant. The following are descriptions that companies provided of their company. Rate the following descriptions on a scale of 1 to 5 based on how well they encompass what the company does, in general.
            Rating Criteria, and provide a reason why you gave the rating:
            - 5: Highly specific and very useful, clearly describes what the company is and does.
            - 4: Mostly relevant and mostly useful, but could still be improved.
            - 3: Somewhat relevant and somewhat useful. 
            - 2: Weakly relevant, but not useful to get an idea about the company.
            - 1: Not relevant to the company at all."""

# Run the loop


descriptions = read_json('data/ids_descriptions_noduplicates.json')


rate_online(descriptions, prompt)
