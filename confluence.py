import os
import sys
import requests
import json
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CONFLUENCE_BASE_URL = os.getenv('CONFLUENCE_BASE_URL')
CONFLUENCE_USERNAME = os.getenv('CONFLUENCE_USERNAME')
CONFLUENCE_PASSWORD = os.getenv('CONFLUENCE_PASSWORD')

# Define a function to strip all characters except Cyrillic characters, Latin characters, and numbers
def strip_non_alpha_numeric(text):
    pattern = r'[^a-zA-Z0-9\u0400-\u04FF]+'
    return re.sub(pattern, ' ', text)

# Define a function to get all the titles, their links, and content in Confluence
def get_all_titles_links_and_content(base_url, username, password):
    start = 0
    limit = 50
    all_titles_links_and_content = {}

    while True:
        url = f"{base_url}/rest/api/content?start={start}&limit={limit}&expand=body.storage"
        response = requests.get(url, auth=(username, password))

        if response.status_code == 200:
            data = response.json()
            results = data.get("results")

            if not results:
                break

            for result in results:
                title = result.get("title")
                link = f"{base_url}{result['_links']['webui']}"
                storage_format_content = result['body']['storage']['value']
                soup = BeautifulSoup(storage_format_content, 'html.parser')
                plain_text_content = soup.get_text(separator=" ")
                stripped_content = strip_non_alpha_numeric(plain_text_content)
                answer = f"{title} - {link}"
                all_titles_links_and_content[stripped_content] = answer

            start += limit

        else:
            print(f"Error: {response.status_code}")
            break

    return all_titles_links_and_content

# Check if the output file name is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py output_file_name.json")
    sys.exit(1)

output_file = sys.argv[1]

# Get all the titles, their links, and content in Confluence
question_answer_pairs = get_all_titles_links_and_content(CONFLUENCE_BASE_URL, CONFLUENCE_USERNAME, CONFLUENCE_PASSWORD)

# Save the results to a JSON file
with open(output_file, "w", encoding="utf-8") as jsonfile:
    json.dump(question_answer_pairs, jsonfile, ensure_ascii=False, indent=2)

print(f"Results saved to {output_file}")
