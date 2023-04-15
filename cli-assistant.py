import asyncio
import json
import logging
import os
from typing import Dict

import openai
import torch
from openai import ChatCompletion
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger('assistant')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename='assistant.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# Load embeddings and transformers
with open('questions_answers.json', 'r') as file:
    qa_embeddings = json.load(file)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
openai.api_key = os.environ.get("OPENAI_API_KEY")

SIMILARITY_THRESHOLD = 0.90

async def find_best_question_match(prompt, questions):
    prompt_embedding = model.encode(prompt)
    question_embeddings = model.encode(questions)

    cos_sim = torch.nn.CosineSimilarity(dim=1)(torch.tensor(prompt_embedding).unsqueeze(0), torch.tensor(question_embeddings))
    max_sim, index = torch.max(cos_sim, dim=0)

    if len(questions) > 0 and index.item() < len(questions):
        return questions[index.item()], max_sim.item()
    else:
        return None, 0

async def get_more_context(question: str, answer: str, user_question: str, similarity: float) -> str:
    logger.info(f"Answer: {answer}")
    logger.info(f"Similarity: {similarity}")
    if question and answer and similarity >= SIMILARITY_THRESHOLD:
        chat_messages = [
            {"role": "system", "content": "You are a helpful assistant. Please respond only in Bulgarian"},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\n\nПредстави повече контекст към въпроса като включиш в отговора и линка от отговора"}
        ]
    else:
        chat_messages = [
            {"role": "system", "content": "You are a funny chatbot. Please respond only in Bulgarian"},
            {"role": "user", "content": user_question}
        ]

    response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].message['content'].strip()

def main():
    print("Type your questions in Bulgarian. Type '.exit' to quit the program.")
    while True:
        user_question = input("Your question: ")
        if user_question.strip() == ".exit":
            break

        loop = asyncio.get_event_loop()
        best_question, similarity = loop.run_until_complete(find_best_question_match(user_question, list(qa_embeddings.keys())))
        best_answer = qa_embeddings.get(best_question, "")
        context_answer = loop.run_until_complete(get_more_context(best_question, best_answer, user_question, similarity))
        print(f"Assistant: {context_answer}")

if __name__ == "__main__":
    main()
