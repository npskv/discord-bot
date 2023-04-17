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

print("Loading embeddings and transformers...")
with open('kb_simple.json', 'r') as file:
    qa_data = json.load(file)

model = SentenceTransformer('sentence-transformers/LaBSE')
openai.api_key = os.environ.get("OPENAI_API_KEY")

SIMILARITY_THRESHOLD = 0.75

print("Defining helper functions...")

async def precompute_embeddings():
    qa_embeddings = {}
    for question, answer in qa_data.items():
        question_embedding = model.encode(question, convert_to_tensor=True)
        qa_embeddings[question] = {
            'answer': answer,
            'embedding': question_embedding
        }
    return qa_embeddings

print("Precomputing embeddings...")
qa_embeddings = asyncio.run(precompute_embeddings())

async def find_best_question_match(prompt, embeddings):
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)

    questions = list(embeddings.keys())
    question_embeddings = torch.stack([embeddings[q]['embedding'] for q in questions])

    cos_sim = util.pytorch_cos_sim(prompt_embedding, question_embeddings)
    max_sim, index = torch.max(cos_sim, dim=1)

    if len(questions) > 0 and index.tolist()[0] < len(questions):
        return questions[index.tolist()[0]], max_sim.tolist()[0]
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

async def process_question(user_question):
    best_question, similarity = await find_best_question_match(user_question, qa_embeddings)
    best_answer = qa_embeddings.get(best_question, {}).get('answer', "")
    context_answer = await get_more_context(best_question, best_answer, user_question, similarity)
    return context_answer

async def main():
    print("Type your questions in Bulgarian. Type '.exit' to quit the program.")
    
    while True:
        user_question = input("Your question: ")
        if user_question.strip() == ".exit":
            break

        context_answer = await process_question(user_question)
        print(f"Assistant: {context_answer}")
        print()

if __name__ == "__main__":
    asyncio.run(main())


