import argparse
import asyncio
import json
import os
import pickle
from typing import Dict

import openai
import torch
from flask import Flask, render_template, request, jsonify
from openai import ChatCompletion
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = SentenceTransformer('sentence-transformers/LaBSE')
openai.api_key = os.environ.get("OPENAI_API_KEY")
SIMILARITY_THRESHOLD = 0.75
qa_embeddings = None

def save_embeddings(embeddings: Dict, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(embeddings, file)

def load_embeddings(filename: str) -> Dict:
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
async def precompute_embeddings():
    print("Loading embeddings from file and precomputing tensors...")
    with open('kb_simple.json', 'r') as file:
        qa_data = json.load(file)

    qa_embeddings = {}
    for question, answer in qa_data.items():
        question_embedding = model.encode(question, convert_to_tensor=True)
        qa_embeddings[question] = {
            'answer': answer,
            'embedding': question_embedding
        }
    return qa_embeddings

def initialize_app():
    global qa_embeddings

    embeddings_filename = 'qa_embeddings.pkl'

    if os.path.exists(embeddings_filename):
        print(f'Loading embeddings from {embeddings_filename}')
        qa_embeddings = load_embeddings(embeddings_filename)

    else:
        print('No saved embeddings found, precomputing...')
        qa_embeddings = asyncio.run(precompute_embeddings())
        save_embeddings(qa_embeddings, embeddings_filename)
        print(f'Embeddings saved to {embeddings_filename}')

def main():
    parser = argparse.ArgumentParser(description='Backend application for embedding-based search.')
    parser.add_argument('--precompute', action='store_true', help='Precompute embeddings and save them to disk')
    args = parser.parse_args()

    if args.precompute:
        print('Precomputing embeddings...')
        qa_embeddings = asyncio.run(precompute_embeddings())
        save_embeddings(qa_embeddings, 'qa_embeddings.pkl')
        print(f'Embeddings saved to qa_embeddings.pkl')

initialize_app()

app = Flask(__name__)

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
   
    chat_messages = [
        {"role": "system", "content": "You are an support assistant for company Centio #CYBERSECURITY. You use a tone that is technical and scientific. You answer only in Bulgarian language. You answer only questions related to products of ESET and Sophos."},
        {"role": "user", "content": f"{user_question}"},
        {"role": "assistant", "content": f"{answer}"}
    ]


    response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0,
    )

    print("Въпрос от клиент:",user_question)
    print("Отговор от базата:",answer)
    print("Сходство:",similarity)
    print("Prompt:",chat_messages)
    print("Отговор от OpenAI:",response.choices[0].message['content'])
    
    return response.choices[0].message['content'].strip()

async def process_question(user_question):
    best_question, similarity = await find_best_question_match(user_question, qa_embeddings)
    best_answer = qa_embeddings.get(best_question, {}).get('answer', "")
    context_answer = await get_more_context(best_question, best_answer, user_question, similarity)
    return context_answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    context_answer = asyncio.run(process_question(user_question))
    return jsonify({'answer': context_answer})

if __name__ == '__main__':
    main()
    app.run()