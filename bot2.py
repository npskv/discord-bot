import discord
import openai
import json
from sentence_transformers import SentenceTransformer, util
from typing import Dict
from openai import ChatCompletion
import torch
import logging

logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# Set up intents
intents = discord.Intents.default()
intents.typing = False
intents.presences = True
intents.members = True
intents.messages = True
intents.guild_messages = True
intents.message_content = True 
client = discord.Client(intents=intents)


# Load embeddings and transformers
with open('questions_answers.json', 'r') as file:
    qa_embeddings = json.load(file)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
openai.api_key = os.environ.get("OPENAI_API_KEY")

SIMILARITY_THRESHOLD = 0.91

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
    print(f"Answer: {answer}")
    print(f"Similarity: {similarity}")
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

@client.event
async def on_message_edit(before, after):
    print(f"Before edit: {before.content}")
    print(f"After edit: {after.content}")

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    if message.author.bot:
        return

    print(f"Message content: {message.content}")
    print(f"Message author: {message.author}")
    print(f"Message channel: {message.channel}")
    print(f"Message guild: {message.guild}")


    bot_member = message.guild.get_member(client.user.id) if message.guild else None
    bot_nickname = bot_member.nick if bot_member else None

    if client.user.mention in message.content or client.user.mention.replace('!', '') in message.content or client.user.name in message.content or (bot_nickname and bot_nickname in message.content):
        user_question = message.clean_content.replace(f"@{client.user.name}", "").strip()
        best_question, similarity = await find_best_question_match(user_question, list(qa_embeddings.keys()))
        best_answer = qa_embeddings.get(best_question, "")
        context_answer = await get_more_context(best_question, best_answer, user_question, similarity)
        await message.channel.send(context_answer)

    if isinstance(message.channel, discord.channel.DMChannel):
        hint = "Моля, задайте въпросите си в канал, като ме споменете с моето име на Discord."
        await message.channel.send(hint)

client.run(os.environ.get("DISCORD_BOT_TOKEN"))
