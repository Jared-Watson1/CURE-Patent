from embedding import pineconeFilter
from metapub import PubMedFetcher
import pandas as pd
import openai
import json
import os
import time
from dotenv import load_dotenv
load_dotenv()
NCBI_API_KEY = os.getenv("NCBI_API_KEY")


openai.api_key = os.getenv("OPENAI_API_KEY")
fineTunedModel = "davinci:ft-personal-2022-12-30-20-49-43"

fetch = PubMedFetcher()
gpt4 = "gpt-4"
gptTurbo = "gpt-3.5-turbo"


def getArticlesInfo(articles):
    titles = []
    for article in articles:
        article = fetch.article_by_pmid(article)
        titles.append(article.title)
    return titles


def addToFineTune(prompt, completion, file="data/questionsCreator.jsonl"):
    with open(file, "a") as f:
        data = {"prompt": prompt, "completion": completion}
        f.write(json.dumps(data) + "\n")


def promptRefiner(question):
    question = f"Thoroughly examine the provided input prompt, ascertain its medical category, pinpoint the essential keywords, and construct a refined, detailed prompt that incorporates supplementary information. Always structure the prompt in the form of a question. This enhanced prompt should facilitate a more precise and in-depth response when utilized by the medical assistant chat-GPT AI, thereby improving its understanding of the user's inquiry and yielding a more satisfactory answer. If the question is conversational in manner or is referring to another message in the thread just return the category without any further prompt refinement. \nprompt: {question}"
    response = openai.ChatCompletion.create(
        model=gptTurbo,
        messages=[
            {"role": "system", "content": "You are an AI-driven Prompt Refinement Assistant trained to analyze user inputs, categorize questions, identify essential keywords, and generate improved, detailed prompts that lead to more accurate and comprehensive responses from the medical assistant chat-GPT AI."},
            {"role": "user", "content": question},
        ]
    )
    refinedPrompt = response['choices'][0]['message']['content']
    return refinedPrompt


def jared_bot(prompt):
    startTime = time.time()
    prompt = promptRefiner(prompt)
    filter = pineconeFilter(prompt, topK=3)
    context = filter[0]
    p = f"""
    Context: {context}

    Question: ### {prompt} ###
    """
    print(p)

    response = openai.ChatCompletion.create(
        model=gptTurbo,
        messages=[
            {"role": "system", "content": "As a medical assistant chatGPT AI named CURE (Comprehensive Understanding and Research Engine AI), please provide a comprehensive answer to the given medical inquiry based on the relevant context supplied. Your response should address relevant aspects mentioned in the user's question, use appropriate medical terminology, and convey the information in a clear and concise manner. Remember to consider any recommendations or limitations stated in the provided medical literature, and ensure your answer is descriptive, thorough, and accurate. Do not make up any information. When referncing a specific study in your response use intext citations at the end of the sentence with the specific PM ID like this (PM ID: 1234567)"},
            {"role": "user", "content": p},
        ]
    )
    answer = response['choices'][0]['message']['content']
    endTime = time.time()
    print(f"Execution time: {endTime - startTime}")
    return answer, filter[1]
