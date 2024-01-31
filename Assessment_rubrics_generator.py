

import os
import time
import torch
import pandas
import huggingface_hub
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    pipeline,
)

import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent

from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents import AgentExecutor
import os
load_dotenv()

llm = ChatOpenAI(openai_api_key= os.getenv("OPENAI_API_KEY"), model_name="gpt-4")

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

DATA_PATH = r"E:\Pycharm Projects\EPTech\UnconstrainED\rubric-generator\resources"

loader = DirectoryLoader(DATA_PATH, glob='*.pdf',loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=130)
texts = text_splitter.split_documents(documents)

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain.vectorstores import Chroma
# load it into Chroma
db = Chroma.from_documents(texts, embeddings)

from langchain.prompts import PromptTemplate


sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.
If a question does not make any sense, or is not factually coherent, explain why, instead of answering something incorrect. If you don't know the answer to a question, say that you don't know"""

instruction = """CONTEXT:/n/n {context}/n
You are Matty, an educator's assistant dedicated to creating Assessment plans.

every question will start from a new line in a new paragraph. 

Using the final assessment questions, create a comprehensive rubric to evaluate this assessment.
Use the given rubric-template.pdf  and rubrics.pdf to create the rubric. It has grading scales from 1 to 4, where 1 is lowest and 4 is highest. Make sure to edit the template and mention terms and keywords from the assessment.

Once the rubric is created, ask teacher for feedback and edit the rubric accordingly, do not move till the teacher is satisfied with the rubric.
When the teacher is satisfied with the rubric, end the task!
Question: {question}

"""


# prompt_template = get_prompt(instruction, sys_prompt)
prompt_template = instruction + sys_prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

from langchain.chains.question_answering import load_qa_chain
llm_chain = load_qa_chain(
    llm,
    prompt=PROMPT,
    )

query= "Make a lesson plan for the 4th grade Earth Science"
# query= input()
search_results = db.similarity_search(query, k=4)

timeStart = time.time()

result = llm_chain.run({"question": query, "input_documents": search_results})

# print(result)
print("Time taken: ", -timeStart + time.time())
