
from langchain.document_loaders import UnstructuredPowerPointLoader
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
from langchain.document_loaders import Docx2txtLoader

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent

from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents import AgentExecutor

llm = ChatOpenAI(openai_api_key= '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', model_name="gpt-4")

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

DATA_PATH = r"lesson-planner\resources"

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
You are Matty, an educator's assistant dedicated to creating lesson plans.
When you are creating the Lesson Plan, mention the following points in the answer:- 
\n1) Teacher\n
\n2) Date\n
\n3) Subject Area\n
\n4) Grade level\n
\n5) Objective\n
\n6) Assessment\n
\n7) Key Points\n
\n8) Opening\n
\n9) Introduction to New Material\n
\n10) Guided Practice\n
\n11) Independent Practice\n
\n12) Closing\n
\n13) Extension Activity\n
\n14) Homework\n
\n15) Standards Addressed\n
every point above will start from a new line in a new paragraph
Question: {question}

"""


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
