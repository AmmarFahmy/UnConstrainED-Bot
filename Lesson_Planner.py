
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
from langchain.chains import LLMChain
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

import os
load_dotenv()

llm = ChatOpenAI(openai_api_key= os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

DATA_PATH = r"E:\Pycharm Projects\EPTech\UnconstrainED\lesson-planner\resources"

loader = DirectoryLoader(DATA_PATH, glob='*.pdf',loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=130)
texts = text_splitter.split_documents(documents)

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain.vectorstores import Chroma
# load it into Chroma
db = Chroma.from_documents(texts, embeddings)

from langchain.prompts import PromptTemplate



prompt_template = """
Create a detailed lesson plan for the specified subject and topic for the specified grade level, that runs for the given duration. Consider the additional requests by the teacher when creating the lesson plan.
Adhere to the following guidelines when creating the lesson plan:  
1. This lesson plan will be the teacher's roadmap of what students need to learn and how it will be done effectively during class time. A successful lesson plan addresses and integrates these three key components: Objectives for student learning, teaching/learning activities, and strategies to check student understanding.
2. The lesson plan should begin with a review of prerequisite knowledge. Alert students of the lesson goals and then present new information a little at a time.
3. Model procedures, give clear samples, and check often to make sure students understand. Allow substantial practice with the new information. Ask lots of questions to allow students to correctly repeat or explain a procedure or concept. 
4. It's important to represent the learning topic through a variety of examples, analogies, and connections to other knowledge. The lesson plan should anticipate and interpret student errors, represent ideas in multiple forms and develop alternative explanations.
5. The lesson plan should specify concrete objectives for student learning and outline teaching and learning activities that will be used in class. 
6. It will also define how the teacher will check whether the learning objectives have been accomplished. The lesson plan should include ways for teachers to check student mastery and understanding by posing questions, providing examples, and correcting misconceptions.
7. The lesson plan should be clear about what students will learn and how they should show or demonstrate understanding of the material.
8. You will be provided with files that offer a set of concrete suggestions that can be applied to any discipline or domain to ensure that all learners can access and participate in meaningful, challenging learning opportunities. Please use this as guidelines in creating your lesson plan.
9. Use the Madelyn Hunter Lesson Cycle to structure your lesson plan.

User Query:- {user_query}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["user_query"])

from langchain.chains.question_answering import load_qa_chain
llm_chain = LLMChain(
    llm=llm,
    prompt=PROMPT,
    )

query= "Make a lesson plan for the 4th grade Earth Science"


timeStart = time.time()

result = llm_chain.run(user_query= query)

# print(result)
print("Time taken: ", -timeStart + time.time())
