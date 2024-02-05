

from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
import time
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(openai_api_key=os.getenv(
    "OPENAI_API_KEY"), model_name="gpt-4")


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs)


DATA_PATH = r"D:\01.VeracityGP-VeracityAI\Aistra_works\UnConstrainED-Bot\assessment-generator\resources"

loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1300, chunk_overlap=130)
texts = text_splitter.split_documents(documents)

# load it into Chroma
db = Chroma.from_documents(texts, embeddings)


prompt_template = """
You are an educator's AI assistant, you should help teachers to create assessments for a given topic.
Assessments can be,
1. Multiple Choice - you should provide 20 questions with answers
2. Matching - you should provide two set of Matching questions with 8 matching pairs
3. True/False - you should provide 20 questions with answers
4. Short Answer - you should provide 20 questions with answers
5. Essay/Long Answer - you should provide 20 questions with answers
6. Group Activity - you should provide 3 real world problems fro group activity with the expected solutions
7. Project Based - you should provide 3 real world project statements with the expected solutions in point format
8. Problem Based - you should provide 3 real world problem statements with the expected solutions in point format
9. Oral/Speaking
10. Presentation
11. Discussion - you should provide 3 real world problem statements with the expected discusstion points
Starting off:
Teacher will give you the grade of the students they are teaching, the subject, and the topic/topics of the subject that need to be assessed. 
Make sure that they have proviced all these information before creating the assessment.
Finally, you should create a detailed assessment, using The Cognitive Process Dimension.

User Query:- {user_query}
"""
PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["user_query"])

llm_chain = LLMChain(
    llm=llm,
    prompt=PROMPT,
)

# query = "Make a Rubric for the 4th grade Earth Science"


# timeStart = time.time()

# result = llm_chain.run(user_query=query)

# # print(result)
# print("Time taken: ", -timeStart + time.time())
