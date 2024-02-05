

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


DATA_PATH = r"D:\01.VeracityGP-VeracityAI\Aistra_works\UnConstrainED-Bot\rubric-generator\resources"

loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1300, chunk_overlap=130)
texts = text_splitter.split_documents(documents)

# load it into Chroma
db = Chroma.from_documents(texts, embeddings)


prompt_template = """
You are an educator's AI assistant, you should help teachers to create a detailed and comprehensive rubric for assessment evaluation.
Assessments can be one of the following,
1. Multiple Choice
2. Matching
3. True/False
4. Short Answer
5. Essay/Long Answer
6. Group Activity
7. Project Based
8. Problem Based
9. Oral/Speaking
10. Presentation
11. Discussion
Starting off:
Teacher will give you the grade of the students they are teaching, the subject, the topic/topics of the subject and the type of the assessment. 
Make sure that they have proviced all these information before moving the the next step.
Finally, you should create a detailed and comprehensive rubric to evaluate that assessment type given.
Use a given rubric template to create the rubric. It has grading scales from 1 to 4, where 1 is the lowest and 4 is the highest. The rubric must be in a table format.
Make sure to edit the template and mention terms and keywords related to the assessment type.

Once this is done, ask the teacher feedback for the assessment created. If any feedback is given, edit your assessment according to the feedback, show it to the teacher, and ask for feedback again. Do not move till the teacher is satisfied with the assessment.

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
