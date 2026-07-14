from langchain_community.document_loaders import DirectoryLoader, TextLoader,PyPDFLoader , Docx2txtLoader 
from pathlib import Path 
documents =[]
DATA = "data"
if Path(DATA).exists():
    documents += DirectoryLoader(DATA,glob="**/*.pdf",loader_cls = PyPDFLoader).load()
    documents += DirectoryLoader(DATA,glob="**/*.docx",loader_cls = Docx2txtLoader).load()
    documents += DirectoryLoader(DATA,glob="**/*.txt",loader_cls = TextLoader).load()
import os 
from google import genai 

from dotenv import load_dotenv 
load_dotenv() 
key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key= key)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunk = text.split_documents(documents)
print("chunk created...")

from langchain_huggingface import HuggingFaceEmbeddings
ch = "chroma_db"
embedding= "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name = embedding)

from langchain_chroma import Chroma 

if Path(ch).exists():
    vector_db = Chroma(persist_directory=ch , embedding_function=embedding_model)
else: 
    vector_db= Chroma.from_documents(documents=chunk,embedding=embedding_model,persist_directory=ch)
    print("new chroma created")

def topk_retrival(vector_db,question,K):
    retriever = vector_db.as_retriever(search_kwargs = {"k":K})
    retrived_doc = retriever.invoke(question) 
    context = "\n\n".join(doc.page_content for doc in retrived_doc)
    return context 

while True:
    question= input("enter question :")
    if question in ["quit","stop"]:

        break 
    context = topk_retrival(vector_db, question, 4)
    sys_prompt = f'hey give ans with reference to {context} for the  {question}'
    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=sys_prompt)
    print(response.text)  

