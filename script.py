# Install the packages that needed
!pip install --upgrade langchain openai -q
!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6
!apt-get install poppler-utils
!pip install tiktoken -q
!pip install pinecone-client -q
!pip install --upgrade pillow==6.2.2

import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"]="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" # provide openai api key

# generate the pinecone index 
pinecone.init(
    api_key="0b8494ce-c994-4389-9b59-7cd4d94f0f3d",
    environment="us-west4-gcp-free"
)


class Answering_agent:

    def __init__(self,model_name):
        self.documents=None
        self.model_name = model_name
        self.docs=None

    def load_docs(self,directory):
        loader = DirectoryLoader(directory)
        self.documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        self.docs = text_splitter.split_documents(documents)
        

    def processing(self,documents):
        embeddings = OpenAIEmbeddings(model_name="text-davinci-002")
        index = Pinecone.from_documents(self.docs, embeddings, index_name="myindex")
        return index 
    
    def get_similiar_docs(self,query, k=2, score=False):
        index= processing(self.documents)
        if score:
            similar_docs = index.similarity_search_with_score(query, k=k)
        else:
            similar_docs = index.similarity_search(query, k=k)
        return similar_docs


    def get_answer(self,query):
      llm = OpenAI(model_name=self.model_name)
      chain = load_qa_chain(llm, chain_type="stuff")
      similar_docs = get_similiar_docs(query)
      answer = chain.run(input_documents=similar_docs, question=query)
      if len(answer)!=0:
          return answer
      else:
          return "I don't have answers for your question"


agent = Answering_agent("model_name") #give the model name whichever the user wants

# For upload the file
uploaded_files = "directory path"
agent.load_docs(uploaded_files)

while True:
    question = input("Ask a question: ")

    if question.lower() == 'end it':
        break

    answer = agent.get_answer(question)
    print(answer)







    

