from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, #split 된 chunk의 최대 크기
    chunk_overlap=200, #앞 뒤로 나뉘어진 chunk들이 얼마나 겹쳐도 되는지 지정
)

loader = Docx2txtLoader('./tax_with_table.docx')
# document = loader.load()
document_list = loader.load_and_split(text_splitter=text_splitter)

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 환경변수를 불러옴
load_dotenv()

# OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
embedding = OpenAIEmbeddings(model='text-embedding-3-large') #text-embedding-3-small 은 성능이 좋지 않음

from langchain_chroma import Chroma

# 데이터를 처음 저장할 때 
#database = Chroma.from_documents(documents=document_list, embedding=embedding, collection_name='chroma-tax', persist_directory="./chroma")

# 이미 저장된 데이터를 사용할 때 
database = Chroma(collection_name='chroma-tax', persist_directory="./chroma", embedding_function=embedding)

query = '국군포로가 받는 퇴직일시금은 얼마인가요?'  

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini')

prompt = f"""[Identity]
- 아래는 세금과 관련된 정보가 담긴 문서입니다.
- [Context]를 참고해서 질문에 답해주세요.

[Context]
{retrieved_docs}

Question: {query}
"""

from langsmith import Client

client = Client()

prompt = client.pull_prompt("rlm/rag-prompt")
print(prompt)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 문서 리스트 → 문자열 변환 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer in Korean."
)

# Retriever
retriever = database.as_retriever(search_kwargs={"k": 3})

# 체인
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

response = qa_chain.invoke(query)