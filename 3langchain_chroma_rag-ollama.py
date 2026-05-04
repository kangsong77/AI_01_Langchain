from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. 문서 로드 및 분할
loader = Docx2txtLoader('./tax_with_table.docx')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150
)
document_list = text_splitter.split_documents(docs)
print(f"총 {len(document_list)}개의 문서 조각이 생성되었습니다.")

# 2. 로컬 임베딩 모델 설정 (Ollama 사용)
# 'nomic-embed-text'는 Ollama에서 가장 많이 쓰이는 임베딩 모델입니다.
# (미리 터미널에서 'ollama pull nomic-embed-text' 명령어로 받아두어야 합니다.)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. 벡터 데이터베이스 생성/로드
# persist_directory를 지정하여 데이터를 로컬에 저장합니다.
database = Chroma.from_documents(
    documents=document_list, 
    embedding=embeddings, 
    collection_name='ollama-tax',
    persist_directory="./chroma_ollama"
)

# 4. Ollama EXAONE 3.5 모델 설정
llm = ChatOllama(
    model="exaone3.5:2.4b",
    temperature=0  # 답변의 일관성을 위해 0으로 설정
)

# 5. RAG 시스템 구성 (Chain)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    "당신은 친절한 세무 전문가입니다. 아래 제공된 [맥락]만을 사용하여 질문에 답하세요.\n"
    "정답을 모를 경우 억지로 지어내지 말고 '문서에서 관련 내용을 찾을 수 없습니다'라고 답하세요.\n\n"
    "[맥락]\n{context}\n\n"
    "질문: {question}\n\n"
    "답변:"
)

retriever = database.as_retriever(search_kwargs={"k": 3})

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# 6. 실행
query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
response = qa_chain.invoke(query)

print("-" * 30)
print(f"질문: {query}")
print(f"답변: {response.content}")
print("-" * 30)

# 답변: 제공된 맥락에서 연봉 5천만원 직장인의 구체적인 소득세 금액을 계산하거나 특정 세율 정보를 도출할 수 없습니다. 주어진 정보는 세금 공제 및 특례 적용 시기와 관련된 규정들에 초점을 맞추고 있으며, 개인의 소득 수준에 따른 정확한 세금 계산에는 필요한 세부적인 세율 정보가 부족합니다. 따라서, 정확한 소득세 금액을 알기 위해서는 추가적인 세부 세율 정보나 현재 적용 중인 세법 조항이 필요합니다. 문서에서 관련 내용을 찾을 수 없습니다.