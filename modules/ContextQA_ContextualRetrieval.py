from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document

class ContextQA:
    def __init__(self, temp_file_path):
        self.temp_file_path = temp_file_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="elyza:jp8b", temperature=0)
        self.chain = None

    def load_context(self):
        with open(self.temp_file_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
        documents = [Document(page_content=document_content)]
        return documents

    def setup_qa_chain(self):
        if self.chain is None:
            documents = self.load_context()
            
            # チャンクに分割
            chunks = self.text_splitter.split_text(documents[0].page_content)
            
            # 各チャンクに対して文脈情報を生成
            contextualized_chunks = []
            for chunk in chunks:
                context = self.llm.invoke(f"""
                <document>
                ドキュメント全体の内容
                </document>
                ドキュメント全体の中に配置したいチャンクは次のとおりです。
                <chunk>
                {chunk}
                </chunk>
                このチャンクを文書全体の中に位置づけるための簡潔なコンテキストを記述してください。
                """)
                contextualized_chunks.append(Document(page_content=f"{context} {chunk}"))

            db = FAISS.from_documents(contextualized_chunks, self.embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 3})

            prompt = PromptTemplate.from_template(
                "質問: {user_query}\n\n背景情報:\n{context}\n\n回答:"
            )

            self.chain = (
                {"context": retriever, "user_query": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
        return self.chain

    def get_answer(self, user_query):
        chain = self.setup_qa_chain()
        answer = chain.invoke(input=user_query)  
        return answer

