from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import logging
from typing import Dict
import os, glob
from modules.DocLoader import DocumentLoader
from modules.TaskHandler import TaskHandler
from modules.ContextQA import ContextQA
from modules.Summarize import DocumentSummarizer
from modules.WebSearch import WebSearchAgent
import asyncio
from modules.WebSearch import WebSearchAgent


# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# メモリーをセッションごとに管理
sessions: Dict[str, ConversationBufferMemory] = {}

def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return sessions[session_id]

def check_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        if files:  
            return True
    return False

def delete_files_in_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        try:
            os.remove(file)
            logger.info("Files cleaned up")
        except Exception as e:
            print(f'Error deleting files: {e}')

async def generate_response(query: str, session_id: str) -> str:
    temp_file_path = None
    response = None
    try:
        memory = get_or_create_memory(session_id)

        dir_files = check_files_in_directory("uploads")
        if dir_files == True:
            doc_loader = DocumentLoader(directory_path="uploads")
            temp_file_path = doc_loader.create_temp_file()
            if temp_file_path is not None:
              temp_file_path = "uploads/temp_combined.txt"

        handler = TaskHandler()
        task = handler.process_query(query)
        logger.info(f"Task type determined: {task}")

        if task == "task1":
            llm = ChatOllama(
                model="elyza:jp8b",
                temperature=0,
                timeout=30
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "あなたは誠実で優秀なAIアシスタントです。ユーザーとの会話履歴を考慮しながら、丁寧に日本語で回答してください。"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            chain = prompt | llm
            
            ai_messages = await chain.ainvoke({
                "input": query,
                "chat_history": memory.chat_memory.messages
            })
            response = ai_messages.content

        elif task == "task2":
            context_qa = ContextQA(temp_file_path)
            response = context_qa.get_answer(query) 

        elif task == "task3":
            summarizer = DocumentSummarizer(temp_file_path)
            response = summarizer.summarize(query) 
        
        elif task == "task4":
            web_search_agent = WebSearchAgent()
            response = web_search_agent.answer_query(query)

        # responseをメモリに保存
        if response is not None:
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response)
            
            # responseのコピーを作成
            final_response = response

            # ファイル削除を実行
            if dir_files:
                try:
                    folder_path = 'uploads'
                    delete_files_in_folder(folder_path)
                    logger.info(f"Cleaned up files for session {session_id}")
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {cleanup_error}")

            return final_response
        else:
            raise ValueError("No response generated")

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="メッセージが空です")
            
        response = await generate_response(request.message, request.session_id)
        return ChatResponse(reply=response)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="サーバーエラーが発生しました")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501)
