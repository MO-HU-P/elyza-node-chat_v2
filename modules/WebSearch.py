from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.tools import BaseTool
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import os
from fake_useragent import UserAgent

os.environ['USER_AGENT'] = UserAgent().chrome

from langchain_community.document_loaders import WebBaseLoader


class SearchResult(BaseModel):
    title: str = Field(description="ページのタイトル")
    url: str = Field(description="ページのURL")
    content: str = Field(description="ページの内容")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Searches the web for relevant information using DuckDuckGo"
    
    def __init__(self):
        super().__init__()
        self._wrapper = DuckDuckGoSearchAPIWrapper(
            backend='api',
            max_results=5,
            region='jp-jp',
            safesearch='moderate',
            source='text',
            time='y'
        )
        self._embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len)
    
    def _get_search_results(self, query: str) -> List[SearchResult]:
        results = []
        try:
            search_results = self._wrapper.results(query=query, max_results=3)
            for r in search_results:
                results.append(SearchResult(
                    title=r.get('title', ''),
                    url=r.get('link', ''),
                    content=r.get('snippet', '')
                ))
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _process_web_content(self, urls: List[str]) -> List[Document]:
        processed_documents = []
        for url in urls:
            try:
                loader = WebBaseLoader([url])
                documents = loader.load()
                processed_documents.extend(self._text_splitter.split_documents(documents))
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                continue
        return processed_documents

    def _create_vector_store(self, texts: List[Document], query: str) -> List[str]:
        if not texts:
            return []
        try:
            db = FAISS.from_documents(texts, self._embeddings)
            similar_docs = db.similarity_search(query, k=3)
            return [doc.page_content for doc in similar_docs]
        except Exception as e:
            print(f"Vector store error: {str(e)}")
            return []

    def _run(self, query: str) -> Dict[str, Any]:
        search_results = self._get_search_results(query)
        if not search_results:
            return {"error": "検索結果が見つかりませんでした"}

        urls = [result.url for result in search_results]
        documents = self._process_web_content(urls)
        if not documents:
            return {"error": "ウェブコンテンツの処理に失敗しました"}

        similar_contents = self._create_vector_store(documents, query)
        
        return {
            "search_results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "summary": r.content[:200] + "..." if len(r.content) > 200 else r.content
                } for r in search_results[:3]
            ],
            "relevant_contents": similar_contents
        }


class WebSearchAgent:
    def __init__(self, model_name: str = "elyza:jp8b"):
        self.search_tool = WebSearchTool()
        self.chat_ollama = ChatOllama(model=model_name)
        self.prompt_template = ChatPromptTemplate.from_template(
            """以下の情報に基づいて、ユーザーの質問に答えてください。
情報:
{context}

質問: {query}

できるだけ具体的に、わかりやすく説明してください。"""
        )

    def answer_query(self, query: str) -> str:
        try:
            search_results = self.search_tool.run(query)
            
            if "error" in search_results:
                return f"【回答】\n{search_results['error']}\n"

            relevant_contents = search_results.get("relevant_contents", [])
            if not relevant_contents:
                return "【回答】\n申し訳ありませんが、関連情報が見つかりませんでした。\n"

            messages = self.prompt_template.format_messages(
                context="\n".join(relevant_contents),
                query=query
            )
            
            response = self.chat_ollama.generate([messages])
            llm_response = response.generations[0][0].text

            sources = search_results["search_results"]
            formatted_sources = "\n【情報源】\n" + "\n".join(
                f"タイトル: {source['title']}\nURL: {source['url']}\nサマリー: {source['summary']}\n"
                for source in sources
            )

            return f"【回答】\n{llm_response}\n{formatted_sources}"

        except Exception as e:
            return f"【回答】\nエラーが発生しました: {str(e)}\n"

