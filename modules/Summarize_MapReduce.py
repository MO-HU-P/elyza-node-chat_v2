from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from typing import Optional
from pathlib import Path
import logging

class DocumentSummarizer:
    def __init__(self, temp_file_path: Optional[str] = None):
        # loggingの設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.temp_file_path = temp_file_path
        self.llm = ChatOllama(model="elyza:jp8b", temperature=0)

        self.output_parser = StrOutputParser()
        
        # テキスト分割の設定
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
        )
        
        # プロンプトテンプレートの設定
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはプロの編集者です。
            以下の文書をユーザーの意図を反映させて要約してください。

            以下の点に注意してください:
            - 重要なキーワードを漏らさない
            - 文書の本質的な意味を保持する
            - 架空の表現を使用しない
            - 数値は変更しない

            ユーザーのメッセージ: 
            {query}

            要約する文書:
            {text}""")
        ])
        
        self.reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", """以下は部分的な要約です。
            ユーザーの意図を反映させながら、これらを統合してください。

            ユーザーのメッセージ:
            {query}

            部分要約:
            {text}

            以下の形式で出力してください：

            【要約】
            （ここに統合された要約を記載）

            【要約の観点】
            - 重視した点
            - 抽出したキーワード
            - 要約方針の説明""")
        ])

    def _load_document(self) -> Optional[str]:
        if not self.temp_file_path:
            return None
            
        file_path = Path(self.temp_file_path)
        if not file_path.exists():
            self.logger.error(f"ファイルが存在しません: {self.temp_file_path}")
            return None
            
        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()
            return content if content.strip() else None
        except Exception as e:
            self.logger.error(f"ファイルの読み込みに失敗しました: {e}")
            return None

    def prepare_documents(self, text: str) -> list:
        try:
            texts = self.text_splitter.split_text(text)
            return [Document(page_content=t) for t in texts]
        except Exception as e:
            self.logger.error(f"テキスト分割中にエラーが発生しました: {e}")
            raise

    def map_step(self):
        return self.summary_prompt | self.llm | self.output_parser

    def reduce_step(self):
        return self.reduce_prompt | self.llm | self.output_parser

    def summarize(self, query: str = "この文書を要約してください") -> str:
        try:
            # ファイルから文書を読み込む
            content = self._load_document()
            if content is None:
                return "文書の読み込みに失敗しました。"
            
            if not content.strip():
                return "文書が空です。"
                
            # ドキュメントの準備
            docs = self.prepare_documents(content)
            
            # Mapステップ: 各チャンクを要約
            map_chain = self.map_step()
            individual_summaries = []
            
            for doc in docs:
                summary = map_chain.invoke({
                    "text": doc.page_content,
                    "query": query
                })
                individual_summaries.append(summary)
            
            # Reduceステップ: 要約を統合
            if len(individual_summaries) == 1:
                return individual_summaries[0]
                
            reduce_chain = self.reduce_step()
            final_summary = reduce_chain.invoke({
                "text": "\n\n".join(individual_summaries),
                "query": query
            })
            
            return final_summary
            
        except Exception as e:
            self.logger.error(f"要約処理中にエラーが発生しました: {e}")
            return f"要約処理中にエラーが発生しました: {e}"
