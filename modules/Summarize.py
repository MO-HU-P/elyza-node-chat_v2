from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
from pathlib import Path
import logging

class DocumentSummarizer:   
    def __init__(self, temp_file_path: Optional[str] = None):
        self.temp_file_path = temp_file_path
        # loggingの設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.llm = ChatOllama(model="elyza:jp8b", temperature=0)
        except Exception as e:
            self.logger.error(f"言語モデルの初期化に失敗しました: {e}")
            raise

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

    def summarize(self, query: str) -> str:
        if not query.strip():
            return "入力が提供されていません。"

        try:
            document = self._load_document() if self.temp_file_path else \
                "要約対象となる文書を検出できませんでした。文書を添付し、メッセージでご指示ください。"

            template = """あなたはプロの編集者です。以下の文書をユーザーの意図を反映させて要約してください。

            以下の点に注意してください:
            - 重要なキーワードを漏らさない
            - 文書の本質的な意味を保持する
            - 架空の表現を使用しない
            - 数値は変更しない

            ユーザーのメッセージ: 
            {user_query}

            要約する文書:
            {document}

            要約結果は以下の形式で出力してください：

            【要約】
            （ここに要約を記載）

            【要約の観点】
            - 重視した点
            - 抽出したキーワード
            - 要約方針の説明
            """

            prompt = PromptTemplate(
                input_variables=["user_query", "document"],
                template=template
            )

            chain = prompt | self.llm | StrOutputParser()

            return chain.invoke({
                "user_query": query,
                "document": document
            })
            
        except Exception as e:
            self.logger.error(f"要約処理中にエラーが発生しました: {e}")
            return f"要約処理中にエラーが発生しました: {e}"