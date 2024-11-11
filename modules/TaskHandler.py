from openai import OpenAI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
import instructor
from typing import Literal
import os

class TaskDetail(BaseModel):
    Task: Literal["task1", "task2", "task3", "task4"]

class TaskHandler:
    def __init__(self, base_url="http://localhost:11434/v1", api_key="ollama", directory="uploads"):
        self.prompt_template = ChatPromptTemplate.from_template("""
            あなたは高性能な言語モデルです。タスクは、4種類です:
            1. 通常の会話（参照テキストはありません）
            2. 参照テキストに基づくQ&A (参照テキストがあります)
            3. 文書の要約（参照テキストがあります）
            4. あなたが学習していない最新情報や専門性の高い情報を取得するためのweb検索（参照テキストはありません）

            ユーザーのクエリを分析し、クエリに対応したタスクを適切に判断してください:
            参照テキストの有無もタスク判断の参考になります。

            回答は、通常の会話の場合"task1"、参照テキストに基づくQ&Aの場合"task2"、文書の要約の場合"task3"、最新情報や高度な専門情報を取得するためのweb検索の場合"task4"とだけ答えてください。

            参照テキスト:
            {reference}

            ユーザーのクエリ:
            {input}

            回答:
        """)
        
        self.client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key=api_key,
            ),
            mode=instructor.Mode.JSON,
        )
        self.directory = directory

    def search_file(self, filename):
        """
        Searches for a specified file in the directory and its subdirectories.
        """
        for root, dirs, files in os.walk(self.directory):
            if filename in files:
                with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
                    return f.read()
        return None

    def process_query(self, query, filename="temp_combined.txt"):
        """
        Processes the user's query and determines the task based on the presence of a specified file.
        """
        reference_text = self.search_file(filename)
        reference_status = "参照テキストあり" if reference_text else "参照テキストなし"

        # Prepare the input for the prompt
        prompt_content = self.prompt_template.format(reference=reference_status, input=query)

        # Get the task decision from the LLM
        response = self.client.chat.completions.create(
            model="elyza:jp8b",
            messages=[{"role": "user", "content": prompt_content}],
            response_model=TaskDetail,
        )

        # Validate and return the task, with a default in case of unexpected output
        task = response.Task if response.Task in ["task1", "task2", "task3", "task4"] else "task1"  # デフォルト: task1
        return task
    
















# from langchain_ollama import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# import os

# class TaskHandler:
#     def __init__(self):
#         self.directory = "uploads"
#         self.filename = "temp_combined.txt"
        
#         # LLMの初期化
#         self.llm = ChatOllama(
#             model="elyza:jp8b",
#             temperature=0,
#         )
        
#         # プロンプトテンプレートの設定
#         self.template = """
#         あなたは高性能な言語モデルです。タスクは、4種類です:
#         1. 通常の会話（参照テキストはありません）
#         2. 参照テキストに基づくQ&A (参照テキストがあります)
#         3. 文書の要約（参照テキストがあります）
#         4. 最新情報を取得するためのweb検索（参照テキストはありません）

#         ユーザーのクエリを分析し、クエリに対応したタスクを適切に判断してください:
#         参照テキストの有無もタスク判断の参考になります。

#         回答は、通常の会話の場合"task1"、参照テキストに基づくQ&Aの場合"task2"、文書の要約の場合"task3、最新情報を取得するためのweb検索の場合"task4とだけ答えてください。

#         参照テキスト:
#         {reference}

#         ユーザーのクエリ:
#         {input}

#         回答:
#         """
        
#         self.prompt = ChatPromptTemplate.from_template(self.template)
#         self.chain = self.prompt | self.llm | StrOutputParser()

#     def search_file(self):
#         for root, dirs, files in os.walk(self.directory):
#             if self.filename in files:
#                 return os.path.join(root, self.filename)
#         return None

#     def process_query(self, query):
#         result = self.search_file()
        
#         if result is None:
#             reference_text = "参照テキストなし"
#         else:
#             reference_text = "参照テキストあり"
            
#         response = self.chain.invoke({
#             "input": query,
#             "reference": reference_text
#         })
        
#         return response

