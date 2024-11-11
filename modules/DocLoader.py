from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import os
import logging
from typing import List, Dict, Optional

class DocumentLoader:
    def __init__(self, directory_path: str = "uploads", debug: bool = False):
        self.directory_path = directory_path
        self.temp_file_path = os.path.join(directory_path, "temp_combined.txt")
        self.debug = debug
        
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
        }

    def get_files_by_type(self) -> Dict[str, List[str]]:
        files_by_type = {ext: [] for ext in self.loaders.keys()}
        
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in self.loaders:
                    full_path = os.path.join(root, file)
                    files_by_type[ext.lower()].append(full_path)
                    self.logger.debug(f"Found {ext} file: {full_path}")
        
        return files_by_type

    def create_directory_loader(self, file_type: str) -> DirectoryLoader:
        return DirectoryLoader(
            path=self.directory_path,
            glob=f"**/*{file_type}",
            loader_cls=self.loaders[file_type],
            show_progress=self.debug,
        )

    def load_documents(self) -> List:
        all_documents = []
        files_by_type = self.get_files_by_type()
        
        for file_type, files in files_by_type.items():
            if not files:
                self.logger.info(f"No {file_type} files found")
                continue
                
            try:
                self.logger.info(f"Loading {len(files)} {file_type} files...")
                loader = self.create_directory_loader(file_type)
                documents = loader.load()
                all_documents.extend(documents)
                self.logger.info(f"Successfully loaded {len(documents)} {file_type} documents")
            except Exception as e:
                self.logger.error(f"Error loading {file_type} files: {str(e)}")
        
        return all_documents

    def create_temp_file(self) -> Optional[str]:
        try:
            documents = self.load_documents()
            if not documents:
                self.logger.warning("No documents were loaded")
                return None
            
            # 各ドキュメントのテキストを処理して余計な改行を削除
            processed_texts = []
            for doc in documents:
                # 複数の改行を単一の改行に置換
                text = ' '.join(doc.page_content.split())
                processed_texts.append(text)
            
            # 処理したテキストを単一の文字列に結合
            combined_text = ' '.join(processed_texts)
            
            with open(self.temp_file_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            self.logger.info(f"Combined text saved to: {self.temp_file_path}")
            return self.temp_file_path
            
        except Exception as e:
            self.logger.error(f"Error creating temp file: {str(e)}")
            return None

    def __str__(self) -> str:
        return f"DocumentLoader(directory_path='{self.directory_path}', supported_formats={list(self.loaders.keys())})"