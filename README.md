# v.2 Japanese LLM Chatbot - Enhanced Multi-functional System

本リポジトリは、[v.1](https://github.com/MO-HU-P/elyza-node-chat.git)の基本機能をベースに、ファイルアップロードや各種タスクの追加により、ユーザーの多様なニーズに対応する強化版チャットシステムを実装したものです。Ollama LLMやLangChainなどのLLMライブラリを駆使し、高度な情報処理と応答生成を実現しています。本ReadMeでは、v.1との差分を中心に、新機能やシステムの技術詳細を紹介します。

---

## ⚠️ 重要な注意事項

- このアプリケーションは教育・研究目的で作成されています
- このプロジェクトはMITライセンスの下で公開されています。使用するモデル（ELYZA Llama-3-JP-8B）は[Meta Llama 3 Community License](https://www.llama.com/llama3/license/)の下で提供されています。モデルの使用に関しては、このライセンスに従ってください。


---

## リポジトリのファイル構成

### バックエンドコア
- `server.js` - Node.jsサーバー（フロントエンド API）
- `app.py` - メインアプリケーション（ルーティングと各モジュールの統合管理）

### 機能モジュール（modulesディレクトリ）
- `DocLoader.py` - 文書管理モジュール
- `Summarize.py` - 文書要約モジュール
- `ContextQA.py` - 文書Q&A処理モジュール
- `WebSearch.py` - Web検索モジュール
- `TaskHandler.py` - タスク管理モジュール
- `ContextQA_ContextualRetrieval.py` - 拡張Q&A機能モジュール（オプション）
- `Summarize_MapReduce.py` - 拡張文書要約モジュール（オプション）

### フロントエンド（publicディレクトリ）
- `index.html` - チャットUIのメインページ
- `style.css` - UIスタイリング

### 設定ファイル
- `requirements.txt` - Pythonパッケージの依存関係
- `package.json` - Node.jsパッケージの依存関係
- `package-lock.json` - 依存関係の正確なバージョンを記録し、一貫した環境を保証するファイル
- `Llama-3-ELYZA-JP-8B-q4_k_m.gguf` - ELYZAの言語モデルファイル（要ダウンロード）
- `Modelfile` - 言語モデルの設定ファイル

---

## ディレクトリ構造
プロジェクトのファイルは以下のような構造で配置してください：
```
project_root/
│
├── modules/
│   ├── DocLoader.py
│   ├── Summarize.py
│   ├── ContextQA.py
│   ├── WebSearch.py
│   └── TaskHandler.py
│
├── public/
│   ├── index.html
│   └── style.css
│
├── uploads/
│
├── server.js
├── app.py
│
├── requirements.txt
├── package.json
├── package-lock.json
├── Llama-3-ELYZA-JP-8B-q4_k_m.gguf
└── Modelfile
```

---

## システム要件および環境構築手順（参考資料）
システム要件や環境構築手順の詳細については、[v.1リポジトリ](https://github.com/MO-HU-P/elyza-node-chat.git)を参照してください。

### Ollama Embeddingモデルの追加

v.1プロジェクトのセットアップに加えて、nomic-embed-textのセットアップを実行してください。

```bash
ollama pull nomic-embed-text
```

---

## 主な機能更新とUIの改善

### 1. ファイルアップロード機能
UIには、メッセージ入力ボックスの横にファイルアップロードボタンが追加され、PDF、CSV、TXT、DOCX形式のファイルをアップロード可能です。
- **ファイルの表示と削除**: アップロード後、ファイル名がメッセージ入力ボックス下に表示され、「×」ボタンで削除が可能です。
- **複数ファイルの同時アップロード**: 各ファイルが個別に添付され、UIに即座に反映されます。
- **自動テキスト抽出**: PDF、CSV、TXT、DOCXファイルについては、Pythonスクリプトがテキストを抽出し、後続のQ&Aや要約タスクに利用します。

### 2. 拡張されたタスクエージェント
このシステムは、以下の4つのエージェントタスクを統合し、クエリに応じた多角的な応答を提供します。
1. **通常会話**: 基本的なチャット応答
2. **参照文書を基にしたQ&A**: アップロード文書から適切な回答を検索
3. **文書の要約**: 長文の要約と要点抽出
4. **ウェブ検索**: DuckDuckGoを使用して最新情報を提供

---

## アーキテクチャ概要

### DocumentLoader クラス
ユーザーがアップロードした文書（PDF、CSV、TXT、DOCX）を効率的に管理し、後続タスクで利用するために統合テキストを生成します。
- **ファイルの自動検出**: `get_files_by_type` メソッドでファイル形式別に分類し、各形式に最適なローダー（例: `PyPDFLoader`、`CSVLoader`など）で読み込みます。
- **一時ファイルの生成**: `create_temp_file` メソッドで抽出テキストを保存し、各プロセスがアクセスできるようにします。
- **デバッグ**: `debug`フラグで詳細なログを記録し、エラーハンドリングも強化されています。

### DocumentSummarizer クラス
アップロードされた文書を要約し、わかりやすい情報を生成します。
- **要約ポリシー**: 重要キーワードを網羅し、正確な要約を提供します。
- **プロンプトテンプレートと例外処理**: エラーハンドリングやログ機能を備え、LLMの安定稼働をサポートします。

### ContextQA クラス
文書ベースのQ&Aを行い、ベクトル化による検索と高精度な回答を実現します。
- **ベクトル検索**: `nomic-embed-text`でベクトル化し、FAISSデータベースを利用して関連する文脈を検索し、LLMへの精度の高い回答を提供します。
- **チャンク化**: 初期設定では、チャンクサイズを200、重複サイズを20に設定しています。参照する文書量に応じて調整してください。

### WebSearchTool クラス
最新情報が必要なクエリに対し、リアルタイムでウェブ検索を行い、関連情報を収集します。
- **検索プロセス**: DuckDuckGo APIで情報を取得し、FAISSを使用して関連する検索結果をベクトル化。Q&Aに最適化した情報提供が可能です。

### TaskHandler クラス
ユーザーからのクエリ内容に応じて適切なエージェントを選択します。
- **タスク選定**: `Ollama`や`OpenAI`のプロンプトテンプレートを使用し、タスク（通常会話、参照文書Q&A、文書要約、ウェブ検索）を判定します。
- **JSONレスポンス**: 結果はJSON形式で返され、エージェント実行が効率化されます。

---

## システム全体のフロー

1. **クエリの解析**: ユーザーのクエリはTaskHandlerにより解析され、参照文書の有無とともに最適なタスクが選定されます。
2. **エージェント実行**:
   - **通常会話**: 簡易応答
   - **参照文書Q&A**: 文書からの回答
   - **文書要約**: DocumentSummarizerで要約
   - **ウェブ検索**: 最新情報や高度な専門情報の収集
3. **応答の生成と提供**: 結果を要約や関連リンクとともに返し、ユーザーに正確で信頼性の高い情報を提供します。

---

# オプション

## 1. Context QA 実装
### ファイル構成と切り替え方法
リポジトリには`ContextQA.py`と`ContextQA_ContextualRetrieval.py`の2つの実装が存在します。デフォルトでは前者が使用されています。

**切り替え手順**：
1. ルートディレクトリ内の`ContextQA.py`を削除
2. `ContextQA_ContextualRetrieval.py`のファイル名を`ContextQA.py`に変更
3. `app.py`を実行することで、`__pycache__`ディレクトリが更新され新しい実装が有効になります

### 実装の特徴と比較

#### 現在の実装（ContextQA.py）
- `split_documents`メソッドによる直接的なドキュメント分割
- シンプルで軽量な処理
- メタデータや文脈の追加処理なし

**適用場面**：
- 高速な処理が必要な場合
- リソースが制限された環境
- 基本的な質問応答タスク

#### オプションの実装（ContextQA_ContextualRetrieval.py）
本実装は、Anthropic社が公開した[Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)の手法を採用しています。

**主な機能**：
- 文脈を考慮したチャンク処理
- `split_text`メソッドによるテキストの分割
- LLMを使用した各チャンクへの文脈付与
- 文脈情報の統合による文書理解の強化

```python
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
```

**適用場面**：
- 高精度な回答が必要な場合
- 文書の文脈理解が重要な場合
- リソース制約の少ない環境
- 複雑な質問応答タスク

### 使用時の注意点
オプションの実装では、各チャンクに対してLLMを呼び出すため、計算リソースとメモリ使用量が増加します。一方で、豊富な文脈情報により精度の向上が期待できます。

---

## 2. Summarize 実装
### ファイル構成と切り替え方法
リポジトリには`Summarize.py`と`Summarize_MapReduce.py`の2つの実装が存在します。デフォルトでは前者が使用されています。

**切り替え手順**：
1. ルートディレクトリ内の`Summarize.py`を削除
2. `Summarize_MapReduce.py`のファイル名を`Summarize.py`に変更
3. `app.py`を実行することで、`__pycache__`ディレクトリが更新され新しい実装が有効になります

### 実装の特徴と比較

#### 現在の実装（Summarize.py）
- 文書全体を一度に処理する単純な実装(Stuff方式)
- 一つのプロンプトテンプレートで要約を生成
- メモリ使用量が文書サイズに比例して増加

**適用場面**：
- 比較的短い文書（数千文字程度まで）の要約
- 文書の全体的な文脈を維持する必要がある場合
- 迅速な処理が必要な場合

#### オプションの実装（Summarize_MapReduce.py）
本実装は、Map-Reduce手法を用いて長文を効率的に要約する実装です。

**主な機能**：
- Map-Reduce方式による段階的な要約処理
- 文書を200文字単位のチャンクに分割して処理
- 20文字のオーバーラップでコンテキストの連続性を確保
- 二段階のプロンプトテンプレート（要約用と統合用）を使用

**適用場面**：
- 長文書の要約
- メモリ効率を重視する場合
- より詳細な情報の保持が必要な場合

### 使用時の注意点
1. 文書の長さや性質に応じて適切な実装を選択してください。
2. `Summarize_MapReduce.py`を使用する場合、チャンクサイズとオーバーラップの値は必要に応じて調整可能です。
3. 処理時間は文書の長さと分割数に比例して増加する可能性があります。
