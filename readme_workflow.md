# 薬剤併用禁忌チェックワークフロー

## システム概要

このシステムは、複数の医薬品添付文書PDFを分析し、薬剤間の併用禁忌や注意事項を自動的に抽出・分析するツールです。Anthropic Claude LLMを活用して、PDFから関連情報を抽出し、薬剤間の相互作用を評価します。

![システムアーキテクチャ](https://i.imgur.com/example.png)

## 特徴

- 複数の添付文書PDFのURLを入力して一括分析
- PDFからのマークダウン変換と構造化
- 「禁忌」「相互作用」「併用禁忌」「併用注意」に関する情報の自動抽出
- 薬剤間の相互作用の詳細分析
- チャット形式での質問応答機能

## セットアップ手順

### 前提条件

- Python 3.8以上
- Anthropic Claude API キー
- インターネット接続（PDFダウンロード用）

### インストール

1. リポジトリをクローン

```bash
git clone https://github.com/username/drug-contraindication-checker.git
cd drug-contraindication-checker
```

2. 仮想環境を作成して有効化

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. 必要なパッケージをインストール

```bash
pip install -r requirements.txt
```

4. 環境変数を設定

```bash
# Windows
set CLAUDE_API_KEY=your_api_key_here

# macOS/Linux
export CLAUDE_API_KEY=your_api_key_here
```

## 実行方法

1. バックエンドサーバーを起動

```bash
python implementation.py
```

2. Webブラウザで以下のURLにアクセス

```
file:///パス/to/frontend.html
```

または、任意のWebサーバーを使用してfrontend.htmlを配信することもできます。

## 使い方

1. Web画面で添付文書PDFのURLを入力します（複数可）
   - 例: `https://www.pmda.go.jp/PmdaSearch/iyakuDetail/ResultDataSetPDF/xxxxx_xxxxxxxx_x_xx`

2. 必要に応じて患者情報を入力します（任意）
   - 例: `50歳男性、うつ病、前立腺肥大症、エスシタロプラム内服中`

3. 「併用禁忌をチェック」ボタンをクリックします

4. システムが添付文書を解析し、結果を表示します
   - 併用禁忌の有無
   - 併用注意の有無
   - 詳細な分析結果

5. 分析結果に対して質問ができます
   - 例: `この併用禁忌の理由は何ですか？`
   - 例: `代替薬として何が考えられますか？`

## 技術的詳細

### システムの処理フロー

1. PDFのURL入力
2. PDFダウンロード
3. PDFからテキスト抽出
4. テキストからマークダウン変換
5. Claude LLMによる関連情報抽出
6. Claude LLMによる薬剤相互作用分析
7. 結果の表示
8. チャット形式でのフォローアップ質問対応

### 使用ライブラリ

- **FastAPI**: RESTful APIフレームワーク
- **Anthropic Claude**: LLMによる情報抽出と分析
- **PyPDF2**: PDFテキスト抽出
- **Requests**: HTTP通信
- **Bootstrap**: フロントエンドUI
- **Marked.js**: マークダウンレンダリング

## 注意事項・免責事項

- このシステムは医療専門家の判断を支援するためのツールであり、医療判断の代替となるものではありません。
- 分析結果は必ず医療専門家（医師・薬剤師）が確認してください。
- 添付文書の内容や形式によっては、正確に情報を抽出できない場合があります。
- 患者の状態や具体的な用法用量によって、禁忌・注意事項は変わる可能性があります。
- 常に最新の添付文書情報を参照してください。

## 将来の開発計画

- 医薬品DBとの連携
- 電子カルテシステムとの統合
- オフライン処理のサポート
- モバイルアプリの開発
- 時系列での薬剤相互作用トラッキング 