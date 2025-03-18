"""
薬剤併用禁忌チェックワークフロー
Anthropic Claude APIを利用して医薬品添付文書PDFの分析を行うシステム
"""

import os
import sys
import traceback
import requests
from typing import List, Dict, Any
import tempfile
import anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import pypdf
import markdown
import uvicorn
import logging
from dotenv import load_dotenv
from markitdown import MarkItDown  # 追加: markitdownパッケージをインポート

# .env ファイルから環境変数を読み込む
load_dotenv()

# ロギング設定
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

# 環境変数からAPIキーを取得
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    logger.warning("CLAUDE_API_KEY 環境変数が設定されていません")

# Document Intelligence エンドポイント設定 (なければNone)
DOCINTEL_ENDPOINT = os.getenv("DOCINTEL_ENDPOINT", None)
logger.info(f"Document Intelligence エンドポイント: {'設定あり' if DOCINTEL_ENDPOINT else 'なし'}")

# MarkItDownインスタンスの初期化
md_converter = MarkItDown(enable_plugins=False)

# APIキーの確認（セキュリティのため最初の数文字のみログ出力）
if CLAUDE_API_KEY:
    masked_key = CLAUDE_API_KEY[:4] + "..." if len(CLAUDE_API_KEY) > 4 else "設定済み"
    logger.info(f"Claude APIキー: {masked_key}")
else:
    logger.warning("Claude APIキーが設定されていません")

# FastAPI アプリケーションの初期化
app = FastAPI(title="薬剤併用禁忌チェックシステム")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:10000",
        "http://localhost:5000",
        "https://your-render-app-name.onrender.com",
        "https://your-frontend-domain.com"  # フロントエンドをホストするドメイン
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# データモデル
class PDFUrlInput(BaseModel):
    urls: List[HttpUrl]
    patient_info: str = ""

class ChatRequest(BaseModel):
    conversation_id: str
    question: str

class AnalysisResult(BaseModel):
    conclusion: str
    details: Dict[str, Any]
    conversation_id: str

# Anthropic Claudeクライアントの初期化
try:
    logger.info("Anthropicクライアントの初期化を開始")
    
    # 最新のAnthropic API (0.8.x)に合わせた初期化
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # モデル名は新しいものに更新
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
    logger.info(f"Anthropicクライアント初期化成功。モデル: {CLAUDE_MODEL}")
    
except Exception as e:
    logger.error(f"Anthropicクライアント初期化エラー: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# 会話履歴を保存する簡易データストア（本番環境ではDBを使用）
conversation_store = {}

# PDFダウンロード関数
async def download_pdf(url: str) -> str:
    """URLからPDFをダウンロードし、一時ファイルに保存"""
    try:
        logger.info(f"PDFのダウンロードを開始: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 一時ファイルを作成
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        
        # PDFコンテンツを書き込み
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        logger.info(f"PDFのダウンロードに成功: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"PDFのダウンロードに失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"PDFのダウンロードに失敗しました: {str(e)}")

# PDF→テキスト変換関数
async def pdf_to_text(pdf_path: str) -> str:
    """PDFファイルからテキストを抽出"""
    try:
        logger.info(f"PDFからテキスト抽出を開始: {pdf_path}")
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            logger.info(f"PDFページ数: {len(reader.pages)}")
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n\n"
                if i < 2 or i > len(reader.pages) - 3:  # 最初と最後のページのみログ出力
                    logger.info(f"ページ {i+1} からのテキスト抽出 (抜粋): {page_text[:100]}...")
        
        logger.info(f"PDFからのテキスト抽出完了: 文字数 {len(text)}")
        return text
    except Exception as e:
        logger.error(f"PDFの解析に失敗しました: {str(e)}")
        logger.error(traceback.format_exc())
        # デモ用のダミーテキストを返す
        logger.warning("PDFの解析に失敗したため、デモ用のダミーテキストを返します")
        return """
医薬品添付文書（デモ用テキスト）

【禁忌】
特定の条件がある患者には使用禁止

【相互作用】
他の薬剤との併用に注意

【併用禁忌】
一部の薬剤との併用は禁止されています

【副作用】
一般的な副作用と対処法

【用法・用量】
適切な服用方法
        """

# テキスト→マークダウン変換（MarkItDown使用版）
async def text_to_markdown(text: str, pdf_path: str = None) -> str:
    """抽出したテキストを簡易的なマークダウンに変換、またはPDFを直接マークダウンに変換"""
    try:
        logger.info("テキストからマークダウンへの変換を開始")
        
        # PDFパスが提供されている場合は、直接PDFからマークダウンに変換
        if pdf_path and os.path.exists(pdf_path):
            try:
                logger.info(f"MarkItDownを使用してPDF直接変換を試みます: {pdf_path}")
                result = md_converter.convert(pdf_path)
                if result and hasattr(result, 'text_content') and result.text_content:
                    markdown_text = result.text_content
                    logger.info(f"MarkItDownによる変換成功: 文字数 {len(markdown_text)}")
                    return markdown_text
                else:
                    logger.warning("MarkItDownによる変換結果が空でした。従来の方法にフォールバックします")
            except Exception as e:
                logger.error(f"MarkItDownによる変換に失敗しました: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("従来の変換方法にフォールバックします")
        
        # MarkItDownが使えない場合や失敗した場合は、従来の方法でテキスト処理
        lines = text.split('\n')
        markdown_text = ""
        
        for line in lines:
            # 見出しと思われる行の処理
            if line.strip() and len(line.strip()) < 50 and line.strip().isupper():
                markdown_text += f"## {line.strip()}\n\n"
            # 箇条書きと思われる行の処理
            elif line.strip().startswith('•') or line.strip().startswith('・'):
                markdown_text += f"- {line.strip()[1:].strip()}\n"
            # 通常のテキスト
            elif line.strip():
                markdown_text += f"{line.strip()}\n\n"
        
        logger.info(f"マークダウン変換完了: 文字数 {len(markdown_text)}")
        return markdown_text
    except Exception as e:
        logger.error(f"マークダウン変換エラー: {str(e)}")
        logger.error(traceback.format_exc())
        # シンプルなフォールバック処理
        return f"## 変換エラー\n\n{text}"

# LLMによる情報抽出
async def extract_relevant_info(markdown_content: str, document_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Claude LLMを使用して関連情報を抽出
    
    Args:
        markdown_content: マークダウン形式の添付文書内容
        document_metadata: 文書のメタデータ（URL、ファイル名など）
        
    Returns:
        Dict: 抽出した情報と文書メタデータを含む辞書
    """
    prompt = f"""あなたは医薬品情報の専門家です。以下の添付文書から詳細な情報を抽出し、構造化してください。
特に薬剤の相互作用と禁忌に関する情報を詳細に抽出することが重要です。

添付文書:
{markdown_content}

抽出すべき情報:
1. 薬剤の基本情報（薬剤名、一般名、剤形、製造販売元など）
2. 効能・効果
3. 用法・用量
4. 禁忌となる患者の状態や疾患（すべての禁忌条件を詳細に列挙）
5. 併用禁忌となる薬剤とその理由（詳細な作用機序や危険性も含む）
6. 併用注意が必要な薬剤とその理由（詳細な作用機序や危険性も含む）
7. 特定の患者（高齢者、腎機能障害患者、肝機能障害患者、妊婦など）への注意事項
8. 重大な副作用
9. その他の重要な警告や注意事項

抽出した情報は、項目ごとにマークダウン形式で整理してください。各セクションはレベル2の見出し（##）を使用してください。
情報が見つからない項目については「情報なし」と明記してください。

可能な限り添付文書の原文の表現を保持し、情報を省略せずに抽出してください。特に禁忌事項や相互作用については完全に抽出することが重要です。
"""

    try:
        logger.info("Claude APIを使用して情報抽出を開始")
        
        # 最新の標準的なAPI呼び出し
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,  # トークン数を増やして情報量を確保
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("Claude API呼び出しに成功しました")
        
        # メタデータがない場合は空の辞書を使用
        if document_metadata is None:
            document_metadata = {}
            
        # 抽出結果を辞書形式で返す
        return {
            "content": response.content[0].text,
            "metadata": document_metadata
        }
            
    except Exception as e:
        logger.error(f"Claude APIの呼び出しに失敗しました: {str(e)}")
        logger.error(traceback.format_exc())
        # エラー時のモックレスポンス
        error_response = f"""
## 薬剤の基本情報
- 情報なし（APIエラーのためモックレスポンス）

## 効能・効果
- 情報なし（APIエラーのためモックレスポンス）

## 用法・用量
- 情報なし（APIエラーのためモックレスポンス）

## 禁忌となる患者の状態や疾患
- 情報なし（APIエラーのためモックレスポンス）

## 併用禁忌となる薬剤とその理由
- 情報なし（APIエラーのためモックレスポンス）
- エラー詳細: {str(e)}

## 併用注意が必要な薬剤とその理由
- 情報なし（APIエラーのためモックレスポンス）

## 特定の患者への注意事項
- 情報なし（APIエラーのためモックレスポンス）

## 重大な副作用
- 情報なし（APIエラーのためモックレスポンス）

## その他の重要な警告や注意事項
- 情報なし（APIエラーのためモックレスポンス）
"""
        return {
            "content": error_response,
            "metadata": document_metadata or {}
        }

# LLMによる薬剤併用分析
async def analyze_drug_interactions(extracted_infos: List[Dict[str, Any]], patient_info: str) -> Dict[str, Any]:
    """複数の薬剤情報を分析し、相互作用と患者情報を考慮した総合評価を行う"""
    # 抽出情報を薬剤ごとに整理
    drug_info_text = ""
    drug_details = []  # 薬剤の詳細情報を保持
    
    for i, info in enumerate(extracted_infos):
        if isinstance(info, dict) and "content" in info:
            # 新しい形式の場合
            content = info["content"]
            metadata = info.get("metadata", {})
            drug_name = get_drug_name_from_content(content, metadata.get("title", "不明な薬剤"))  # コンテンツから薬剤名を抽出する関数
            url = metadata.get("url", f"文書{i+1}")
            
            drug_details.append({
                "index": i + 1,
                "name": drug_name,
                "url": url,
                "content": content
            })
            
            drug_info_text += f"薬剤{i+1}情報（{drug_name}）:\n{content}\n\n"
        else:
            # 古い形式または文字列の場合（互換性のため）
            drug_info_text += f"薬剤{i+1}情報:\n{info}\n\n"
            drug_details.append({
                "index": i + 1,
                "name": f"文書{i+1}",
                "url": "",
                "content": info if isinstance(info, str) else str(info)
            })
    
    # 患者情報の解析と構造化
    patient_analysis_prompt = ""
    if patient_info and patient_info.strip():
        patient_analysis_prompt = f"""
まず、以下の患者情報を解析し、薬剤適正使用の観点から重要な以下の点を特定してください:
- 年齢・性別
- 診断されている疾患
- 現在の健康状態
- 腎機能/肝機能の状態（言及されていれば）
- 妊娠/授乳の状態（該当する場合）
- アレルギー（言及されていれば）
- その他、薬剤選択に影響する可能性のある特記事項

患者情報:
{patient_info}

上記の分析結果を踏まえて、以下の薬剤情報と併せて総合的な評価を行ってください。
各薬剤について、この患者への投与の適切性も判断してください。
"""
    
    prompt = f"""{patient_analysis_prompt}あなたは薬剤師として、複数の薬剤の併用に関する分析と患者適合性の評価を行います。
以下の薬剤情報に基づいて、薬剤の適切性と併用の安全性を詳細に評価してください。

{drug_info_text}

分析すべき内容:
1. 個別の薬剤評価:
   a. 各薬剤は対象患者に投与可能か？禁忌事項に該当しないか？
   b. 患者の状態（年齢、性別、疾患、臓器機能など）から投与に注意が必要な薬剤はあるか？

2. 併用評価:
   a. これらの薬剤間に併用禁忌はあるか？あれば詳細と理由を説明してください。
   b. 併用注意はあるか？あれば詳細と対処法を説明してください。

3. 用量適切性:
   a. 患者の状態から、用量調整が必要な薬剤はあるか？
   b. 現在の用量は適切か？

4. 総合評価:
   a. 総合的な判断として、これらの薬剤の使用は安全か？
   b. 代替案や調整が必要な場合は具体的に提案してください。
   c. モニタリングすべき副作用や相互作用の症状は何か？

回答はマークダウン形式で、明確な構造と見出しを用いて提供してください。
特に重要な警告や禁忌事項は強調して表示してください。
各薬剤について「投与可」「投与注意」「投与禁忌」の評価を明示してください。
"""

    try:
        logger.info("薬剤相互作用と患者適合性の分析を開始")
        
        # 最新の標準的なAPI呼び出し
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("Claude API呼び出しに成功しました")
        analysis_text = response.content[0].text
        
        # 患者固有の注意事項を抽出
        patient_specific_warnings = "患者固有の注意事項はありません"
        if patient_info and patient_info.strip():
            if "患者" in analysis_text and ("注意" in analysis_text or "警告" in analysis_text or "禁忌" in analysis_text):
                patient_lines = [line for line in analysis_text.split('\n') if "患者" in line 
                                and ("注意" in line or "警告" in line or "禁忌" in line)]
                if patient_lines:
                    patient_specific_warnings = "\n".join(patient_lines)
        
        # 投与可否の評価を抽出
        drug_suitability = []
        for drug in drug_details:
            drug_name = drug["name"]
            
            # デフォルト値
            suitability = {
                "name": drug_name,
                "verdict": "評価なし",
                "verdict_class": "neutral",
                "reasons": []
            }
            
            # 内容からこの薬剤の投与可否を検出
            if f"{drug_name}" in analysis_text or f"薬剤{drug['index']}" in analysis_text:
                # 投与禁忌の検出
                if (f"{drug_name}.*投与禁忌" in analysis_text or 
                    f"薬剤{drug['index']}.*投与禁忌" in analysis_text or
                    f"{drug_name}.*禁忌" in analysis_text):
                    suitability["verdict"] = "投与禁忌"
                    suitability["verdict_class"] = "danger"
                    
                    # 理由の抽出（簡易的）
                    pattern = f"({drug_name}|薬剤{drug['index']}).*?(禁忌|投与すべきでない).*?([^\n]+)"
                    import re
                    matches = re.findall(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        for match in matches:
                            reason = match[2].strip()
                            if reason and len(reason) < 200:  # 長すぎる理由は除外
                                suitability["reasons"].append(reason)
                
                # 投与注意の検出
                elif (f"{drug_name}.*投与注意" in analysis_text or 
                      f"薬剤{drug['index']}.*投与注意" in analysis_text or
                      f"{drug_name}.*注意" in analysis_text):
                    suitability["verdict"] = "投与注意"
                    suitability["verdict_class"] = "warning"
                    
                    # 理由の抽出（簡易的）
                    pattern = f"({drug_name}|薬剤{drug['index']}).*?(注意|慎重投与).*?([^\n]+)"
                    import re
                    matches = re.findall(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        for match in matches:
                            reason = match[2].strip()
                            if reason and len(reason) < 200:
                                suitability["reasons"].append(reason)
                
                # 投与可の検出
                elif (f"{drug_name}.*投与可" in analysis_text or 
                      f"薬剤{drug['index']}.*投与可" in analysis_text or
                      f"{drug_name}.*適切" in analysis_text):
                    suitability["verdict"] = "投与可"
                    suitability["verdict_class"] = "success"
            
            # 理由が見つからない場合、デフォルトの理由を追加
            if not suitability["reasons"]:
                if suitability["verdict"] == "投与禁忌":
                    suitability["reasons"].append("患者の状態により禁忌")
                elif suitability["verdict"] == "投与注意":
                    suitability["reasons"].append("患者の状態により注意が必要")
            
            drug_suitability.append(suitability)
        
        # 危険度レベルの評価
        risk_level = "低"
        if "投与禁忌" in analysis_text or "併用禁忌" in analysis_text or "絶対禁忌" in analysis_text:
            risk_level = "高"
        elif "投与注意" in analysis_text or "併用注意" in analysis_text or "慎重投与" in analysis_text:
            risk_level = "中"
        
        # 簡易的な構造化（実際には解析ライブラリを使用）
        # 結果構造
        result = {
            "full_analysis": analysis_text,
            "summary": analysis_text.split("\n\n")[0] if "\n\n" in analysis_text else analysis_text[:200],
            "has_contraindications": "併用禁忌" in analysis_text or "禁忌" in analysis_text,
            "has_precautions": "併用注意" in analysis_text or "注意" in analysis_text,
            "risk_level": risk_level,
            "patient_specific_warnings": patient_specific_warnings,
            "drug_details": drug_details,  # 薬剤詳細情報
            "drug_suitability": drug_suitability  # 薬剤の患者適合性評価
        }
        
        return result
    except Exception as e:
        logger.error(f"薬剤相互作用分析エラー: {str(e)}")
        logger.error(traceback.format_exc())
        # エラー時のフォールバックレスポンス
        mock_analysis = f"""
# 薬剤併用分析結果（システムエラー）

## エラー情報

システムエラーが発生したため、分析結果を提供できません。
エラー: {str(e)}

## 推奨事項

医療専門家に直接ご相談ください。
"""
        return {
            "full_analysis": mock_analysis,
            "summary": "システムエラーが発生しました。医療専門家に直接ご相談ください。",
            "has_contraindications": False,
            "has_precautions": True,
            "risk_level": "不明",
            "patient_specific_warnings": "システムエラーのため評価できません",
            "drug_details": drug_details,
            "drug_suitability": []  # 空の適合性評価
        }

# 薬剤名を抽出する補助関数
def get_drug_name_from_content(content: str, pdf_title: str = "不明な薬剤") -> str:
    """マークダウンコンテンツから薬剤名を抽出する
    
    基本情報セクションから薬剤名を見つけようとします。
    見つからない場合はデフォルト値を返します。
    """
    try:
        lines = content.split('\n')
        # 基本情報セクションを探す
        for i, line in enumerate(lines):
            if "## 薬剤の基本情報" in line and i < len(lines) - 1:
                # 次の5行を検索
                for j in range(1, min(10, len(lines) - i)):
                    if "薬剤名" in lines[i + j] and ":" in lines[i + j]:
                        return lines[i + j].split(":", 1)[1].strip()
                    elif "商品名" in lines[i + j] and ":" in lines[i + j]:
                        return lines[i + j].split(":", 1)[1].strip()
                    elif "一般名" in lines[i + j] and ":" in lines[i + j]:
                        return lines[i + j].split(":", 1)[1].strip()
                    # リスト形式の場合
                    elif "- " in lines[i + j]:
                        item = lines[i + j].strip("- ").strip()
                        if "薬剤名" in item or "商品名" in item:
                            parts = item.split(":", 1)
                            if len(parts) > 1:
                                return parts[1].strip()
                            parts = item.split("：", 1)
                            if len(parts) > 1:
                                return parts[1].strip()
        
        # 見つからなければタイトルを探す
        for line in lines[:10]:
            if line.startswith("# "):
                return line.strip("# ").strip()
    except Exception as e:
        logger.warning(f"薬剤名抽出中にエラー: {str(e)}")
    
    return pdf_title

# フォローアップ質問応答
async def answer_followup_question(conversation_id: str, question: str) -> str:
    """会話履歴に基づいてフォローアップ質問に回答"""
    try:
        logger.info(f"フォローアップ質問処理: {question[:30]}...")
        
        if conversation_id not in conversation_store:
            logger.warning(f"指定された会話ID {conversation_id} が見つかりません")
            raise HTTPException(status_code=404, detail="指定された会話IDが見つかりません")
        
        # 会話履歴を取得
        conversation = conversation_store[conversation_id]
        logger.info("会話履歴を取得しました")
        
        # 質問を追加
        prompt = f"""以下は薬剤併用に関する分析結果です：

{conversation['analysis_result']}

ユーザーからの質問:
{question}

医薬品情報の専門家として、上記の分析結果に基づいて質問に詳細かつ正確に回答してください。
"""

        # 最新の標準的なAPI呼び出し
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("Claude API呼び出しに成功しました")
        answer_text = response.content[0].text
        
        # 回答を会話履歴に追加
        conversation['messages'].append({
            "question": question,
            "answer": answer_text
        })
        logger.info("回答を会話履歴に追加しました")
        
        return answer_text
    except HTTPException:
        # 既にHTTPExceptionがraiseされている場合はそのまま再raise
        raise
    except Exception as e:
        logger.error(f"フォローアップ質問処理エラー: {str(e)}")
        logger.error(traceback.format_exc())
        return f"""
システムエラーが発生しました。

ご質問: "{question}"

エラー詳細: {str(e)}

医療に関する質問については、医師または薬剤師に直接ご相談ください。
"""

# API エンドポイント
@app.post("/analyze-pdfs", response_model=AnalysisResult)
async def analyze_pdfs(input_data: PDFUrlInput):
    """複数のPDF URLを受け取り、薬剤情報を分析"""
    try:
        logger.info(f"PDF分析リクエスト受信: URL数 {len(input_data.urls)}")
        extracted_infos = []
        
        # 各PDFを処理
        for i, url in enumerate(input_data.urls):
            try:
                logger.info(f"PDF {i+1}/{len(input_data.urls)} の処理を開始: {url}")
                # PDFをダウンロード
                pdf_path = await download_pdf(str(url))
                try:
                    # テキスト抽出
                    text = await pdf_to_text(pdf_path)
                    # マークダウン変換
                    markdown_text = await text_to_markdown(text, pdf_path)
                    # 関連情報抽出
                    extracted_info = await extract_relevant_info(markdown_text, {"url": url})
                    extracted_infos.append(extracted_info)
                    logger.info(f"PDF {i+1} の処理が完了しました")
                finally:
                    # 一時ファイルの削除
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                        logger.info(f"一時ファイル {pdf_path} を削除しました")
            except Exception as e:
                logger.error(f"PDF {i+1} の処理中にエラー: {str(e)}")
                logger.error(traceback.format_exc())
                # エラーが発生しても処理を続行し、エラーメッセージを追加
                error_info = f"""
## PDF処理エラー

このPDFの処理中にエラーが発生しました: {str(e)}
URL: {url}

以下の理由が考えられます:
- PDFのフォーマットが不適切
- URLが間違っている
- サーバーアクセスの問題

エラーの詳細は管理者にご連絡ください。
"""
                extracted_infos.append({
                    "content": error_info,
                    "metadata": {"url": str(url), "error": str(e)}
                })
        
        # 薬剤相互作用の分析
        logger.info("薬剤相互作用の分析を開始します")
        analysis_result = await analyze_drug_interactions(extracted_infos, input_data.patient_info)
        
        # 会話IDを生成（実際の実装ではもっと堅牢な方法を使用）
        import uuid
        conversation_id = str(uuid.uuid4())
        logger.info(f"会話ID生成: {conversation_id}")
        
        # 会話履歴を保存
        conversation_store[conversation_id] = {
            "analysis_result": analysis_result["full_analysis"],
            "messages": []
        }
        logger.info("会話履歴を保存しました")
        
        # 結果を返す
        return AnalysisResult(
            conclusion=analysis_result["summary"],
            details=analysis_result,
            conversation_id=conversation_id
        )
    except Exception as e:
        logger.error(f"PDF分析処理全体でエラー: {str(e)}")
        logger.error(traceback.format_exc())
        # デモ用のフォールバック結果
        import uuid
        conversation_id = str(uuid.uuid4())
        fallback_analysis = f"""
# システムエラー

PDF分析処理中にエラーが発生しました: {str(e)}

医療情報については、医師または薬剤師に直接ご相談ください。
"""
        conversation_store[conversation_id] = {
            "analysis_result": fallback_analysis,
            "messages": []
        }
        
        return AnalysisResult(
            conclusion="システムエラーが発生しました。医療専門家に直接ご相談ください。",
            details={
                "full_analysis": fallback_analysis,
                "summary": "システムエラーが発生しました",
                "has_contraindications": False,
                "has_precautions": True,
            },
            conversation_id=conversation_id
        )

@app.post("/ask-followup")
async def ask_followup(request: ChatRequest):
    """分析結果に基づいてフォローアップ質問に回答"""
    answer = await answer_followup_question(request.conversation_id, request.question)
    return {"answer": answer}

# メイン実行関数
if __name__ == "__main__":
    logger.info("薬剤併用禁忌チェックシステムを起動します")
    try:
        port = int(os.getenv("PORT", 10000))
        host = os.getenv("HOST", "0.0.0.0")
        logger.info(f"サーバーを {host}:{port} で起動します")
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"サーバー起動エラー: {str(e)}")
        logger.error(traceback.format_exc()) 