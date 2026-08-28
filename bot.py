"""
================================================================================
 日本株AI自動売買システム（検証・シミュレーション用）
================================================================================

【スマホ Codespaces での準備】
Terminal を開いて、まず以下のコマンドを実行してライブラリをインストールしてください。

    pip install google-genai yfinance pandas

その後、Gemini APIキーを環境変数にセットしてください（Codespaces の Secrets 機能でも可）。

    export GEMINI_API_KEY="ここに自分のAPIキーを貼る"

準備ができたら、以下のコマンドで実行します。

    python ai_trading_simulator.py

================================================================================
 【設計思想：データ翻訳マッピング（アダプターパターン）】
================================================================================
証券会社ごとに注文APIが要求するJSON構造はバラバラです（例：立花証券は
issueCode / orderSide="1" のような独自パラメータを要求します）。

このシステムでは、AI（Gemini）には常に「共通のシンプルな指示」だけを
出させます。例えば以下のような形です。

    { "decision": "BUY", "qty": 100, "reason": "..." }

そして、この「共通指示」を実際の証券会社ごとの注文フォーマットに変換する
役目を「アダプター（翻訳者）」に担わせます。AIロジックや売買判断ロジックは
一切変更せずに、この翻訳者（アダプター）だけを差し替えれば、接続先の証券会社を
自由に切り替えられる、という設計になっています。
================================================================================
"""

import os
import json
import datetime
from abc import ABC, abstractmethod

import yfinance as yf
from google import genai
from google.genai import types


# ============================================================================
# 1. 共通データ構造（AIが必ずこの形式で判断を返す）
# ============================================================================
# Geminiにはこの3つのキーだけを含むJSONを出力させます。
# これが「証券会社に依存しない、共通の言語」になります。
#
#   decision : "BUY"（買い）/ "SELL"（売り）/ "HOLD"（様子見・何もしない）
#   qty      : 株数（整数。日本株は通常100株単位＝単元株）
#   reason   : AIがそう判断した理由（人間が確認するためのメモ）
# ============================================================================


# ============================================================================
# 2. 安全ブレーキ（リミッター）
# ============================================================================
class SafetyLimiter:
    """
    AIの判断をそのまま証券会社アダプターに渡すのは危険なので、
    その手前に必ずこのクラスを通します。

    - 1回あたりの発注金額が上限を超えていないか
    - 1日の取引回数が上限を超えていないか

    のチェックを行い、危険な発注は「物理的に」ブロックします。
    """

    def __init__(self, max_order_amount: int = 300_000, max_daily_trades: int = 5):
        self.max_order_amount = max_order_amount      # 1回あたりの最大投資上限額（円）
        self.max_daily_trades = max_daily_trades        # 1日の最大取引回数
        self.trade_count_today = 0                      # 本日すでに行った取引回数
        self.today = datetime.date.today()               # 日付が変わったらカウンターをリセットするために保持

    def _reset_if_new_day(self):
        """日付が変わっていたら、取引回数カウンターを0にリセットする"""
        current_date = datetime.date.today()
        if current_date != self.today:
            self.today = current_date
            self.trade_count_today = 0
            print(f"[安全装置] 日付が変わったため、取引回数カウンターをリセットしました。")

    def check(self, decision: dict, current_price: float) -> tuple[bool, str]:
        """
        発注前チェック。
        戻り値: (発注してよいか True/False, 理由メッセージ)
        """
        self._reset_if_new_day()

        # HOLD（様子見）は常に許可（発注自体が発生しないため）
        if decision["decision"] == "HOLD":
            return True, "HOLDのため発注なし"

        # --- チェック1: 1日の取引回数上限 ---
        if self.trade_count_today >= self.max_daily_trades:
            return False, (
                f"本日の取引回数が上限（{self.max_daily_trades}回）に達しているため、"
                f"本日の自動売買を停止します。"
            )

        # --- チェック2: 1回あたりの投資金額上限 ---
        order_amount = current_price * decision["qty"]
        if order_amount > self.max_order_amount:
            return False, (
                f"発注金額 {order_amount:,.0f}円 が上限 "
                f"{self.max_order_amount:,.0f}円 を超えているためブロックしました。"
            )

        return True, "安全チェックOK"

    def record_trade(self):
        """実際に発注が実行されたら、取引回数カウンターを1つ増やす"""
        self.trade_count_today += 1


# ============================================================================
# 3. 証券会社アダプター（アダプターパターンの本体）
# ============================================================================
class BrokerAdapter(ABC):
    """
    すべての証券会社アダプターが必ず実装しなければならない「共通の窓口」。
    place_order() は常に「共通データ構造」を受け取る前提で設計する。
    証券会社ごとの独自フォーマットへの変換は、このメソッドの「内部」で行う。
    """

    @abstractmethod
    def place_order(self, symbol: str, decision: dict, current_price: float):
        ...

    @abstractmethod
    def get_balance_info(self) -> str:
        ...


class VirtualBrokerAdapter(BrokerAdapter):
    """
    【スマホ検証用】仮想お財布アダプター。
    実際の証券会社には接続せず、変数（メモリ上）で残高と保有株数を自己管理する。
    100万円の仮想資金からスタートし、AIの判断に沿って売買のシミュレーションを行う。
    """

    def __init__(self, initial_balance: int = 1_000_000):
        self.balance = initial_balance          # 仮想の現金残高
        self.holdings: dict[str, int] = {}       # 銘柄コードごとの保有株数 {"7203.T": 100}
        self.history: list[dict] = []            # 売買履歴のログ

    def place_order(self, symbol: str, decision: dict, current_price: float):
        # ---- ここが「翻訳」部分：仮想ブローカーの場合は、共通データをそのまま
        #      内部の残高・保有株数の増減計算に変換するだけでよい ----
        qty = decision["qty"]
        amount = current_price * qty

        if decision["decision"] == "BUY":
            if amount > self.balance:
                print(f"  [仮想ブローカー] 残高不足のため発注できません（残高: {self.balance:,.0f}円）")
                return False
            self.balance -= amount
            self.holdings[symbol] = self.holdings.get(symbol, 0) + qty
            print(f"  [仮想ブローカー] {symbol} を {qty}株 買付 → {amount:,.0f}円 支払い")

        elif decision["decision"] == "SELL":
            held = self.holdings.get(symbol, 0)
            if held < qty:
                print(f"  [仮想ブローカー] 保有株数不足のため売却できません（保有: {held}株）")
                return False
            self.balance += amount
            self.holdings[symbol] = held - qty
            print(f"  [仮想ブローカー] {symbol} を {qty}株 売却 → {amount:,.0f}円 受取")

        else:
            print("  [仮想ブローカー] HOLDのため何もしません")
            return True

        self.history.append({
            "time": datetime.datetime.now().isoformat(),
            "symbol": symbol,
            **decision,
            "price": current_price,
        })
        return True

    def get_balance_info(self) -> str:
        return f"仮想現金残高: {self.balance:,.0f}円 / 保有銘柄: {self.holdings}"


class TachibanaDemoAdapter(BrokerAdapter):
    """
    【将来用・骨組みのみ】立花証券デモ環境用アダプター。

    実際に発注APIへ接続するには、立花証券が提供するAPI仕様書に基づいて
    ログイン・セッショントークンの取得などが別途必要になります。
    ここでは「共通データ → 立花証券独自のJSON構造への翻訳マッピング」の
    骨組みだけを示しています（実際の送信処理はコメントアウトしています）。
    """

    # 立花証券APIは銘柄コードの前にゼロ埋めなどのルールがある場合があるため、
    # symbol（例: "7203.T"）から数字部分だけを取り出すヘルパー
    @staticmethod
    def _to_issue_code(symbol: str) -> str:
        return symbol.split(".")[0]  # "7203.T" -> "7203"

    def place_order(self, symbol: str, decision: dict, current_price: float):
        # ------------------------------------------------------------------
        # ★★★ ここが「翻訳マッピング」の核心部分 ★★★
        # 共通データ {"decision": "BUY", "qty": 100} を、
        # 立花証券が実際に要求するであろうJSON構造へ変換する。
        # （フィールド名やコード値は仕様書のイメージに沿ったサンプルです）
        # ------------------------------------------------------------------
        order_side_map = {
            "BUY": "1",   # 立花証券APIでは「買い」を "1" という文字列コードで表す想定
            "SELL": "2",  # 「売り」は "2" という文字列コードで表す想定
        }

        if decision["decision"] == "HOLD":
            print("  [立花デモ] HOLDのため発注リクエストは作成しません")
            return True

        tachibana_payload = {
            "issueCode": self._to_issue_code(symbol),       # 銘柄コード（証券会社独自の項目名）
            "orderSide": order_side_map[decision["decision"]],  # "1"（買）または "2"（売）
            "orderQuantity": str(decision["qty"]),             # 数量は文字列で渡す想定
            "orderPrice": "0",                                  # 成行注文の想定（0=成行のことが多い）
            "orderType": "0",                                   # 執行条件（0=成行 のサンプル）
        }

        print("  [立花デモ] 共通データ → 立花フォーマットへ翻訳しました:")
        print(f"    {json.dumps(tachibana_payload, ensure_ascii=False)}")

        # ------------------------------------------------------------------
        # 実際にAPIへ送信する場合は、以下のようなイメージになります。
        # （デモ環境が利用可能になった時点で、認証情報を設定して有効化してください）
        #
        # import requests
        # response = requests.post(
        #     "https://demo-kabuka.e-shiten.jp/e_api_v4r5/receiveorder",
        #     json=tachibana_payload,
        #     headers={"Authorization": f"Bearer {YOUR_TACHIBANA_TOKEN}"},
        # )
        # response.raise_for_status()
        # ------------------------------------------------------------------

        print("  [立花デモ] ※現在は骨組みのみのため、実際の送信は行っていません。")
        return True

    def get_balance_info(self) -> str:
        return "[立花デモ] 残高照会APIは未実装（骨組みのみ）"


# ============================================================================
# 4. Gemini APIとの連携（AIに「共通データ構造」で判断させる）
# ============================================================================
def get_stock_data(symbol: str) -> dict:
    """
    yfinanceを使って日本株の直近データを取得する。
    symbol例: "7203.T"（トヨタ自動車）
    """
    ticker = yf.Ticker(symbol)
    # 直近10営業日分の日足データを取得
    hist = ticker.history(period="10d")

    if hist.empty:
        raise ValueError(f"{symbol} のデータが取得できませんでした。銘柄コードを確認してください。")

    latest = hist.iloc[-1]
    return {
        "symbol": symbol,
        "current_price": float(latest["Close"]),
        "recent_close_prices": [round(float(p), 1) for p in hist["Close"].tolist()],
        "recent_volumes": [int(v) for v in hist["Volume"].tolist()],
    }


def get_ai_decision(client: genai.Client, stock_data: dict) -> dict:
    """
    Geminiに株価データを渡して、売買判断を「共通データ構造」のJSONで受け取る。

    ポイント：
    - response_mime_type="application/json" を指定することで、
      Geminiに「JSON以外の説明文（前置き等）を一切出力させない」ようにする。
    - プロンプト内でも出力すべきJSONのキーを明示し、AIの出力形式をブレさせない。
    """
    prompt = f"""
あなたは日本株のテクニカル分析アシスタントです。
以下の株価データをもとに、売買判断を行ってください。

銘柄コード: {stock_data['symbol']}
現在値: {stock_data['current_price']}円
直近の終値の推移（古い→新しい順）: {stock_data['recent_close_prices']}
直近の出来高の推移（古い→新しい順）: {stock_data['recent_volumes']}

必ず以下のJSON形式のみで回答してください（他の文章は一切不要です）。
- decision: "BUY"（買い） / "SELL"（売り） / "HOLD"（様子見）のいずれか
- qty: 発注する株数（100株単位の整数。取引しない場合は0）
- reason: 判断理由を日本語で簡潔に（50文字程度）
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    # Geminiはここで「綺麗なJSON文字列」のみを返してくる想定なので、そのままパースする
    decision = json.loads(response.text)

    # 万が一AIが不正な形式で返してきた場合に備えた最低限のバリデーション
    if decision.get("decision") not in ("BUY", "SELL", "HOLD"):
        raise ValueError(f"AIから予期しない形式のレスポンスが返りました: {decision}")
    decision["qty"] = int(decision.get("qty", 0))

    return decision


# ============================================================================
# 5. メイン処理
# ============================================================================
def main():
    print("=" * 70)
    print(" 日本株AI自動売買システム（検証・シミュレーション用）")
    print("=" * 70)

    # ------------------------------------------------------------------
    # ★★★ ここの1行を書き替えるだけで、接続先の証券会社を切り替えられます ★★★
    # ------------------------------------------------------------------
    my_broker: BrokerAdapter = VirtualBrokerAdapter()
    # my_broker: BrokerAdapter = TachibanaDemoAdapter()   # 将来、立花デモに切り替える場合はこちら
    # ------------------------------------------------------------------

    limiter = SafetyLimiter(max_order_amount=300_000, max_daily_trades=5)

    # 取引対象の銘柄（例: トヨタ自動車）
    target_symbol = "7203.T"

    # Gemini APIクライアントの初期化
    # GEMINI_API_KEY は環境変数（またはCodespacesのSecrets）から自動的に読み込まれます
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[エラー] 環境変数 GEMINI_API_KEY が設定されていません。")
        print("  export GEMINI_API_KEY=\"あなたのAPIキー\" を実行してから再度お試しください。")
        return

    client = genai.Client(api_key=api_key)

    # --- ステップ1: 株価データの取得 ---
    print(f"\n[1/4] {target_symbol} の株価データを取得しています...")
    stock_data = get_stock_data(target_symbol)
    print(f"  現在値: {stock_data['current_price']}円")

    # --- ステップ2: Geminiに売買判断をさせる（共通データ構造で受け取る） ---
    print("\n[2/4] Geminiに売買判断をリクエストしています...")
    decision = get_ai_decision(client, stock_data)
    print(f"  AIの判断: {decision}")

    # --- ステップ3: 安全フィルターを通す ---
    print("\n[3/4] 安全フィルターでチェックしています...")
    ok, message = limiter.check(decision, stock_data["current_price"])
    print(f"  結果: {message}")

    if not ok:
        print("\n[停止] 安全フィルターにより発注はブロックされました。処理を終了します。")
        return

    # --- ステップ4: アダプター経由で発注（ここで初めて証券会社ごとのJSON翻訳が行われる） ---
    print("\n[4/4] アダプター経由で注文を実行しています...")
    success = my_broker.place_order(target_symbol, decision, stock_data["current_price"])

    if success and decision["decision"] != "HOLD":
        limiter.record_trade()

    print("\n" + "-" * 70)
    print(my_broker.get_balance_info())
    print(f"本日の取引回数: {limiter.trade_count_today} / {limiter.max_daily_trades}")
    print("-" * 70)


if __name__ == "__main__":
    main()
