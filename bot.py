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
import sys
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

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "qty": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "qty", "reason"],
}


# ============================================================================
# 2. 安全ブレーキ（リミッター）
# ============================================================================
class SafetyLimiter:
    """AIの判断をそのまま発注せず、金額・回数の上限でチェックする安全装置。"""

    def __init__(self, max_order_amount: int, max_daily_trades: int):
        self.max_order_amount = max_order_amount
        self.max_daily_trades = max_daily_trades
        self.trade_count_today = 0

    def check(self, decision: dict, current_price: float) -> tuple[bool, str]:
        # HOLD（様子見）は常に許可
        if decision["decision"] == "HOLD":
            return True, "AIはHOLD（様子見）と判断。発注はスキップします。"

        # 数量が0以下は不正
        qty = decision.get("qty", 0)
        if qty <= 0:
            return False, f"数量が不正です（qty={qty}）。発注を中止します。"

        # 1日の取引回数制限
        if self.trade_count_today >= self.max_daily_trades:
            return False, (
                f"本日の取引回数上限（{self.max_daily_trades}回）に達しています。"
            )

        # 発注金額の上限チェック
        order_amount = current_price * qty
        if order_amount > self.max_order_amount:
            return False, (
                f"発注金額 {order_amount:,.0f}円 が上限 "
                f"{self.max_order_amount:,.0f}円 を超えています。"
            )

        return True, (
            f"チェックOK: {decision['decision']} {qty}株 "
            f"（約{order_amount:,.0f}円）"
        )

    def record_trade(self):
        self.trade_count_today += 1


# ============================================================================
# 3. 証券会社アダプター（アダプターパターンの本体）
# ============================================================================
class BrokerAdapter(ABC):
    """証券会社ごとの発注APIの違いを吸収する共通インターフェース。"""

    @abstractmethod
    def place_order(self, symbol: str, decision: dict, current_price: float) -> bool:
        """共通データ構造(decision)を、その証券会社の発注形式に翻訳して送信する。"""
        raise NotImplementedError

    @abstractmethod
    def get_balance_info(self) -> str:
        raise NotImplementedError


class VirtualBrokerAdapter(BrokerAdapter):
    """実際には発注せず、メモリ上でシミュレーションするだけの仮想証券会社。"""

    def __init__(self, initial_balance: float = 1_000_000):
        self.balance = initial_balance
        self.holdings: dict[str, int] = {}

    def place_order(self, symbol: str, decision: dict, current_price: float) -> bool:
        action = decision["decision"]
        qty = decision.get("qty", 0)

        if action == "HOLD":
            print(f"  [仮想証券] {symbol}: HOLD（発注なし）")
            return True

        cost = current_price * qty

        if action == "BUY":
            if cost > self.balance:
                print(f"  [仮想証券] 残高不足のため発注失敗（必要額: {cost:,.0f}円）")
                return False
            self.balance -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + qty
            print(f"  [仮想証券] {symbol} を {qty}株 買付（{cost:,.0f}円）")
            return True

        if action == "SELL":
            held = self.holdings.get(symbol, 0)
            if qty > held:
                print(f"  [仮想証券] 保有数不足のため発注失敗（保有: {held}株）")
                return False
            self.balance += cost
            self.holdings[symbol] = held - qty
            print(f"  [仮想証券] {symbol} を {qty}株 売却（{cost:,.0f}円）")
            return True

        print(f"  [仮想証券] 不明な指示のため発注失敗: {action}")
        return False

    def get_balance_info(self) -> str:
        return f"仮想現金残高: {self.balance:,.0f}円 / 保有銘柄: {self.holdings}"


class TachibanaDemoAdapter(BrokerAdapter):
    """立花証券デモAPI向けの骨組み（実装は今後追加）。"""

    def place_order(self, symbol: str, decision: dict, current_price: float) -> bool:
        print("[立花デモ] 発注APIは未実装のため、発注をスキップしました。")
        print(f"  （本来送るはずだった内容: {symbol}, {decision}）")
        return False

    def get_balance_info(self) -> str:
        return "[立花デモ] 残高照会APIは未実装（骨組みのみ）"


# ============================================================================
# 4. Gemini APIとの連携（AIに「共通データ構造」で判断させる）
# ============================================================================
def get_stock_data(symbol: str) -> dict:
    """yfinanceで直近の株価データを取得する。"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d")

    if hist.empty:
        raise ValueError(f"銘柄 {symbol} の株価データが取得できませんでした。")

    current_price = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
    change_pct = (current_price - prev_close) / prev_close * 100 if prev_close else 0.0

    return {
        "symbol": symbol,
        "current_price": round(current_price, 1),
        "previous_close": round(prev_close, 1),
        "change_pct": round(change_pct, 2),
        "volume": int(hist["Volume"].iloc[-1]),
        "recent_high": round(float(hist["High"].max()), 1),
        "recent_low": round(float(hist["Low"].min()), 1),
        "timestamp": datetime.datetime.now().isoformat(),
    }


def get_ai_decision(client: genai.Client, stock_data: dict) -> dict:
    """Geminiに株価データを渡し、共通データ構造(JSON)で売買判断を返させる。"""
    prompt = f"""
あなたは日本株のトレーディングアシスタントです。
以下の株価データだけを根拠に、売買判断を行ってください。

銘柄: {stock_data['symbol']}
現在値: {stock_data['current_price']}円
前日終値: {stock_data['previous_close']}円
騰落率: {stock_data['change_pct']}%
出来高: {stock_data['volume']}
直近高値: {stock_data['recent_high']}円
直近安値: {stock_data['recent_low']}円

指定されたJSON形式のみで回答してください。
qtyは100株単位（単元株）にしてください。HOLDの場合はqtyを0にしてください。
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DECISION_SCHEMA,
        ),
    )

    try:
        decision = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Geminiの応答をJSONとして解析できませんでした: {e}\n応答内容: {response.text}")

    # 必須キーの検証
    for key in ("decision", "qty", "reason"):
        if key not in decision:
            raise ValueError(f"Geminiの応答に必須キー '{key}' がありません: {decision}")

    return decision


# ============================================================================
# 5. メイン処理
# ============================================================================
def main():
    print("=" * 70)
    print(" 日本株AI自動売買システム（検証・シミュレーション用）")
    print("=" * 70)
sys.exit()
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

    try:
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

    except Exception as e:
        print(f"\n[エラー] 処理中に例外が発生しました: {e}")


if __name__ == "__main__":
    main()
