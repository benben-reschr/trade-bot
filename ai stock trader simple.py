import os
import json
import yfinance as yf
from google import genai
from google.genai import types

# ============================================================
# 日本株AI自動売買（シミュレーション用・シンプル版）
# ============================================================

# 設定
SYMBOL = "7203.T"          # 対象銘柄（トヨタ自動車）
MAX_ORDER_AMOUNT = 300_000  # 1回の発注金額の上限（円）
INITIAL_BALANCE = 1_000_000  # 仮想の初期資金（円）

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "qty": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "qty", "reason"],
}


def get_stock_data(symbol: str) -> dict:
    """直近の株価データを取得する。"""
    hist = yf.Ticker(symbol).history(period="5d")
    if hist.empty:
        raise ValueError(f"銘柄 {symbol} の株価データが取得できませんでした。")

    current_price = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price

    return {
        "symbol": symbol,
        "current_price": round(current_price, 1),
        "previous_close": round(prev_close, 1),
        "change_pct": round((current_price - prev_close) / prev_close * 100, 2) if prev_close else 0.0,
    }


def get_ai_decision(client: genai.Client, stock_data: dict) -> dict:
    """Geminiに売買判断をさせ、JSONで受け取る。"""
    prompt = f"""
あなたは日本株のトレーディングアシスタントです。
以下の株価データだけを根拠に、売買判断を行ってください。

銘柄: {stock_data['symbol']}
現在値: {stock_data['current_price']}円
前日終値: {stock_data['previous_close']}円
騰落率: {stock_data['change_pct']}%

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
    return json.loads(response.text)


def check_order(decision: dict, current_price: float) -> tuple[bool, str]:
    """発注してよいかどうかの簡単な安全チェック。"""
    if decision["decision"] == "HOLD":
        return True, "HOLD（様子見）のため発注はスキップします。"

    qty = decision.get("qty", 0)
    if qty <= 0:
        return False, f"数量が不正です（qty={qty}）。"

    amount = current_price * qty
    if amount > MAX_ORDER_AMOUNT:
        return False, f"発注金額 {amount:,.0f}円 が上限 {MAX_ORDER_AMOUNT:,.0f}円 を超えています。"

    return True, f"チェックOK: {decision['decision']} {qty}株（約{amount:,.0f}円）"


def place_virtual_order(state: dict, symbol: str, decision: dict, current_price: float) -> None:
    """仮想証券口座で売買をシミュレーションする（実際には発注しない）。"""
    action = decision["decision"]
    qty = decision.get("qty", 0)

    if action == "HOLD":
        print("  [仮想証券] HOLD（発注なし）")
        return

    cost = current_price * qty

    if action == "BUY":
        if cost > state["balance"]:
            print(f"  [仮想証券] 残高不足のため発注失敗（必要額: {cost:,.0f}円）")
            return
        state["balance"] -= cost
        state["holdings"][symbol] = state["holdings"].get(symbol, 0) + qty
        print(f"  [仮想証券] {symbol} を {qty}株 買付（{cost:,.0f}円）")

    elif action == "SELL":
        held = state["holdings"].get(symbol, 0)
        if qty > held:
            print(f"  [仮想証券] 保有数不足のため発注失敗（保有: {held}株）")
            return
        state["balance"] += cost
        state["holdings"][symbol] = held - qty
        print(f"  [仮想証券] {symbol} を {qty}株 売却（{cost:,.0f}円）")


def main():
    print("=" * 60)
    print(" 日本株AI自動売買（シミュレーション・シンプル版）")
    print("=" * 60)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[エラー] 環境変数 GEMINI_API_KEY が設定されていません。")
        return

    client = genai.Client(api_key=api_key)
    state = {"balance": INITIAL_BALANCE, "holdings": {}}

    try:
        print(f"\n[1/3] {SYMBOL} の株価データを取得しています...")
        stock_data = get_stock_data(SYMBOL)
        print(f"  現在値: {stock_data['current_price']}円")

        print("\n[2/3] Geminiに売買判断をリクエストしています...")
        decision = get_ai_decision(client, stock_data)
        print(f"  AIの判断: {decision}")

        ok, message = check_order(decision, stock_data["current_price"])
        print(f"  安全チェック: {message}")
        if not ok:
            print("\n[停止] 発注はブロックされました。")
            return

        print("\n[3/3] 仮想証券口座で発注を実行しています...")
        place_virtual_order(state, SYMBOL, decision, stock_data["current_price"])

        print("\n" + "-" * 60)
        print(f"仮想現金残高: {state['balance']:,.0f}円 / 保有銘柄: {state['holdings']}")
        print("-" * 60)

    except Exception as e:
        print(f"\n[エラー] 処理中に例外が発生しました: {e}")


if __name__ == "__main__":
    main()
