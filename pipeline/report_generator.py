"""
report_generator.py — Call Gemini to generate the structured research note.
"""

import os
import json
import re
import google.generativeai as genai


def generate_report(
    ticker: str,
    contexts: dict[str, str],
    focus_areas: list[str],
) -> dict:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")

    focus_str = ", ".join(focus_areas) if focus_areas else "general analysis"

    # Build explicit per-area rules so Gemini actually weights them
    focus_instructions = ""
    if focus_areas:
        rules = []
        if "Revenue & Growth" in focus_areas:
            rules.append("- At least 2 bull_thesis points must discuss revenue figures, growth rates, or guidance.")
        if "Risk Factors" in focus_areas:
            rules.append("- At least 2 key_risks must be specific named risk factors from the filing.")
        if "Competition" in focus_areas:
            rules.append("- At least 1 bear_thesis point and 1 key_risk must address competitive dynamics or market share.")
        if "Management Discussion" in focus_areas:
            rules.append("- The overview must include management's own commentary on outlook or strategy.")
        if "Cash Flow" in focus_areas:
            rules.append("- At least 1 bull_thesis point must cite free cash flow, operating cash flow, or capex figures.")
        if "Debt & Leverage" in focus_areas:
            rules.append("- analyst_notes must comment on the debt profile, leverage ratio, or interest expense.")
        focus_instructions = "\n\nFOCUS AREA RULES (strictly follow these):\n" + "\n".join(rules)

    prompt = f"""You are a senior equity research analyst at a top-tier hedge fund.
Write a structured research note for {ticker} based ONLY on the filing excerpts below.
Be specific — cite actual numbers, percentages, and facts from the text.
Do NOT hallucinate metrics not mentioned in the excerpts.

Focus areas requested: {focus_str}{focus_instructions}

---
FILING EXCERPTS — COMPANY OVERVIEW:
{contexts.get('overview', 'Not available')}

---
FILING EXCERPTS — GROWTH & OPPORTUNITY SIGNALS:
{contexts.get('bull', 'Not available')}

---
FILING EXCERPTS — CHALLENGES & HEADWINDS:
{contexts.get('bear', 'Not available')}

---
FILING EXCERPTS — RISK FACTORS:
{contexts.get('risks', 'Not available')}

---

Return ONLY a valid JSON object, no markdown, no backticks:
{{
  "overview": "2-3 sentence company overview with key business segments and recent performance",
  "bull_thesis": [
    "Point 1 with specific data from the filing",
    "Point 2 with specific data from the filing",
    "Point 3 with specific data from the filing"
  ],
  "bear_thesis": [
    "Point 1 with specific challenge or headwind",
    "Point 2 with specific challenge or headwind",
    "Point 3 with specific challenge or headwind"
  ],
  "key_risks": [
    "Risk 1 — specific risk factor from the filing",
    "Risk 2 — specific risk factor",
    "Risk 3 — specific risk factor"
  ],
  "sentiment_score": <integer 1-10, where 10 is extremely bullish>,
  "recommendation": "<BUY|HOLD|SELL>",
  "analyst_notes": "1-2 sentences of nuanced analyst commentary not captured above"
}}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=8192,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[report_generator] JSON parse error: {e}\nRaw output: {raw[:500]}")
        return _fallback_report(ticker)
    except Exception as e:
        print(f"[report_generator] Gemini error: {e}")
        return _fallback_report(ticker)


def _fallback_report(ticker: str) -> dict:
    return {
        "overview": f"Analysis of {ticker} could not be completed. Please check your API key and try again.",
        "bull_thesis": ["Analysis unavailable"],
        "bear_thesis": ["Analysis unavailable"],
        "key_risks": ["Analysis unavailable"],
        "sentiment_score": 5,
        "recommendation": "HOLD",
        "analyst_notes": "Report generation failed. Please retry.",
    }