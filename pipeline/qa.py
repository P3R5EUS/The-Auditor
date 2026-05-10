"""
qa.py — Answer follow-up questions about a filing using Gemini.
Uses start_chat() for real multi-turn conversational memory.
"""

import os
import google.generativeai as genai


def answer_question(
    ticker: str,
    question: str,
    context: str,
    history: list[dict],
) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=f"""You are a hedge fund analyst assistant specializing in {ticker}.
Answer questions using ONLY the provided filing excerpts.
Be specific — cite numbers and facts. If information isn't in the excerpts, say so clearly."""
    )

    # Build real Gemini chat history (last 4 turns)
    gemini_history = []
    for h in history[-4:]:
        gemini_history.append({"role": "user", "parts": [h["q"]]})
        gemini_history.append({"role": "model", "parts": [h["a"]]})

    chat = model.start_chat(history=gemini_history)

    message = f"""Relevant filing excerpts:
---
{context}
---

Question: {question}"""

    try:
        response = chat.send_message(
            message,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}. Please check your API key."