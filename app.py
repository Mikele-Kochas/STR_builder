import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Any

load_dotenv()

try:
	from openai import OpenAI
except Exception:
	OpenAI = None  # type: ignore


def get_client() -> Optional[Any]:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		try:
			# Fallback for Streamlit Cloud
			api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
		except Exception:
			api_key = None
	if not api_key or OpenAI is None:
		return None
	return OpenAI(api_key=api_key)


SYSTEM_PROMPT = (
	"You are an expert subtitle aligner and formatter.\n"
	"INPUTS:\n"
	"(A) A noisy subtitle-like text WITH CORRECT TIMESTAMPS and segmentation (order is correct).\n"
	"(B) A clean transcript WITHOUT timestamps.\n\n"
	"GOAL: Produce a valid SubRip (.srt) file.\n\n"
	"STRICT RULES:\n"
	"1) Use EXACT timestamps and EXACT segmentation from (A):\n"
	"   - Same number of blocks, same order, 1:1 mapping.\n"
	"   - Do NOT change any times. Do NOT merge or split blocks.\n"
	"2) Text content policy (very important):\n"
	"   - Start from block text in (A) as the baseline.\n"
	"   - Replace phrases with the most similar phrases from (B) ONLY if the meaning is identical\n"
	"     and the differences are orthographic: casing, diacritics, punctuation, spacing, numerals (e.g. 'rozdział drugi' ↔ 'Rozdział II'),\n"
	"     minor typos.\n"
	"   - If a phrase exists in (A) but NOT in (B) (e.g., 'czyta XYZ'), KEEP it from (A). DO NOT delete it.\n"
	"   - If a phrase exists in (B) but NOT in (A), DO NOT add it anywhere.\n"
	"   - If unsure whether a piece from (B) corresponds to (A), prefer keeping the (A) text.\n"
	"   - Do NOT shorten content from (A). No paraphrasing.\n"
	"3) Output format (SRT):\n"
	"   - 'HH:MM:SS,mmm --> HH:MM:SS,mmm' (exactly as in A)\n"
	"   - 1–2 lines of text for readability (you may insert line breaks, but do not change words).\n"
	"   - Blank line between blocks.\n\n"
	"Return ONLY the .srt content. No explanations. Do not include index for the timestamps. Only those elements that described in the prompt."
)


def one_call_srt(noisy_with_times: str, clean_without_times: str, client: Any) -> str:
	payload = {
		"noisy_with_times": noisy_with_times,
		"clean_without_times": clean_without_times,
	}
	resp = client.chat.completions.create(
		model="gpt-5-mini",
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
		]
	)
	try:
		return resp.choices[0].message.content or ""
	except Exception:
		return ""


st.set_page_config(page_title="SRT Builder", layout="wide")
st.title("SRT Builder")

col1, col2 = st.columns(2)
with col1:
	st.subheader("Tekst A: z timestampami")
	text_a = st.text_area("Wklej tekst A", height=360)
with col2:
	st.subheader("Tekst B: poprawny")
	text_b = st.text_area("Wklej tekst B", height=360)

run = st.button("Generuj SRT (one-shot)")

if run:
	if not text_a.strip():
		st.error("Brak tekstu A (z timestampami).")
		st.stop()
	if not text_b.strip():
		st.error("Brak tekstu B (poprawnego).")
		st.stop()
	client = get_client()
	if client is None:
		st.error("Brak OPENAI_API_KEY (env lub st.secrets) lub biblioteki openai.")
		st.stop()
	with st.spinner("Generuję SRT..."):
		srt_text = one_call_srt(text_a, text_b, client)
		if "-->" not in srt_text:
			st.warning("Model nie zwrócił treści w formacie SRT. Wyświetlam surową odpowiedź poniżej.")
		st.subheader("Wynik SRT")
		st.code(srt_text, language="text")
		st.download_button("Pobierz .srt", data=(srt_text or "").encode("utf-8"), file_name="aligned.srt", mime="text/plain")
