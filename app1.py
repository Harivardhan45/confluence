import os
import re
import streamlit as st
from dotenv import load_dotenv
from atlassian import Confluence
from fpdf import FPDF
from io import BytesIO
import google.generativeai as genai
from moviepy import VideoFileClip
from faster_whisper import WhisperModel
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Helper
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text).encode('latin-1', 'ignore').decode('latin-1')

# Init
@st.cache_resource
def init_confluence():
    try:
        return Confluence(
            url=os.getenv("CONFLUENCE_BASE_URL"),
            username=os.getenv("CONFLUENCE_USER_EMAIL"),
            password=os.getenv("CONFLUENCE_API_KEY"),
            timeout=30
        )
    except Exception as e:
        st.error(f"Confluence init failed: {e}")
        return None

genai.configure(api_key=os.getenv("GENAI_API_KEY"))
ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")

# Main UI
st.title("📄 Confluence Video Summarizer")
confluence = init_confluence()

if confluence:
    st.success("✅ Connected to Confluence!")

    space_key = st.text_input("Enter your Confluence Space Key:")
    if space_key:
        try:
            pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
            page_titles = [p["title"] for p in pages]
            selected_pages = st.multiselect("Select Pages to Process:", page_titles)
            if selected_pages:
                summaries = []
                for page in pages:
                    if page["title"] not in selected_pages:
                        continue
                    title = page["title"]
                    page_id = page["id"]
                    st.markdown(f"### 🎬 Processing: {title}")
                    attachments = confluence.get(f"/rest/api/content/{page_id}/child/attachment?limit=50")
                    for attachment in attachments["results"]:
                        video_name = attachment["title"].strip()
                        if not video_name.lower().endswith(".mp4"):
                            continue
                        session_key = f"{page_id}_{video_name}".replace(" ", "_")
                        if session_key in st.session_state:
                            st.info(f"🟡 Cached summary found for {video_name}")
                        else:
                            with st.spinner("📥 Downloading and processing..."):
                                try:
                                    video_url = attachment["_links"]["download"]
                                    full_url = f"{os.getenv('CONFLUENCE_BASE_URL').rstrip('/')}{video_url}"
                                    local_path = f"{title}_{video_name}".replace(" ", "_")
                                    video_data = confluence._session.get(full_url).content
                                    with open(local_path, "wb") as f:
                                        f.write(video_data)
                                    video = VideoFileClip(local_path)
                                    video.audio.write_audiofile("temp_audio.mp3")
                                    model = WhisperModel("tiny", device="cpu", compute_type="int8")
                                    segments, _ = model.transcribe("temp_audio.mp3")
                                    transcript = "\n".join(
                                        f"[{int(s.start // 60)}:{int(s.start % 60):02}] {s.text}" for s in segments
                                    )
                                    quote_prompt = f"Set a title \"Quotes:\" in bold. Extract powerful or interesting quotes:\n{transcript}"
                                    quotes = ai_model.generate_content(quote_prompt).text
                                    summary_prompt = (
                                        f"Start with title as \"Summary:\" in bold, followed by a paragraph.\n"
                                        f"Then add title \"Timestamps:\" and list bullet points with [min:sec]:\n{transcript}"
                                    )
                                    summary = ai_model.generate_content(summary_prompt).text
                                    st.session_state[session_key] = {
                                        "transcript": transcript,
                                        "summary": summary,
                                        "quotes": quotes
                                    }
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                                    continue

                        content = st.session_state[session_key]
                        st.markdown(f"#### 📄 {video_name}")
                        st.markdown(content["quotes"])
                        st.markdown(content["summary"])
                        summaries.append((f"{title}_{video_name}", content["summary"], content["quotes"]))

                if summaries:
                    st.markdown("## 📦 Export All Summaries")
                    all_text = ""
                    for t, s, q in summaries:
                        all_text += f"\n\n---\n\n{t}\n\nQuotes:\n{q}\n\nSummary:\n{s}\n"

                    file_name = st.text_input("Filename (without extension):", value="All_Summaries")
                    export_format = st.selectbox("Format:", ["PDF", "TXT"])
                    if export_format == "PDF":
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        for line in remove_emojis(all_text).split("\n"):
                            pdf.multi_cell(0, 10, line)
                        file_data = pdf.output(dest="S").encode("latin-1")
                        mime = "application/pdf"
                        ext = "pdf"
                    else:
                        file_data = all_text.encode("utf-8")
                        mime = "text/plain"
                        ext = "txt"

                    st.download_button(
                        label=f"📥 Download All as {ext.upper()}",
                        data=BytesIO(file_data),
                        file_name=f"{file_name.strip() or 'All_Summaries'}.{ext}",
                        mime=mime
                    )
        except Exception as e:
            st.error(f"Error loading pages: {e}")
else:
    st.error("❌ Could not connect to Confluence.")
