# Optimized Confluence Video Summarizer (No Duplicate Summary, CPU-Friendly)
import os
import streamlit as st
from dotenv import load_dotenv
from atlassian import Confluence
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from fpdf import FPDF
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# Load environment
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))

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

st.title("📄 Confluence Video Summarizer (Fast Mode)")

confluence = init_confluence()
ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")

if confluence:
    st.success("✅ Connected to Confluence!")

    space_key = st.text_input("Enter your space key:")
    if space_key:
        try:
            pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
            page_titles = [page["title"] for page in pages]
            selected_pages = st.multiselect("Select Confluence Pages:", page_titles)

            if selected_pages:
                summaries = []
                displayed_sessions = set()

                for title in selected_pages:
                    page = next((p for p in pages if p["title"] == title), None)
                    if page:
                        page_id = page["id"]
                        st.markdown(f"---\n### 🎬 Processing: {title}")
                        try:
                            attachments = confluence.get(f"/rest/api/content/{page_id}/child/attachment?limit=50")
                            for attachment in attachments["results"]:
                                video_name = attachment["title"].strip()
                                if video_name.lower().endswith(".mp4"):
                                    session_key = f"{page_id}{attachment['id']}".replace(" ", "")

                                    if session_key not in st.session_state:
                                        with st.spinner("📥 Downloading and processing video..."):
                                            video_url = attachment["_links"]["download"]
                                            full_url = f"{os.getenv('CONFLUENCE_BASE_URL').rstrip('/')}{video_url}"
                                            video_data = confluence._session.get(full_url).content
                                            local_path = f"{title}{video_name}".replace(" ", "")

                                            with open(local_path, "wb") as f:
                                                f.write(video_data)

                                            video_clip = VideoFileClip(local_path)
                                            duration = min(120, video_clip.duration)
                                            video = video_clip.subclip(0, duration)
                                            video.audio.write_audiofile("temp_audio.mp3")

                                            model = WhisperModel("tiny", device="cpu", compute_type="int8")
                                            segments, _ = model.transcribe("temp_audio.mp3")
                                            transcript_lines = [
                                                f"[{int(s.start // 60)}:{int(s.start % 60):02}] {s.text}" for s in segments
                                            ]
                                            full_transcript = "\n".join(transcript_lines)
                                            short_transcript = full_transcript[:3500]

                                            quote_prompt = f"Set a title \"Quotes:\" in bold. Extract powerful or interesting quotes:\n{full_transcript}"
                                            quotes = ai_model.generate_content(quote_prompt).text
                                            summary_prompt = (
                                                    f"Start with title as \"Summary:\" in bold, followed by a paragraph.\n"
                                                    f"Then add title \"Timestamps:\" and list bullet points with imporatnt timestamps [min:sec] and small one line summarized description of each\n{full_transcript}"
                                                )

                                            quotes = ai_model.generate_content(quote_prompt).text
                                            summary = ai_model.generate_content(summary_prompt).text

                                            st.session_state[session_key] = {
                                                "transcript": full_transcript,
                                                "summary": summary,
                                                "quotes": quotes
                                            }

                                    if session_key not in displayed_sessions:
                                        displayed_sessions.add(session_key)

                                        transcript = st.session_state[session_key]["transcript"]
                                        summary = st.session_state[session_key]["summary"]
                                        quotes = st.session_state[session_key]["quotes"]
                                        summary_title = f"{title} - {video_name}"
                                        summaries.append((summary_title, summary, quotes))

                                        st.markdown(f"### 📄 {summary_title}")
                                        st.markdown(quotes)
                                        st.markdown(summary)

                                        file_base = f"{summary_title.replace(' ', '')}".replace(":", "").replace("/", "")
                                        format_choice = st.selectbox(
                                            f"Download format for {summary_title}", ["PDF", "TXT"], key=f"format_{file_base}"
                                        )

                                        export_content = f"Top Quotes:\n{quotes}\n\nSummary with Timestamps:\n{summary}"

                                        if format_choice == "PDF":
                                            pdf = FPDF()
                                            pdf.add_page()
                                            pdf.set_font("Arial", size=12)
                                            for line in export_content.split('\n'):
                                                # ✅ Unicode-safe encoding
                                                pdf.multi_cell(0, 10, line.encode('latin-1', errors='replace').decode('latin-1'))
                                            file_bytes = pdf.output(dest='S').encode('latin-1', errors='replace')
                                            mime = "application/pdf"
                                        else:
                                            file_bytes = export_content.encode("utf-8")
                                            mime = "text/plain"

                                        st.download_button(
                                            label=f"📥 Download {format_choice}",
                                            data=BytesIO(file_bytes),
                                            file_name=f"{file_base}.{format_choice.lower()}",
                                            mime=mime,
                                            key=f"download_{file_base}_{format_choice}"
                                        )

                        except Exception as e:
                            st.warning(f"⚠ Couldn't read attachments for {title}: {e}")

                if len(summaries) > 1:
                    st.markdown("## 📦 Download All Summaries")
                    all_export = ""
                    for title, summary, quotes in summaries:
                        all_export += f"\n\n---\n\n{title}\n\nTop Quotes:\n{quotes}\n\nSummary with Timestamps:\n{summary}\n"

                    file_name = st.text_input("Filename for all summaries:", value="All_Summaries")
                    all_format = st.selectbox("Choose format:", ["PDF", "TXT"])

                    if all_format == "PDF":
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        for line in all_export.split("\n"):
                            # ✅ Unicode-safe encoding
                            pdf.multi_cell(0, 10, line.encode('latin-1', errors='replace').decode('latin-1'))
                        file_data = pdf.output(dest="S").encode("latin-1", errors='replace')
                        mime_type = "application/pdf"
                        ext = "pdf"
                    else:
                        file_data = all_export.encode("utf-8")
                        mime_type = "text/plain"
                        ext = "txt"

                    st.download_button(
                        label=f"📥 Download All as {ext.upper()}",
                        data=BytesIO(file_data),
                        file_name=f"{file_name.strip() or 'All_Summaries'}.{ext}",
                        mime=mime_type,
                        key=f"download_all_{ext}"
                    )

        except Exception as e:
            st.error(f"❌ Error fetching pages: {str(e)}")
else:
    st.error("❌ Connection to Confluence failed.")
