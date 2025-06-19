import os
import io
import csv
import json
import streamlit as st
from fpdf import FPDF
from docx import Document
from atlassian import Confluence
import google.generativeai as genai
from bs4 import BeautifulSoup

# Initialize Confluence connection
@st.cache_resource
def init_confluence():
    try:
        return Confluence(
            url=os.environ["CONFLUENCE_BASE_URL"],
            username=os.environ["CONFLUENCE_USER_EMAIL"],
            password=os.environ["CONFLUENCE_API_KEY"],
            timeout=10
        )
    except Exception as e:
        st.error(f"Confluence initialization failed: {str(e)}")
        return None

# Initialize Gemini AI
def init_ai():
    genai.configure(api_key=os.environ["GENAI_API_KEY"])
    return genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")

# Clean HTML to plain text
def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n")

# Export formats
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return io.BytesIO(pdf.output(dest='S').encode('latin1'))

def create_docx(text):
    doc = Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def create_csv(text):
    output = io.StringIO()
    writer = csv.writer(output)
    for line in text.strip().split('\n'):
        writer.writerow([line])
    return io.BytesIO(output.getvalue().encode())

def create_json(text):
    return io.BytesIO(json.dumps({"response": text}, indent=4).encode())

def create_html(text):
    html = f"<html><body><pre>{text}</pre></body></html>"
    return io.BytesIO(html.encode())

def create_txt(text):
    return io.BytesIO(text.encode())

# UI Starts
st.set_page_config(page_title="Confluence AI Search", page_icon="🔗")
st.title("🔗 Confluence AI Powered Search")

confluence = init_confluence()
ai_model = init_ai()

selected_pages = []
full_context = ""

if confluence:
    st.success("✅ Connected to Confluence!")

    space_key = st.text_input("Enter your space key:")

    if space_key:
        try:
            pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
            all_titles = [p["title"] for p in pages]

            select_all = st.checkbox("Select All Pages")
            selected_titles = st.multiselect("Select Page(s):", all_titles, default=all_titles if select_all else [])
            show_content = st.checkbox("Show Page Content")

            selected_pages = [p for p in pages if p["title"] in selected_titles]

            if selected_pages:
                st.success(f"✅ Loaded {len(selected_pages)} page(s).")
                for page in selected_pages:
                    page_id = page["id"]
                    page_data = confluence.get_page_by_id(page_id, expand="body.storage")
                    raw_html = page_data["body"]["storage"]["value"]
                    text_content = clean_html(raw_html)
                    full_context += f"\n\nTitle: {page['title']}\n{text_content}"

                    if show_content:
                        with st.expander(f"📄 {page['title']}"):
                            st.markdown(raw_html, unsafe_allow_html=True)
            else:
                st.warning("Please select at least one page.")
        except Exception as e:
            st.error(f"Error fetching pages: {str(e)}")
else:
    st.error("❌ Connection to Confluence failed.")

# AI Query
if confluence and selected_pages:
    st.subheader("🤖 Generate AI Response")
    query = st.text_input("Enter your question:")

    if st.button("Generate Answer"):
        if query and full_context:
            try:
                prompt = (
                    f"Answer the following question using the provided Confluence page content as context.\n"
                    f"Context:\n{full_context}\n\n"
                    f"Question: {query}\n"
                    f"Instructions: Begin with the answer based on the context above. Then, if applicable, supplement with general knowledge."
                )
                response = ai_model.generate_content(prompt)
                st.session_state.ai_response = response.text.strip()
            except Exception as e:
                st.error(f"AI generation failed: {str(e)}")
        else:
            st.error("Please enter a query.")

# Response + Export
if "ai_response" in st.session_state:
    st.markdown("### 💬 AI Response")
    st.markdown(st.session_state.ai_response)

    file_name = st.text_input("Enter file name (without extension):", value="ai_response")
    export_format = st.selectbox("Choose file format to export:", ["TXT", "PDF", "Markdown", "HTML", "DOCX", "CSV", "JSON"])

    export_map = {
        "TXT": (create_txt, "text/plain", ".txt"),
        "PDF": (create_pdf, "application/pdf", ".pdf"),
        "Markdown": (create_txt, "text/markdown", ".md"),
        "HTML": (create_html, "text/html", ".html"),
        "DOCX": (create_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        "CSV": (create_csv, "text/csv", ".csv"),
        "JSON": (create_json, "application/json", ".json")
    }

    if file_name:
        creator_func, mime, ext = export_map[export_format]
        buffer = creator_func(st.session_state.ai_response)
        st.download_button(
            label="📥 Download File",
            data=buffer,
            file_name=f"{file_name.strip() or 'ai_response'}{ext}",
            mime=mime
        )

    st.markdown("---")
    st.subheader("📝 Save to Confluence Page")

    target_page_title = st.text_input("Enter the Confluence page title to save this to:")
    if st.button("✏ Save AI Response to Confluence"):
        if target_page_title:
            try:
                matching_pages = [p for p in selected_pages if p["title"] == target_page_title]
                if not matching_pages:
                    st.error("Page not found in selected pages.")
                else:
                    page_id = matching_pages[0]["id"]
                    existing_page = confluence.get_page_by_id(page_id, expand="body.storage")
                    existing_content = existing_page["body"]["storage"]["value"]

                    updated_body = f"{existing_content}<hr/><h3>AI Response</h3><p>{st.session_state.ai_response.replace('\n', '<br>')}</p>"

                    confluence.update_page(
                        page_id=page_id,
                        title=target_page_title,
                        body=updated_body,
                        representation="storage"
                    )
                    st.success("✅ AI response saved to Confluence page.")
            except Exception as e:
                st.error(f"❌ Failed to update page: {str(e)}")
        else:
            st.warning("Please enter a page title.")
