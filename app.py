import streamlit as st
import time
import re
from rag_pipeline import NewsRAGPipeline

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="News RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# URL Validator (UI-side only)
# ---------------------------------
def is_valid_article_url(url: str) -> bool:
    section_patterns = [
        r'/incoming/$',
        r'/section/',
        r'/category/',
        r'/tag/',
        r'/search/',
        r'\?.*page=',
        r'/$'
    ]

    article_patterns = [
        r'/article\d+\.ece$',
        r'/story/',
        r'-\d+$',
        r'\.html$',
        r'\.php\?id='
    ]

    url = url.lower()

    for p in section_patterns:
        if re.search(p, url):
            return False

    for p in article_patterns:
        if re.search(p, url):
            return True

    return len(url) > 40


# ---------------------------------
# Session State
# ---------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = NewsRAGPipeline()

if "article_result" not in st.session_state:
    st.session_state.article_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, message)


# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("News RAG System")
st.sidebar.markdown("""
### Capabilities
- Paste news article URL  
- Auto-clean content  
- Chunking + Embeddings  
- Ask questions  
- Article summary  
""")

if st.sidebar.button("New Article"):
    st.session_state.article_result = None
    st.session_state.chat_history = []
    st.rerun()


# ---------------------------------
# Main Title
# ---------------------------------
st.title("News Article Question Answering")

# ---------------------------------
# URL Input Section
# ---------------------------------
if st.session_state.article_result is None:
    st.subheader("Enter News Article URL")

    url = st.text_input(
        "Paste FULL article URL",
        placeholder="https://www.thehindu.com/.../article123.ece"
    )

    if url:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if not is_valid_article_url(url):
            st.warning("This looks like a section page. Try a full article URL.")

        if st.button("Process Article"):
            with st.spinner("Processing article..."):
                start = time.time()

                result = st.session_state.pipeline.process_news_url(
                    url, use_playwright=False
                )

                if "error" in result:
                    st.info("Standard fetch failed. Retrying with browser...")
                    result = st.session_state.pipeline.process_news_url(
                        url, use_playwright=True
                    )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.article_result = result
                    st.success(f"Processed in {time.time() - start:.1f}s")
                    st.rerun()

# ---------------------------------
# Article Loaded View
# ---------------------------------
else:
    article = st.session_state.article_result["article"]

    st.subheader("Article Loaded")
    st.markdown(f"### {article['title']}")
    st.caption(
        f"{len(article['text'])} characters | "
        f"{st.session_state.article_result['chunks_count']} chunks"
    )

    with st.expander("View Article Text"):
        st.write(article["text"])

    # ---------------------------------
    # Summary Button
    # ---------------------------------
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            response = st.session_state.pipeline.ask_question(
                "Provide a concise 3-point summary of this article."
            )
            with st.chat_message("assistant"):
                st.markdown("### Summary")
                st.markdown(response["answer"])

    # ---------------------------------
    # Chat UI (ChatGPT Style)
    # ---------------------------------
    st.divider()
    st.subheader("Chat with Article")

    # Show previous messages
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input
    user_input = st.chat_input("Ask a question about this article...")

    if user_input:
        # User message
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.pipeline.ask_question(user_input)
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))