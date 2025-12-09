import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore import VectorStore
from src.agents import ReportAgent, HomeworkAgent, TutorAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Homework Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "vector_store": None,
        "vs_session_id": None,
        "report_agent": None,
        "homework_agent": None,
        "tutor_agent": None,
        "messages": [],
        "current_agent": "homework",
        "uploaded_files": [],
        "hw_started": False,
        "report_started": False,
        "tutor_started": False,
        "report_topic": "",
        "report_draft": "",
        "report_outline": "",
        "report_research": "",
        "report_citations": [],
        "report_all_citations": [],
        "report_outline_ready": False,
        "report_draft_ready": False,
        "report_research_ready": False,
        "report_research_approved": False,
        "report_outline_unlocked": False,
        "report_outline_approved": False,
        # Homework solver state
        "hw_solutions": [],
        "hw_doc_text": "",
        "hw_processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def _merge_citations(existing, new):
    seen = set(existing or [])
    merged = list(existing or [])
    for c in new or []:
        if c not in seen:
            seen.add(c)
            merged.append(c)
    return merged


def init_vector_store():
    """Initialize vector store and session ID (call after init_session_state)."""
    import uuid
    # Generate unique session ID for this browser session
    if st.session_state.vs_session_id is None:
        st.session_state.vs_session_id = str(uuid.uuid4())[:8]
    
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore(
            collection_name="homework_helper",
            persist_directory="./chroma_data"
        )


def initialize_agents():
    """Initialize agents."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        if st.session_state.report_agent is None:
            st.session_state.report_agent = ReportAgent(
                vector_store=st.session_state.vector_store,
                model="gpt-5.1",
                enable_web_search=bool(os.getenv("TAVILY_API_KEY")),
                session_id=st.session_state.vs_session_id,
            )
        
        if st.session_state.homework_agent is None:
            st.session_state.homework_agent = HomeworkAgent(
                vector_store=st.session_state.vector_store,
                model="gpt-5.1"
            )
        
        if st.session_state.tutor_agent is None:
            st.session_state.tutor_agent = TutorAgent(
                vector_store=st.session_state.vector_store,
                model="gpt-5.1",
                session_id=st.session_state.vs_session_id,
            )
        return True
    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return False


def process_message(user_input: str, image_bytes=None, show_spinner=False):
    """Process a user message and get agent response."""
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "image": image_bytes
    })
    
    try:
        if st.session_state.current_agent == "report":
            response = st.session_state.report_agent.chat(
                user_message=user_input,
                conversation_history=st.session_state.messages[:-1],
                topic=st.session_state.report_topic
            )
        elif st.session_state.current_agent == "tutor":
            response = st.session_state.tutor_agent.chat(
                user_message=user_input,
                conversation_history=st.session_state.messages[:-1],
            )
        else:
            response = st.session_state.homework_agent.chat(
                user_message=user_input,
                conversation_history=st.session_state.messages[:-1],
                image=image_bytes
            )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"❌ Error: {str(e)}"
        })


def render_sidebar():
    """Render compact sidebar."""
    with st.sidebar:
        st.markdown("### 📚 Homework Helper")
        
        st.markdown("##### Mode")
        
        # Agent selection - vertical layout
        mode_map = {"homework": 0, "report": 1, "tutor": 2}
        agent = st.radio(
            "Select mode",
            options=["homework", "report", "tutor"],
            format_func=lambda x: {"homework": "🧮 Solver", "report": "📝 Writer", "tutor": "📖 Tutor"}[x],
            index=mode_map.get(st.session_state.current_agent, 0),
            label_visibility="collapsed"
        )
        
        if agent != st.session_state.current_agent:
            st.session_state.current_agent = agent
            st.session_state.messages = []
            st.session_state.hw_started = False
            st.session_state.report_started = False
            st.session_state.tutor_started = False
            st.session_state.hw_solutions = []
            st.rerun()
        
        # Show loaded docs count (read-only info)
        if st.session_state.uploaded_files:
            st.caption(f"📚 {len(st.session_state.uploaded_files)} doc(s) loaded")
            if st.button("🗑️ Clear docs", use_container_width=True):
                st.session_state.vector_store.clear()
                st.session_state.uploaded_files = []
                st.rerun()
            st.divider()
        
        # Reset button
        if st.session_state.messages or st.session_state.hw_started or st.session_state.report_started or st.session_state.tutor_started:
            if st.button("🔄 Start over", use_container_width=True):
                st.session_state.messages = []
                st.session_state.hw_started = False
                st.session_state.report_started = False
                st.session_state.tutor_started = False
                st.session_state.hw_solutions = []
                st.session_state.report_topic = ""
                st.rerun()


def render_homework_start():
    """Render homework solver start page."""
    st.markdown("## 🧮 Homework Solver")
    st.markdown("Upload your homework PDF to extract and solve all questions with Wolfram Alpha.")
    
    # Check for Wolfram Alpha
    if os.getenv("WOLFRAM_ALPHA_APPID"):
        st.caption("✅ Wolfram Alpha enabled")
    else:
        st.caption("💡 Set WOLFRAM_ALPHA_APPID for enhanced math solving")
    
    
    uploaded_doc = st.file_uploader(
        "Upload homework PDF",
        type=["pdf"],
        key="hw_doc_upload"
    )
    
    if uploaded_doc:
        st.success(f"📄 {uploaded_doc.name} ready")
        
        if st.button("🚀 Extract & Solve All Questions", type="primary", use_container_width=True):
            # Save and extract text from PDF
            temp_path = f"./data/{uploaded_doc.name}"
            os.makedirs("./data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_doc.getvalue())
            
            with st.spinner("📖 Extracting text from PDF..."):
                from langchain_community.document_loaders import PyMuPDFLoader
                loader = PyMuPDFLoader(temp_path)
                docs = loader.load()
                doc_text = "\n\n".join([d.page_content for d in docs])
            
            st.session_state.hw_doc_text = doc_text
            
            with st.spinner("🔍 Extracting questions..."):
                questions = st.session_state.homework_agent.extract_questions_from_text(doc_text)
            
            if questions:
                st.info(f"Found {len(questions)} questions. Solving each...")
                
                solutions = []
                progress = st.progress(0)
                status = st.empty()
                
                for i, q in enumerate(questions):
                    status.text(f"Solving question {q['number']} ({i+1}/{len(questions)})...")
                    solution = st.session_state.homework_agent.solve_question_with_wolfram(q['question'])
                    solution['number'] = q['number']
                    solutions.append(solution)
                    progress.progress((i + 1) / len(questions))
                
                st.session_state.hw_solutions = solutions
                st.session_state.hw_started = True
                progress.empty()
                status.empty()
                st.rerun()
            else:
                st.error("Could not extract questions from the PDF.")


def render_tutor_start():
    """Render tutor start page - upload docs then chat."""
    st.markdown("## 📖 Study Tutor")
    st.markdown("Upload your textbooks, notes, or study materials and ask questions about them.")
    
    
    # Show current docs
    if st.session_state.uploaded_files:
        st.success(f"📚 {len(st.session_state.uploaded_files)} document(s) loaded")
        with st.expander("Loaded documents"):
            for f in st.session_state.uploaded_files:
                st.markdown(f"- {f}")
    
    # Upload docs
    st.markdown("##### 📁 Upload Study Materials")
    uploaded = st.file_uploader(
        "Upload PDFs (textbooks, notes, slides)",
        type=["pdf"],
        accept_multiple_files=True,
        key="tutor_docs"
    )
    
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.uploaded_files:
                temp_path = f"./data/{f.name}"
                os.makedirs("./data", exist_ok=True)
                with open(temp_path, "wb") as file:
                    file.write(f.getvalue())
                try:
                    chunks = st.session_state.vector_store.add_pdf(
                        temp_path,
                        session_id=st.session_state.vs_session_id
                    )
                    st.session_state.uploaded_files.append(f.name)
                    st.success(f"✅ Added {f.name} ({chunks} chunks)")
                except Exception as e:
                    st.error(f"Failed to index {f.name}: {e}")
    
    
    # Start chat button
    if st.session_state.uploaded_files:
        if st.button("💬 Start Studying", type="primary", use_container_width=True):
            # Generate a welcome message with document summary
            with st.spinner("📖 Preparing your study session..."):
                summary = st.session_state.tutor_agent.summarize_document()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Welcome! Here's an overview of your materials:**\n\n{summary}\n\n---\n\nWhat would you like to learn about? Feel free to ask any questions!"
                })
            st.session_state.tutor_started = True
            st.rerun()


def render_report_start():
    """Render report writer start page."""
    st.markdown("## 📝 Report Writer")
    st.markdown("Enter your topic and I'll research, outline, and draft the paper automatically. You can upload sources to ground the draft. After the draft, provide feedback and I'll revise.")
    
    
    with st.form("report_start_form"):
        topic = st.text_input(
            "What's your paper about?",
            placeholder="e.g., The impact of artificial intelligence on healthcare"
        )
        
        st.markdown("##### 📎 Upload sources (optional, PDFs)")
        sources = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="report_sources"
        )
        
        submitted = st.form_submit_button("🚀 Start Paper Generation", use_container_width=True)
    
    if sources:
        st.caption(f"📄 {len(sources)} source(s) ready")
        for f in sources:
            if f.name not in st.session_state.uploaded_files:
                temp_path = f"./data/{f.name}"
                os.makedirs("./data", exist_ok=True)
                with open(temp_path, "wb") as file:
                    file.write(f.getvalue())
                try:
                    st.session_state.vector_store.add_pdf(
                        temp_path,
                        session_id=st.session_state.vs_session_id
                    )
                    st.session_state.uploaded_files.append(f.name)
                except:
                    pass
    
    if submitted:
        if not topic.strip():
            st.error("Please enter a topic to start.")
        else:
            st.session_state.report_topic = topic.strip()
            # Reset state for new run
            st.session_state.report_research_ready = False
            st.session_state.report_outline_ready = False
            st.session_state.report_draft_ready = False
            st.session_state.report_research_approved = False
            st.session_state.report_outline_unlocked = False
            st.session_state.report_outline_approved = False
            st.session_state.report_draft = ""
            st.session_state.report_outline = ""
            st.session_state.report_research = ""
            st.session_state.report_citations = []
            st.session_state.report_all_citations = []
            with st.spinner("📝 Researching..."):
                research_data = st.session_state.report_agent.generate_research_only(
                    topic=st.session_state.report_topic,
                    messages=st.session_state.messages,
                    has_documents=(st.session_state.vector_store.count > 0) if st.session_state.vector_store else False,
                )
            st.info("✅ Research generated. Please review and continue to outline.", icon="🔎")
            # Save research stage
            st.session_state.report_research = research_data.get("research_notes", "")
            st.session_state.report_citations = research_data.get("citations", [])
            st.session_state.report_all_citations = _merge_citations(
                st.session_state.report_all_citations, st.session_state.report_citations
            )
            st.session_state.report_started = True
            st.session_state.report_research_ready = True
            st.session_state.report_outline_ready = False
            st.session_state.report_draft_ready = False
            # Do not auto-append to chat; show in structured view
            st.rerun()


def render_homework_solutions():
    """Render homework solutions view."""
    st.markdown("## 🧮 Homework Solutions")
    
    solutions = st.session_state.hw_solutions
    if not solutions:
        st.info("No solutions yet.")
        return
    
    st.success(f"✅ Solved {len(solutions)} questions")
    
    for sol in solutions:
        with st.expander(f"**Question {sol.get('number', '?')}**", expanded=True):
            st.markdown("**Problem:**")
            st.markdown(sol.get('question', 'N/A'))
            
            if sol.get('wolfram_result'):
                st.markdown("**Wolfram Alpha:**")
                st.code(sol['wolfram_result'], language=None)
            
            st.markdown("**Solution:**")
            st.markdown(sol.get('solution', 'No solution available'))
    
    st.markdown("### 💬 Follow-up Questions")
    st.caption("Ask about any of the solutions above")


def render_chat():
    """Render the chat interface."""
    # Show topic if report mode
    if st.session_state.current_agent == "report" and st.session_state.report_topic:
        st.caption(f"📝 Topic: {st.session_state.report_topic}")
    
    # Homework solutions display
    if st.session_state.current_agent == "homework" and st.session_state.hw_solutions:
        render_homework_solutions()
    
    # Tutor mode header
    if st.session_state.current_agent == "tutor":
        st.caption(f"📖 Tutor Mode | {len(st.session_state.uploaded_files)} doc(s) loaded")
    
    # Report flow: staged approval
    if st.session_state.current_agent == "report":
        # Stage 1: Research review -> unlock outline
        if st.session_state.report_research_ready and not st.session_state.report_research_approved:
            st.markdown("### 🔎 Research Notes (review & continue)")
            st.markdown(st.session_state.report_research or "_No research notes available_")
            if st.session_state.report_citations:
                with st.expander("📚 Citations"):
                    for c in st.session_state.report_all_citations or st.session_state.report_citations:
                        st.markdown(f"- {c}")
            if st.button("➡️ Continue to Outline", use_container_width=True):
                st.session_state.report_research_approved = True
                st.session_state.report_outline_unlocked = True
                # Generate outline now (separate stage)
                with st.spinner("📋 Generating outline..."):
                    outline_data = st.session_state.report_agent.generate_outline_only(
                        topic=st.session_state.report_topic,
                        research_notes=st.session_state.report_research,
                    )
                st.session_state.report_outline = outline_data.get("outline", "")
                # Merge citations (keep existing research citations too)
                new_citations = outline_data.get("citations", [])
                st.session_state.report_citations = (st.session_state.report_citations or []) + new_citations
                st.session_state.report_all_citations = _merge_citations(
                    st.session_state.report_all_citations, new_citations
                )
                st.session_state.report_outline_ready = True
                st.rerun()
        
        # Stage 2: Outline review -> approve / revise -> draft
        if st.session_state.report_outline_ready and st.session_state.report_outline_unlocked and not st.session_state.report_draft_ready:
            st.markdown("### 📋 Outline (review & approve)")
            st.markdown(st.session_state.report_outline or "_No outline available_")
            with st.expander("🔎 Research Notes"):
                st.markdown(st.session_state.report_research or "_No research notes available_")
            if st.session_state.report_citations:
                with st.expander("📚 Citations"):
                    for c in st.session_state.report_all_citations or st.session_state.report_citations:
                        st.markdown(f"- {c}")
            
            feedback = st.text_area("Outline feedback (optional)", key="outline_feedback", placeholder="e.g., add a section on ethics, deepen literature review in section 2")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🧭 Regenerate Outline", use_container_width=True):
                    with st.spinner("📋 Regenerating outline..."):
                        outline_data = st.session_state.report_agent.generate_outline_only(
                            topic=st.session_state.report_topic,
                            research_notes=st.session_state.report_research,
                        )
                    st.session_state.report_outline = outline_data.get("outline", "")
                    st.session_state.report_citations = outline_data.get("citations", st.session_state.report_citations)
                    st.rerun()
            with col2:
                if st.button("✅ Outline looks good", use_container_width=True):
                    with st.spinner("📝 Generating draft from approved outline..."):
                        draft = st.session_state.report_agent.generate_draft_from_outline(
                            topic=st.session_state.report_topic,
                            outline=st.session_state.report_outline,
                            research_notes=st.session_state.report_research,
                            messages=st.session_state.messages,
                        )
                    st.session_state.report_draft = draft
                    st.session_state.report_draft_ready = True
                    st.session_state.report_outline_approved = True
                    # Extract draft citations and merge
                    try:
                        draft_cites = st.session_state.report_agent._extract_citations(draft)
                        st.session_state.report_all_citations = _merge_citations(
                            st.session_state.report_all_citations, draft_cites
                        )
                    except Exception:
                        pass
                    st.rerun()
            with col3:
                if st.button("🔄 Revise outline", use_container_width=True, disabled=not feedback.strip()):
                    with st.spinner("📋 Revising outline..."):
                        new_outline = st.session_state.report_agent.regenerate_outline_with_feedback(
                            topic=st.session_state.report_topic,
                            research_notes=st.session_state.report_research,
                            feedback=feedback,
                        )
                    st.session_state.report_outline = new_outline
                    st.session_state.messages.append({"role": "assistant", "content": "**Revised Outline:**\n\n" + new_outline})
                    st.rerun()
        
        # Stage 3: Draft review & revision
        if st.session_state.report_draft_ready:
            st.markdown("### 📄 Draft")
            st.markdown(st.session_state.report_draft)
            with st.expander("📋 Outline"):
                st.markdown(st.session_state.report_outline or "_No outline available_")
            with st.expander("🔎 Research Notes"):
                st.markdown(st.session_state.report_research or "_No research notes available_")
            if st.session_state.report_citations:
                with st.expander("📚 Citations"):
                    for c in st.session_state.report_all_citations or st.session_state.report_citations:
                        st.markdown(f"- {c}")
            
            st.markdown("#### ✍️ Feedback & Revision")
            feedback = st.text_area("Provide feedback to revise the draft", key="report_feedback", placeholder="e.g., strengthen section 2 with more citations, shorten intro, adjust tone to formal academic")
            if st.button("🔄 Revise Draft", use_container_width=True, disabled=not feedback.strip()):
                with st.spinner("✏️ Revising draft..."):
                    revised = st.session_state.report_agent.revise_report(
                        current_draft=st.session_state.report_draft,
                        feedback=feedback,
                        topic=st.session_state.report_topic
                    )
                st.session_state.report_draft = revised
                try:
                    draft_cites = st.session_state.report_agent._extract_citations(revised)
                    st.session_state.report_all_citations = _merge_citations(
                        st.session_state.report_all_citations, draft_cites
                    )
                except Exception:
                    pass
                st.session_state.messages.append({"role": "assistant", "content": "**Revised Draft:**\n\n" + revised})
                st.rerun()
    
    # Display messages (chat history)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image"):
                st.image(msg["image"], width=300)
            st.markdown(msg["content"])
    
    # Chat input for follow-ups (both agents)
    placeholder = "Ask a follow-up question..."
    if st.session_state.current_agent == "homework":
        placeholder = "Ask for clarification or another problem..."
    
    if prompt := st.chat_input(placeholder):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                process_message(prompt)
            st.markdown(st.session_state.messages[-1]["content"])
        
        st.rerun()


def main():
    """Main application."""
    init_session_state()
    init_vector_store()
    render_sidebar()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key not found")
        st.code("set OPENAI_API_KEY=sk-your-key-here", language="bash")
        st.stop()
    
    if not initialize_agents():
        st.error("Failed to initialize agents")
        st.stop()
    
    # Route to appropriate view
    if st.session_state.current_agent == "homework":
        if st.session_state.hw_started:
            render_chat()
        else:
            render_homework_start()
    elif st.session_state.current_agent == "tutor":
        if st.session_state.tutor_started:
            render_chat()
        else:
            render_tutor_start()
    else:
        # Report flow: show chat view as soon as we have started the process
        if st.session_state.report_started:
            render_chat()
        else:
            render_report_start()


if __name__ == "__main__":
    main()
