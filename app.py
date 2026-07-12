import os
import sys
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import queue

# Import refactored logic
from src.utils import ThreadSafeStreamRedirector, set_log_callback, clear_log_callback, clean_ansi, load_agent_prompts
from src.tools import setup_tools
from src.crew import run_crew_job

# Try importing CrewAI LLM
try:
    from crewai import LLM
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_ERROR = e

# Load environment variables explicitly from the .env file in the same directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Force enable CrewAI tracing to provide full transparency
os.environ["CREWAI_TRACING_ENABLED"] = "true"

# Install thread-safe redirector once
if 'redirector_installed' not in st.session_state:
    if not isinstance(sys.stdout, ThreadSafeStreamRedirector):
        sys.stdout = ThreadSafeStreamRedirector(sys.stdout)
    if not isinstance(sys.stderr, ThreadSafeStreamRedirector):
        sys.stderr = ThreadSafeStreamRedirector(sys.stderr)
    st.session_state.redirector_installed = True

# Setup layout & UI elements
st.set_page_config(
    page_title="ResnicAI - Agentic Research Crew",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .app-title {
        background: linear-gradient(135deg, #FF4B4B 0%, #7E22CE 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05rem;
    }
    .app-subtitle { font-size: 1.15rem; color: #94A3B8; margin-bottom: 2rem; }
    .sidebar-header { font-weight: 700; font-size: 1.4rem; color: #F8FAFC; margin-bottom: 1rem; }
    .console-header { font-weight: 600; font-size: 1rem; color: #38BDF8; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 4px 4px 0px 0px; padding: 10px 16px; font-weight: 600; color: #94A3B8; transition: all 0.2s; }
    .stTabs [data-baseweb="tab"]:hover { color: #F8FAFC; background-color: rgba(255, 255, 255, 0.03); }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #38BDF8; border-bottom-color: #38BDF8; }
</style>
""", unsafe_allow_html=True)

# App Headers
st.markdown('<div class="app-title">🧬 ResnicAI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Multi-Agent Collaborative Research & Fact-Checked Reports powered by CrewAI</div>', unsafe_allow_html=True)

if IMPORT_ERROR:
    st.error(f"Failed to load required libraries. Please check your setup. Error: {IMPORT_ERROR}")
    st.stop()

# Load agent configuration
agent_prompts = load_agent_prompts("Prompts.txt")
if not agent_prompts:
    st.stop()

# Load API keys from environment
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
serper_api_key = os.getenv("SERPER_API_KEY", "")
selected_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Sidebar Setup
with st.sidebar:
    st.markdown('<div class="sidebar-header">🧬 ResnicAI Panel</div>', unsafe_allow_html=True)
    st.markdown("This dashboard monitors the active **CrewAI** agents configured via `Prompts.txt`.")
    st.markdown("---")
    st.markdown("### 📋 Agent Status Map")
    for k, agent_info in agent_prompts.items():
        st.markdown(f"**{agent_info['role']}** is Active")

# Main Workspace Page
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 🔍 Start Your Research")
    research_topic = st.text_input(
        "What topic would you like the agents to research?",
        placeholder="e.g., Impact of Quantum Computing on Cybersecurity in 2026",
        help="Enter a clear research query."
    )
    start_btn = st.button("🚀 Trigger CrewAI Agent Workflow", use_container_width=True)

# Setup Session State for outputs
if "raw_research" not in st.session_state:
    st.session_state.raw_research = ""
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "live_logs" not in st.session_state:
    st.session_state.live_logs = ""

tab_report, tab_raw, tab_logs = st.tabs(["📄 Final Checked Report", "📊 Researcher Raw Data", "🤖 Agent Thinking Process"])

with tab_logs:
    st.markdown('<div class="console-header">💻 Agent Console Logger</div>', unsafe_allow_html=True)
    log_area = st.empty()
    log_area.code(st.session_state.live_logs or "Waiting for workflow to start...")

with tab_raw:
    if st.session_state.raw_research:
        st.markdown(st.session_state.raw_research)
        st.download_button(
            label="Download Raw Data",
            data=st.session_state.raw_research,
            file_name=f"raw_research_{int(time.time())}.txt",
            mime="text/plain"
        )
    else:
        st.info("Trigger the workflow to gather raw research data.")

with tab_report:
    if st.session_state.final_report:
        def stream_data(text):
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.02)
        
        st.write_stream(stream_data(st.session_state.final_report))
            
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        filename = f"report_{int(time.time())}.md"
        filepath = reports_dir / filename
        try:
            filepath.write_text(st.session_state.final_report, encoding="utf-8")
        except Exception as e:
            pass
        st.download_button(
            label="Download Final Report (Markdown)",
            data=st.session_state.final_report,
            file_name=filename,
            mime="text/markdown"
        )
    else:
        st.info("Trigger the workflow to view the synthesized report.")

# Execution logic
if start_btn:
    if not gemini_api_key:
        st.error("`GEMINI_API_KEY` is missing in the environment. Please configure it in your `.env` file.")
        st.stop()
    if not research_topic.strip():
        st.error("Please enter a research topic.")
        st.stop()
        
    st.session_state.live_logs = "Initializing CrewAI Agents...\n"
    log_area.code(st.session_state.live_logs)
    
    try:
        llm = LLM(
            model=f"gemini/{selected_model}",
            api_key=gemini_api_key,
            temperature=0.2
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini LLM: {e}")
        st.stop()
        
    tools, tools_log = setup_tools(serper_api_key)
    st.session_state.live_logs += tools_log
    log_area.code(st.session_state.live_logs)

    result_container = {"raw": "", "final": "", "error": None}

    log_queue = queue.Queue()

    def run_crew_wrapper():
        try:
            raw, final = run_crew_job(research_topic, agent_prompts, llm, tools)
            result_container["raw"] = raw
            result_container["final"] = final
        except Exception as e:
            result_container["error"] = str(e)

    def append_log(text):
        clean_text = clean_ansi(text)
        if clean_text:
            log_queue.put(clean_text)

    crew_thread = threading.Thread(target=run_crew_wrapper)
    set_log_callback(append_log)
    
    with st.spinner("CrewAI workflow executing... Please watch the 'Agent Thinking Process' tab for live details."):
        crew_thread.start()
        while crew_thread.is_alive():
            new_logs = ""
            while not log_queue.empty():
                try:
                    new_logs += log_queue.get_nowait()
                except queue.Empty:
                    break
            
            if new_logs:
                st.session_state.live_logs += new_logs
                log_area.code(st.session_state.live_logs)
                
            time.sleep(0.5)
            
    clear_log_callback()
    
    # Drain any remaining logs
    new_logs = ""
    while not log_queue.empty():
        try:
            new_logs += log_queue.get_nowait()
        except queue.Empty:
            break
    if new_logs:
        st.session_state.live_logs += new_logs
        log_area.code(st.session_state.live_logs)

    if result_container.get("error"):
        st.error(f"Workflow failed: {result_container['error']}")
    else:
        st.session_state.raw_research = result_container["raw"]
        st.session_state.final_report = result_container["final"]
        st.success("ResnicAI workflow completed successfully!")
    
    st.rerun()
