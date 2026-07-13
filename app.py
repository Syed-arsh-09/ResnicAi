import os
import sys
import time
import threading
import re
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
    
    /* New Custom UI for Agent Workflow */
    .workflow-container { display: flex; flex-direction: column; gap: 16px; margin-top: 10px; }
    .topic-header { display: inline-flex; align-items: center; gap: 8px; background-color: rgba(255, 255, 255, 0.05); padding: 6px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: 500; color: #CBD5E1; margin-bottom: 8px; width: fit-content; border: 1px solid rgba(255, 255, 255, 0.1); }
    .agent-card { background-color: #27272A; border: 1px solid #3F3F46; border-radius: 12px; padding: 16px; color: #E2E8F0; }
    .agent-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .agent-icon-wrapper { width: 32px; height: 32px; border-radius: 8px; background-color: #F8FAFC; color: #0F172A; display: flex; justify-content: center; align-items: center; padding: 6px; }
    .agent-title { font-weight: 600; font-size: 1.1rem; flex-grow: 1; }
    .status-badge { font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-queued { background-color: rgba(255, 255, 255, 0.1); color: #94A3B8; }
    .badge-running { background-color: rgba(56, 189, 248, 0.2); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); animation: pulse 2s infinite; }
    .badge-done { background-color: rgba(34, 197, 94, 0.2); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .agent-steps { margin: 0; padding-left: 24px; font-size: 0.9rem; color: #94A3B8; line-height: 1.6; list-style-type: disc; }
    .agent-steps li { margin-bottom: 4px; }
    .agent-steps b { color: #CBD5E1; font-weight: 600; }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(56, 189, 248, 0); }
        100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }
    
    /* Citation styling */
    a[title^="Source"] {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #374151; 
        color: #9CA3AF !important; 
        border-radius: 50%;
        width: 1.15rem;
        height: 1.15rem;
        font-size: 0.7rem;
        font-weight: 700;
        text-decoration: none !important;
        margin: 0 4px;
        vertical-align: super;
        border: 1px solid #4B5563; 
        transition: all 0.2s ease;
    }
    a[title^="Source"]:hover {
        background-color: #4B5563;
        color: #F9FAFB !important; 
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
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
if "log_line_buffer" not in st.session_state:
    st.session_state.log_line_buffer = ""
    
# Structured UI states
def init_agent_states():
    return {
        "Senior Researcher": {"status": "queued", "steps": []},
        "Technical Copywriter & Synthesizer": {"status": "queued", "steps": []},
        "Quality Assurance & Fact-Checker": {"status": "queued", "steps": []}
    }

if "agent_states" not in st.session_state:
    st.session_state.agent_states = init_agent_states()
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "current_label" not in st.session_state:
    st.session_state.current_label = None

def generate_agent_html(states, topic):
    icons = {
        "Senior Researcher": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
        "Technical Copywriter & Synthesizer": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>',
        "Quality Assurance & Fact-Checker": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
    }
    
    html_parts = ['<div class="workflow-container">']
    if topic:
        html_parts.append(
            '<div class="topic-header">'
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="4"></circle></svg>'
            f' Topic: {topic}'
            '</div>'
        )
        
    for agent, state in states.items():
        status = state["status"]
        steps = state["steps"]
        icon = icons.get(agent, icons["Senior Researcher"])
        
        html_parts.append('<div class="agent-card">')
        html_parts.append('<div class="agent-header">')
        html_parts.append(f'<div class="agent-icon-wrapper">{icon}</div>')
        html_parts.append(f'<div class="agent-title">{agent}</div>')
        html_parts.append(f'<div class="status-badge badge-{status}">{status}</div>')
        html_parts.append('</div>')
        html_parts.append('<ul class="agent-steps">')
        
        if not steps and status == "queued":
            html_parts.append('<li>Waiting to start...</li>')
        else:
            for step in steps[-15:]:
                html_parts.append(f'<li>{step}</li>')
        
        html_parts.append('</ul>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    return "".join(html_parts)

def parse_log_line(line, state):
    updated = False
    
    for agent_name in state["agent_states"].keys():
        if agent_name in line and ("Working Agent" in line or "Agent:" in line):
            current = state["current_agent"]
            if current and current in state["agent_states"]:
                state["agent_states"][current]["status"] = "done"
                
            state["current_agent"] = agent_name
            state["agent_states"][agent_name]["status"] = "running"
            if "current_label" in state:
                state["current_label"] = None
            return True
        
    current = state["current_agent"]
    if current and current in state["agent_states"]:
        line_clean = line.strip()
        
        # Clean up CrewAI tracing prefixes (e.g. '| | ', '| ', etc)
        line_clean = re.sub(r'^(\s*\|\s*)+', '', line_clean).strip()
        line_clean = re.sub(r'^[│├─┌└┐┘]+\s*', '', line_clean).strip()
        
        # Filter out rich formatting decorative borders
        if not line_clean or set(line_clean).issubset(set("─╭╰│┌└┐┘ 🚀✅📋🤖🔧")):
            return False
            
        if "Agent Final Answer" in line_clean or "Final Answer:" in line_clean:
            state["agent_states"][current]["steps"].append("Compiling output for handoff...")
            state["current_label"] = "Status"
            return True
            
        if "Tool Execution Completed" in line_clean:
            state["agent_states"][current]["steps"].append("Analyzing gathered findings...")
            state["current_label"] = "Found"
            return True
            
        patterns = {
            r"^Task:\s*(.*)": "Thinking",
            r"^Args:\s*(.*)": "Input",
            r"^Tool:\s*(.*)": "Action",
            # Legacy fallbacks
            r"^Action:\s*(.*)": "Action",
            r"^Action Input:\s*(.*)": "Input",
            r"^Thought:\s*(.*)": "Thinking",
            r"^Observation:\s*(.*)": "Found"
        }
        
        matched = False
        for pattern, label in patterns.items():
            match = re.search(pattern, line_clean, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                
                # Prevent logging 'Tool:' twice (it appears in both Started and Completed boxes)
                if label == "Action" and state.get("current_label") == "Found":
                    break
                    
                if label == "Input":
                    content = re.sub(r'[{}\'"\[\]]', '', content)
                    content = re.sub(r'(?i)(search_query:|query:)\s*', '', content).strip()
                
                state["current_label"] = label
                
                if content and "do i need to use a tool" in content.lower():
                    pass # Ignore internal tool monologue
                else:
                    if len(content) > 100:
                        content = content[:97] + "..."
                        
                    if label == "Action":
                        step_text = f"Initiating action: {content}"
                    elif label == "Input":
                        step_text = f"Searching: {content}"
                    elif label == "Thinking":
                        content = content[0].upper() + content[1:] if content else ""
                        step_text = f"Reasoning: {content}"
                    else:
                        step_text = content
                        
                    state["agent_states"][current]["steps"].append(step_text)
                    updated = True
                matched = True
                break
                
        if not matched and state.get("current_label"):
            if "do i need to use a tool" not in line_clean.lower():
                if line_clean.startswith('* ') or line_clean.startswith('- ') or line_clean.startswith('#'):
                    # Strip the markdown character to match the clean UI design
                    clean_bullet = re.sub(r'^[*#\-]\s*', '', line_clean)
                    if len(clean_bullet) > 100:
                        clean_bullet = clean_bullet[:97] + "..."
                    state["agent_states"][current]["steps"].append(clean_bullet)
                    updated = True
                elif len(state["agent_states"][current]["steps"]) > 0:
                    # Append to the current step but keep it short
                    last_step = state["agent_states"][current]["steps"][-1]
                    if len(last_step) < 100 and not last_step.endswith("..."):
                        new_step = last_step + " " + line_clean
                        if len(new_step) > 100:
                            new_step = new_step[:97] + "..."
                        state["agent_states"][current]["steps"][-1] = new_step
                        updated = True
                
    return updated


def format_citations(text):
    if not text:
        return text
    links = []
    
    def replacer(match):
        is_markdown = match.group(2) is not None
        
        if is_markdown:
            text_part = match.group(1)
            url_part = match.group(2)
        else:
            text_part = ""
            url_part = match.group(3)
            
        if url_part not in links:
            links.append(url_part)
        idx = links.index(url_part) + 1
        
        citation = f"[{idx}]({url_part} \"Source: {url_part}\")"
        
        if not is_markdown:
            return citation
            
        tp_clean = text_part.lower().strip()
        # If the text was just a number, 'source', or the URL itself, replace it entirely with the bubble
        if tp_clean.startswith('http') or tp_clean.startswith('source') or tp_clean.isdigit() or tp_clean == f"[{idx}]":
            return citation
        else:
            return f"{text_part} {citation}"
            
    # Single pass regex for both Markdown links and bare URLs
    # Group 1: text part of markdown link
    # Group 2: URL part of markdown link
    # Group 3: bare URL (negative lookbehind ensures it's not immediately preceded by '(' or '"')
    pattern = r'\[(.*?)\]\((https?://[^\s\)]+)\)|(?<![("])(https?://[a-zA-Z0-9./?=_%&+-]+)'
    
    text = re.sub(pattern, replacer, text)
            
    return text


tab_report, tab_raw, tab_logs = st.tabs(["📄 Final Checked Report", "📊 Researcher Raw Data", "🤖 Agent Thinking Process"])

with tab_logs:
    st.markdown('<div class="console-header">💻 Architected Interactive Agent Workflow</div>', unsafe_allow_html=True)
    log_area = st.empty()
    if st.session_state.live_logs:
        log_area.markdown(generate_agent_html(st.session_state.agent_states, ""), unsafe_allow_html=True)
    else:
        log_area.markdown(generate_agent_html(init_agent_states(), ""), unsafe_allow_html=True)

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
    st.session_state.log_line_buffer = ""
    st.session_state.agent_states = init_agent_states()
    st.session_state.current_agent = None
    
    log_area.markdown(generate_agent_html(st.session_state.agent_states, research_topic), unsafe_allow_html=True)
    
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
            with open("crewai_raw_trace.log", "a", encoding="utf-8") as f:
                f.write(clean_text)

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
                st.session_state.log_line_buffer += new_logs
                
                lines = st.session_state.log_line_buffer.split('\n')
                st.session_state.log_line_buffer = lines[-1]
                
                updated_ui = False
                for line in lines[:-1]:
                    if parse_log_line(line, st.session_state):
                        updated_ui = True
                        
                if updated_ui:
                    log_area.markdown(generate_agent_html(st.session_state.agent_states, research_topic), unsafe_allow_html=True)
                
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
        st.session_state.log_line_buffer += new_logs
        lines = st.session_state.log_line_buffer.split('\n')
        st.session_state.log_line_buffer = ""
        for line in lines:
            parse_log_line(line, st.session_state)
            
    # Mark last agent as done
    if st.session_state.current_agent and st.session_state.current_agent in st.session_state.agent_states:
        st.session_state.agent_states[st.session_state.current_agent]["status"] = "done"
        
    log_area.markdown(generate_agent_html(st.session_state.agent_states, research_topic), unsafe_allow_html=True)

    if result_container.get("error"):
        st.error(f"Workflow failed: {result_container['error']}")
    else:
        st.session_state.raw_research = result_container["raw"]
        st.session_state.final_report = format_citations(result_container["final"])
        st.success("ResnicAI workflow completed successfully!")
    
    st.rerun()
