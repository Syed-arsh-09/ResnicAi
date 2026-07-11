# ResnicAI

Multi-Agent Collaborative Research & Fact-Checked Reports powered by CrewAI.

## Setup

1. Copy `.env.template` to `.env` and fill in your API keys (Gemini, Serper).
2. Install requirements (e.g., `pip install -r requirements.txt`).
3. Run the application: `python -m streamlit run app.py`

## Architecture

- `src/` contains all the core CrewAI orchestration logic, split into tools, tasks, and agents.
- `app.py` is strictly the Streamlit frontend.
