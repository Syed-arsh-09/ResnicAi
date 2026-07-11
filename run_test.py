import os
from dotenv import load_dotenv
from crewai import LLM
from src.utils import load_agent_prompts
from src.tools import setup_tools
from src.crew import run_crew_job

def test_run():
    load_dotenv(override=True)
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    serper_api_key = os.getenv("SERPER_API_KEY", "")
    selected_model = "gemini-3.1-flash-lite"

    print(f"GEMINI_API_KEY found: {bool(gemini_api_key)}")
    print(f"SERPER_API_KEY found: {bool(serper_api_key)}")

    llm = LLM(
        model=f"gemini/{selected_model}",
        api_key=gemini_api_key,
        temperature=0.2
    )

    tools, tools_log = setup_tools(serper_api_key)
    agent_prompts = load_agent_prompts("Prompts.txt")
    research_topic = "Impact of ai"

    print("Starting crew job...")
    raw, final = run_crew_job(research_topic, agent_prompts, llm, tools)
    print("=== FINAL REPORT ===")
    print(final)

if __name__ == "__main__":
    test_run()
