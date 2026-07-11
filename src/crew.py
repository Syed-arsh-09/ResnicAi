from crewai import Crew, Process
from src.agents import create_agents
from src.tasks import create_tasks

def run_crew_job(research_topic, agent_prompts, llm, tools):
    """
    Instantiates the agents and tasks, forms the Crew, and executes it.
    Returns a tuple (raw_research_output, final_report_output).
    """
    researcher, writer, qa = create_agents(agent_prompts, llm, tools)
    task_research, task_write, task_qa = create_tasks(agent_prompts, research_topic, researcher, writer, qa)
    
    crew = Crew(
        agents=[researcher, writer, qa],
        tasks=[task_research, task_write, task_qa],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    
    raw_research = str(task_research.output.raw)
    final_report = str(result)
    
    return raw_research, final_report
