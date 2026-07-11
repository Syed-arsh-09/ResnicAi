from crewai import Task

def create_tasks(agent_prompts, research_topic, researcher, writer, qa):
    """
    Creates and returns the tasks for the CrewAI agents.
    """
    r_info = agent_prompts.get("agent_1", {})
    w_info = agent_prompts.get("agent_2", {})
    q_info = agent_prompts.get("agent_3", {})
    
    task_research = Task(
        description=f"Conduct extensive research on the topic: '{research_topic}'. Gather facts, quotes, stats, and capture exact source URLs.",
        expected_output=r_info.get("expected_output", ""),
        agent=researcher
    )
    
    task_write = Task(
        description=f"Synthesize the research gathered into a beautifully structured technical brief about: '{research_topic}'. Ensure all inline source URLs are preserved.",
        expected_output=w_info.get("expected_output", ""),
        agent=writer
    )
    
    task_qa = Task(
        description=f"Fact-check the draft for '{research_topic}' against the raw research results. Fix any missing/incorrect citations, resolve any hallucinations, and format perfectly.",
        expected_output=q_info.get("expected_output", ""),
        agent=qa
    )
    
    return task_research, task_write, task_qa
