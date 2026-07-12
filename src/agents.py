from crewai import Agent

def agent_step_callback(step):
    try:
        if isinstance(step, list):
            for s in step:
                agent_step_callback(s)
            return

        log = getattr(step, 'log', '')
        tool = getattr(step, 'tool', '')
        tool_input = getattr(step, 'tool_input', '')
        
        if log:
            # LLM output might not have "Thought:" prefix, so we add it if missing
            if "Thought:" not in log and "Action:" not in log:
                print(f"Thought: {log}")
            else:
                print(log)
        if tool:
            print(f"Action: {tool}")
        if tool_input:
            print(f"Action Input: {tool_input}")
    except Exception:
        pass

def create_agents(agent_prompts, llm, tools):
    """
    Creates and returns the Researcher, Writer, and QA agents.
    """
    r_info = agent_prompts.get("agent_1", {})
    w_info = agent_prompts.get("agent_2", {})
    q_info = agent_prompts.get("agent_3", {})
    
    # 1. Senior Researcher
    researcher = Agent(
        role=r_info.get("role", "Senior Researcher"),
        goal=r_info.get("goal", ""),
        backstory=r_info.get("backstory", ""),
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )
    
    writer = Agent(
        role=w_info.get("role", "Technical Copywriter"),
        goal=w_info.get("goal", ""),
        backstory=w_info.get("backstory", ""),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )
    
    qa = Agent(
        role=q_info.get("role", "Quality Assurance & Fact-Checker"),
        goal=q_info.get("goal", ""),
        backstory=q_info.get("backstory", ""),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )
    
    return researcher, writer, qa
