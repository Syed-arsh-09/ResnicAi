from crewai import Agent

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
        allow_delegation=False
    )
    
    # 2. Technical Copywriter
    writer = Agent(
        role=w_info.get("role", "Technical Copywriter"),
        goal=w_info.get("goal", ""),
        backstory=w_info.get("backstory", ""),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # 3. Quality Assurance
    qa = Agent(
        role=q_info.get("role", "Quality Assurance & Fact-Checker"),
        goal=q_info.get("goal", ""),
        backstory=q_info.get("backstory", ""),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    return researcher, writer, qa
