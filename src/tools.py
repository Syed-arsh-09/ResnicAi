import os
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import SerperDevTool

def setup_tools(serper_api_key):
    """
    Setup the search tools for the CrewAI agents.
    Returns a tuple of (tools_list, log_message)
    """
    tools = []
    log_msg = ""
    
    if serper_api_key:
        try:
            # Set key in environment for SerperDevTool
            os.environ["SERPER_API_KEY"] = serper_api_key
            search_tool = SerperDevTool()
            tools.append(search_tool)
            log_msg = "SerperDevTool initialized successfully.\n"
        except Exception as e:
            log_msg = f"Failed to initialize SerperDevTool: {e}. Falling back to DuckDuckGo.\n"
            tools.append(DuckDuckGoSearchRun())
    else:
        log_msg = "No Serper API key provided. Defaulting to DuckDuckGo search.\n"
        tools.append(DuckDuckGoSearchRun())
        
    return tools, log_msg
