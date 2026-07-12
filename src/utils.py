import os
import re
import sys
import threading
import streamlit as st

ACTIVE_CALLBACK = None

class ThreadSafeStreamRedirector:
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, data):
        global ACTIVE_CALLBACK
        if ACTIVE_CALLBACK:
            try:
                ACTIVE_CALLBACK(data)
            except Exception:
                pass
        try:
            self.original_stream.write(data)
        except UnicodeEncodeError:
            self.original_stream.write(data.encode('ascii', 'replace').decode('ascii'))

    def flush(self):
        self.original_stream.flush()

def set_log_callback(callback):
    global ACTIVE_CALLBACK
    ACTIVE_CALLBACK = callback

def clear_log_callback():
    global ACTIVE_CALLBACK
    ACTIVE_CALLBACK = None

def clean_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def load_agent_prompts(filepath="Prompts.txt"):
    if not os.path.exists(filepath):
        st.error(f"Prompts file not found at {filepath}")
        return None
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split content by Agent blocks
    blocks = re.split(r'Agent \d+:\s*', content)
    blocks = [b.strip() for b in blocks if b.strip()]
    
    agents = {}
    for i, block in enumerate(blocks, 1):
        role_match = re.search(r'Role:\s*(.*?)\s*Goal:', block, re.DOTALL)
        goal_match = re.search(r'Goal:\s*(.*?)\s*Backstory:', block, re.DOTALL)
        backstory_match = re.search(r'Backstory:\s*(.*?)\s*Expected Output \(Task Level\):', block, re.DOTALL)
        expected_output_match = re.search(r'Expected Output \(Task Level\):\s*(.*)', block, re.DOTALL)
        
        agents[f"agent_{i}"] = {
            "role": role_match.group(1).strip() if role_match else f"Agent {i}",
            "goal": goal_match.group(1).strip() if goal_match else "",
            "backstory": backstory_match.group(1).strip() if backstory_match else "",
            "expected_output": expected_output_match.group(1).strip() if expected_output_match else ""
        }
    return agents
