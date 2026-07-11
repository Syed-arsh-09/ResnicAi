import os
import re
import sys
import threading
import streamlit as st

class ThreadSafeStdoutRedirector:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.redirects = {} # thread_id -> callback
        self.lock = threading.Lock()

    def register(self, thread_id, callback):
        with self.lock:
            self.redirects[thread_id] = callback

    def unregister(self, thread_id):
        with self.lock:
            self.redirects.pop(thread_id, None)

    def write(self, data):
        tid = threading.get_ident()
        with self.lock:
            callback = self.redirects.get(tid)
        if callback:
            callback(data)
        self.original_stdout.write(data)

    def flush(self):
        self.original_stdout.flush()

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
