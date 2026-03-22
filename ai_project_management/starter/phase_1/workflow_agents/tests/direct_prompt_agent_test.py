import sys
sys.path.append("/workspace/cd14525-agentic-workflows-classroom/project/")
from starter.phase_1.workflow_agents.base_agents import DirectPromptAgent
from dotenv import load_dotenv
import os

################DirectPromptAgent Class Test################

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

direct_agent = DirectPromptAgent(api_key)
result = direct_agent.respond("What is the Capital of France?")
assert result == "The capital of France is Paris."
print("Knowledge source: general knowledge from the LLM model.")
print(result)
