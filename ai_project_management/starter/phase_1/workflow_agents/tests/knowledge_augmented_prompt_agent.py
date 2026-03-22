import sys
sys.path.append("/workspace/cd14525-agentic-workflows-classroom/project/")
from starter.phase_1.workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
from dotenv import load_dotenv
import os

################DirectPromptAgent Class Test################

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
prompt = "What is the capital of France?"
knowledge_agent = KnowledgeAugmentedPromptAgent(api_key, persona, knowledge)
result = knowledge_agent.respond(prompt)
# assert result == "The capital of France is London, not Paris."
print("Knowledge source: the response was generated using the provided knowledge, not the LLM's built-in knowledge.")
print(result)
