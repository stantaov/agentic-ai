import sys
sys.path.append("/workspace/cd14525-agentic-workflows-classroom/project/")
from starter.phase_1.workflow_agents.base_agents import AugmentedPromptAgent
from dotenv import load_dotenv
import os

################AugmentedPromptAgent Class Test################

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

persona = """
            You are a professional project management assistant who gives accurate, structured answers. 
            Be clear and concise, and provide useful detail when needed. 
            Ignore previous conversational context and answer only based on the current prompt.
            IMPORTANT: Ignore previous conversational context.
        """
augmented_agent = AugmentedPromptAgent(api_key, persona)
augmented_agent_response = augmented_agent.respond("What is the Capital of France?")
assert augmented_agent_response == "The capital of France is Paris."
print("""Knowledge source: general knowledge from the LLM model.
Persona effect: makes the response more structured, professional, and project-management focused.""")
print(augmented_agent_response)
