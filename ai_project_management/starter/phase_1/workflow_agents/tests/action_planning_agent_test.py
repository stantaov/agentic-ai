import sys
sys.path.append("/workspace/cd14525-agentic-workflows-classroom/project/")
from starter.phase_1.workflow_agents.base_agents import ActionPlanningAgent
from dotenv import load_dotenv
import os

################ActionPlanningAgent Test################

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
prompt = "One morning I wanted to have scrambled eggs"

knowledge = """
    To make scrambled eggs:
    1. Crack the eggs into a bowl
    2. Whisk the eggs
    3. Heat a pan
    4. Add butter or oil
    5. Pour in the eggs
    6. Stir gently until cooked
    7. Serve the scrambled eggs
    """

planning_agent = ActionPlanningAgent(api_key, knowledge)

result = planning_agent.extract_steps_from_prompt(prompt)
print(result)