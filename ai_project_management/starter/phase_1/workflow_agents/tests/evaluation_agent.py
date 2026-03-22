import sys
sys.path.append("/workspace/cd14525-agentic-workflows-classroom/project/")
from starter.phase_1.workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
from dotenv import load_dotenv
import os

################EvaluationAgent Test################

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
prompt = "What is the capital of France?"
evaluation_criteria = "The response should be clear, accurate, concise, and directly answer the user's question."

knowledge_agent = KnowledgeAugmentedPromptAgent(api_key, persona, knowledge)
eval_agent = EvaluationAgent(api_key, persona, evaluation_criteria, knowledge_agent, 10)

result = eval_agent.evaluate(prompt)
print("EvaluationAgent results")
print(result)
