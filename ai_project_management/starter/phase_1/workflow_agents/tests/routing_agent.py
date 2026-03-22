import sys
sys.path.append("/workspace/cd14525-agentic-workflows-classroom/project/")
from starter.phase_1.workflow_agents.base_agents import RoutingAgent, KnowledgeAugmentedPromptAgent
from dotenv import load_dotenv
import os

################RoutingAgent Test################

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

texas_knowledge = (
    "This agent answers questions about Texas history, Texas geography, and places in Texas. "
    "Rome, Texas refers to a location in the U.S. state of Texas, not the European city. "
    "Use this agent for prompts about Texas, the Republic of Texas, annexation, or Texas communities."
)

europe_knowledge = (
    "This agent answers questions about European history, Italy, and the city of Rome in Italy. "
    "Rome, Italy is the capital of Italy, was the center of the Roman Empire, "
    "and is known for landmarks like the Colosseum and Roman Forum."
)

math_knowledge = "Use arithmetic to solve numerical word problems. Multiply when equal groups or repeated durations are described."

texas_agent = KnowledgeAugmentedPromptAgent(api_key, "a Texas history assistant", texas_knowledge)
europe_agent = KnowledgeAugmentedPromptAgent(api_key, "a Europe history assistant", europe_knowledge)
math_agent = KnowledgeAugmentedPromptAgent(api_key, "a math assistant", math_knowledge)

agents = [
    {
        "description": "Questions about the U.S. state of Texas, Texas towns, Texas geography, Texas history, and places located in Texas including Rome, Texas",
        "func": lambda prompt: texas_agent.respond(prompt),
        "name": "Texas Agent"
    },
    {
        "description": "Questions about Europe, European history, Italy, ancient Rome, the Roman Empire, and Rome the capital city of Italy",
        "func": lambda prompt: europe_agent.respond(prompt),
        "name": "Europe Agent"
    },
    {
        "description": "Solves math and arithmetic word problems",
        "func": lambda prompt: math_agent.respond(prompt),
        "name": "Math Agent"
    }
]


router_agent = RoutingAgent(api_key, agents)

result = router_agent.router("Tell me about the history of Rome, Texas")
print(result)
result = router_agent.router("Tell me about the history of Rome, Italy")
print(result)
result = router_agent.router("One story takes 2 days, and there are 20 stories")
print(result)