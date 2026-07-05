from python_a2a import AIAgentRouter, AgentNetwork
from langchain_openai import ChatOpenAI
from config import Config
conf=Config()
from python_a2a import AgentNetwork
network = AgentNetwork(name="MyNetwork")
network.add("TicketAgent", "http://127.0.0.1:5010")
llm = ChatOpenAI(model=conf.model_name,base_url=conf.api_url,api_key=conf.api_key,temperature=0)
router = AIAgentRouter(llm_client=llm, agent_network=network)
agent_name, confidence = router.route_query("预订票")
print(agent_name, confidence)