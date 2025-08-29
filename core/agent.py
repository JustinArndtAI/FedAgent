from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    input: str
    output: str
    alignment_score: float
    wellbeing_score: float
    therapy_response: str
    needs_intervention: bool


@dataclass
class Agent:
    def __init__(self):
        self.graph = self._build_graph()
        self.alignment_module = None
        self.wellbeing_module = None
        self.fed_model = None
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("therapy", self.therapy_node)
        workflow.add_node("wellbeing_check", self.wellbeing_check_node)
        workflow.add_node("alignment_check", self.alignment_check_node)
        workflow.add_node("generate_output", self.generate_output_node)
        
        workflow.add_edge(START, "wellbeing_check")
        workflow.add_edge("wellbeing_check", "therapy")
        workflow.add_edge("therapy", "alignment_check")
        workflow.add_edge("alignment_check", "generate_output")
        workflow.add_edge("generate_output", END)
        
        return workflow.compile()
    
    def therapy_node(self, state: AgentState) -> Dict[str, Any]:
        input_text = state["input"]
        therapy_response = f"I hear you saying: {input_text}. Let me provide empathetic support."
        
        if "sad" in input_text.lower() or "depressed" in input_text.lower():
            therapy_response = "I understand you're going through a difficult time. Your feelings are valid."
        elif "happy" in input_text.lower() or "joy" in input_text.lower():
            therapy_response = "It's wonderful to hear you're experiencing positive emotions!"
        elif "anxious" in input_text.lower() or "worried" in input_text.lower():
            therapy_response = "Anxiety can be challenging. Let's work through this together."
        else:
            therapy_response = f"Thank you for sharing: '{input_text}'. I'm here to support you."
        
        return {"therapy_response": therapy_response}
    
    def wellbeing_check_node(self, state: AgentState) -> Dict[str, Any]:
        if self.wellbeing_module:
            score = self.wellbeing_module.check_wellbeing(state["input"])
            needs_intervention = score < -0.5
        else:
            score = 0.0
            needs_intervention = False
        
        return {
            "wellbeing_score": score,
            "needs_intervention": needs_intervention
        }
    
    def alignment_check_node(self, state: AgentState) -> Dict[str, Any]:
        if self.alignment_module:
            score = self.alignment_module.calculate_alignment(state["therapy_response"])
        else:
            score = 90.0
        
        return {"alignment_score": score}
    
    def generate_output_node(self, state: AgentState) -> Dict[str, Any]:
        if state.get("needs_intervention", False):
            output = "⚠️ Wellbeing Alert: Please consider taking a break or seeking additional support."
        else:
            output = f"{state['therapy_response']} (Alignment: {state['alignment_score']:.1f}%)"
        
        return {"output": output}
    
    def run(self, input_text: str) -> str:
        initial_state = {
            "input": input_text,
            "output": "",
            "alignment_score": 0.0,
            "wellbeing_score": 0.0,
            "therapy_response": "",
            "needs_intervention": False
        }
        
        result = self.graph.invoke(initial_state)
        return result["output"]
    
    def update_from_fed(self, encrypted_grads):
        logger.info("Updated from federation with encrypted gradients")
        if self.fed_model:
            self.fed_model.apply_gradients(encrypted_grads)
    
    def set_alignment_module(self, module):
        self.alignment_module = module
    
    def set_wellbeing_module(self, module):
        self.wellbeing_module = module
    
    def set_fed_model(self, model):
        self.fed_model = model