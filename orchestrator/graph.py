# orchestrator/graph.py
from langgraph.graph import StateGraph, END

def build_graph(ingestion, imaging, therapy, pharmacy, doctor):
    sg = StateGraph(dict)

    def node_ingestion(state): return {**state, **ingestion.run(state).output}
    def node_imaging(state): return {**state, **imaging.run(state).output}
    def node_therapy(state): return {**state, **therapy.run(state).output}
    def node_pharmacy(state):
        items = [{"sku": state["otc_options"][0]["sku"], "qty": 1}] if state.get("otc_options") else []
        ph_out = pharmacy.run({"pincode": state.get("pincode"), "items": items, "red_flags": state.get("red_flags",[]) }).output
        return {**state, "pharmacy": ph_out, "items": items}
    def node_doctor(state):
        top_prob = max(state.get("condition_probs", {}).values()) if state.get("condition_probs") else 0
        need = top_prob < 0.5 or bool(state.get("red_flags"))
        return {**state, "escalation": doctor.run(state).output if need else {}}

    sg.add_node("ingestion", node_ingestion)
    sg.add_node("imaging", node_imaging)
    sg.add_node("therapy", node_therapy)
    sg.add_node("pharmacy", node_pharmacy)
    sg.add_node("doctor", node_doctor)

    sg.set_entry_point("ingestion")
    sg.add_edge("ingestion","imaging")
    sg.add_edge("imaging","therapy")
    sg.add_edge("therapy","pharmacy")
    sg.add_edge("pharmacy","doctor")
    sg.add_edge("doctor", END)

    return sg.compile()