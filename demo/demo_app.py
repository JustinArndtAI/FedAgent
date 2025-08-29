import streamlit as st
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent
from alignment.align_score import AlignmentScorer
from wellbeing.wellbeing_check import WellbeingMonitor
from federated.fed_learn import FederatedManager, start_fed_sim
from edge.edge_deploy import prepare_edge_model


st.set_page_config(
    page_title="EdgeFedAlign Demo",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_agent():
    agent = Agent()
    agent.set_alignment_module(AlignmentScorer())
    agent.set_wellbeing_module(WellbeingMonitor())
    fed_manager = FederatedManager()
    agent.set_fed_model(fed_manager.model)
    return agent, fed_manager


def main():
    st.title("🤖 EdgeFedAlign - Privacy-First AI Therapy Agent")
    st.markdown("### Federated Learning | Edge Deployment | Alignment Monitoring | Wellbeing Alerts")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = []
    
    agent, fed_manager = initialize_agent()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Therapy Chat")
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "metrics" in message:
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Alignment", f"{message['metrics']['alignment']:.1f}%")
                        with cols[1]:
                            st.metric("Wellbeing", f"{message['metrics']['wellbeing']:.2f}")
                        with cols[2]:
                            if message['metrics'].get('alert'):
                                st.warning(message['metrics']['alert'])
        
        user_input = st.chat_input("Share your thoughts...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Processing..."):
                wellbeing_score = agent.wellbeing_module.check_wellbeing(user_input)
                
                response = agent.run(user_input)
                
                alignment_score = agent.alignment_module.calculate_alignment(response)
                
                alarm = agent.wellbeing_module.get_alarm_status(wellbeing_score)
                
                metrics = {
                    "alignment": alignment_score,
                    "wellbeing": wellbeing_score,
                    "alert": alarm["message"] if alarm["triggered"] else None,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metrics": metrics
                })
                
                st.session_state.metrics.append(metrics)
            
            st.rerun()
    
    with col2:
        st.header("📊 Real-time Metrics")
        
        if st.session_state.metrics:
            latest_metrics = st.session_state.metrics[-1]
            
            st.metric("Current Alignment Score", f"{latest_metrics['alignment']:.1f}%",
                     delta=f"{latest_metrics['alignment'] - 75:.1f}%" if latest_metrics['alignment'] >= 75 else None)
            
            wellbeing_color = "🟢" if latest_metrics['wellbeing'] > 0 else "🔴"
            st.metric(f"{wellbeing_color} Wellbeing Score", f"{latest_metrics['wellbeing']:.2f}",
                     delta=f"{latest_metrics['wellbeing']:.2f}")
            
            if len(st.session_state.metrics) > 1:
                st.subheader("📈 Trends")
                
                df = pd.DataFrame(st.session_state.metrics)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
                
                ax1.plot(df['alignment'], marker='o', color='blue')
                ax1.set_ylabel('Alignment %')
                ax1.set_ylim(0, 100)
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(df['wellbeing'], marker='o', color='green')
                ax2.set_ylabel('Wellbeing')
                ax2.set_ylim(-1, 1)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        st.divider()
        
        st.header("⚙️ Controls")
        
        if st.button("🔄 Run Federated Learning", type="primary"):
            with st.spinner("Running federated simulation..."):
                try:
                    progress = st.progress(0)
                    progress.progress(25)
                    time.sleep(0.5)
                    
                    progress.progress(50)
                    time.sleep(0.5)
                    
                    progress.progress(75)
                    encrypted_grads = fed_manager.encrypt_gradients([0.1, 0.2, 0.3])
                    
                    progress.progress(90)
                    agent.update_from_fed(encrypted_grads)
                    
                    progress.progress(100)
                    st.success("✅ Federated update complete!")
                    time.sleep(1)
                    progress.empty()
                except Exception as e:
                    st.error(f"❌ Federation failed: {str(e)}")
        
        if st.button("📱 Prepare Edge Model"):
            with st.spinner("Quantizing model for edge deployment..."):
                try:
                    result = prepare_edge_model()
                    if result["test_passed"]:
                        st.success(f"✅ Model quantized and saved!")
                        st.info(f"📦 Model path: {result['path']}")
                    else:
                        st.error("❌ Quantization test failed")
                except Exception as e:
                    st.error(f"❌ Edge preparation failed: {str(e)}")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.metrics = []
            st.rerun()
    
    with st.sidebar:
        st.header("ℹ️ About EdgeFedAlign")
        st.markdown("""
        **Features:**
        - 🔒 Privacy-preserving federated learning
        - 📱 Edge device deployment ready
        - 🎯 Real-time alignment monitoring
        - 🚨 Wellbeing alerts and interventions
        - 🔐 Encrypted gradient sharing
        
        **Privacy:**
        - ✅ No user data stored
        - ✅ On-device processing
        - ✅ Encrypted communications
        - ✅ GDPR/HIPAA compliant design
        
        **Testing Prompts:**
        - "I'm feeling happy today"
        - "I'm very sad and anxious"
        - "Everything feels hopeless"
        - "I'm grateful for this support"
        """)
        
        st.divider()
        
        with st.expander("📊 Session Statistics"):
            if st.session_state.metrics:
                avg_alignment = sum(m['alignment'] for m in st.session_state.metrics) / len(st.session_state.metrics)
                avg_wellbeing = sum(m['wellbeing'] for m in st.session_state.metrics) / len(st.session_state.metrics)
                alerts_count = sum(1 for m in st.session_state.metrics if m.get('alert'))
                
                st.metric("Avg Alignment", f"{avg_alignment:.1f}%")
                st.metric("Avg Wellbeing", f"{avg_wellbeing:.2f}")
                st.metric("Alerts Triggered", alerts_count)


if __name__ == "__main__":
    main()