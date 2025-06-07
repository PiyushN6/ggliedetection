# app.py

import streamlit as st
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import for live camera/mic
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av # For handling video/audio frames
import plotly.graph_objects as go # For plotting probabilities
import tempfile # To save uploaded files temporarily

# --- 0. Project Disclaimers (Crucial for Ethical Presentation) ---
st.set_page_config(page_title="Multi-Modal Deception Detection Prototype", layout="wide")

st.markdown("""
# 🕵️‍♂️ Multi-Modal Deception Detection System (Prototype)

This is a **research prototype** demonstrating a multi-modal AI system for deception detection, combining text, audio, video, and physiological cues with an agentic **ReAct** (Reasoning + Acting) framework.

**⚠️ IMPORTANT DISCLAIMER:**
* This system is a **proof-of-concept**. The underlying models are either simplified placeholders or pre-trained models **not fine-tuned on actual deception datasets**.
* **The outputs are for demonstration purposes only and should NOT be used as reliable indicators of truth or deception.**
* Lie detection is a complex and ethically sensitive field. This tool is intended for academic exploration of AI architecture and ethical considerations, not for real-world application.
* The video, audio, and physiological components currently use **simulated random predictions** due to the complexity of live data capture and lack of fine-tuned models for this prototype.
""")

st.divider()

# --- 1. Model Definitions (from your Colab Notebook) ---

# @st.cache_resource ensures the model is loaded only once across reruns
@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    model.eval() # Set model to evaluation mode
    return tokenizer, model

tokenizer, text_model = load_bert_model()

# text_model_predict function (adapted from your notebook)
def text_model_predict(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    return probs.tolist() # Returns list of lists: [[prob_class_0, prob_class_1]]

# --- 2. GSPO Self-Play Agent (Simulated for background context, not interactive in web UI) ---
class SelfPlayAgent:
    def __init__(self, detector_model):
        self.model = detector_model
        self.learning = False
        self.training_history = []

    def enable_learning(self, flag=True):
        self.learning = flag

    def simulate_scenario(self):
        is_deceptive = random.choice([0, 1])
        simulated_data = {
            'video': None, 'audio': None, 'text': "simulated statement", 'physio': None
        }
        return simulated_data, is_deceptive

    def train_self_play(self, episodes=5):
        if not self.learning:
            return
        for ep in range(episodes):
            data, truth_label = self.simulate_scenario()
            pred_label = random.choice([0, 1])
            reward = 1 if pred_label == truth_label else -1
            self.training_history.append(reward)


# --- 3. Fusion Model (from your notebook) ---
def fuse_outputs(modality_results):
    """
    Fuses probabilities from different modalities.
    modality_results: list of dicts, e.g., [{'lie': 0.7, 'truth': 0.3}, ...]
    """
    if not modality_results:
        return "uncertain", 0.0

    total_lie_prob = sum(res['lie'] for res in modality_results)
    avg_lie_prob = total_lie_prob / len(modality_results)

    if avg_lie_prob > 0.5:
        decision = "lie"
        confidence_score = avg_lie_prob
    else:
        decision = "truth"
        confidence_score = 1 - avg_lie_prob # Confidence in truth

    return decision, confidence_score

# --- 4. ReAct Agent Implementation ---
# This function now takes a 'demo_bias_for_lie' argument
def react_agent_decision(video_input=None, audio_input=None, text_input=None, physio_input=None, demo_bias_for_lie=0.0):
    reasoning_trace = []
    modality_results = []
    individual_probs = {} # Store individual probabilities for visualization

    reasoning_trace.append("--- Observing Modalities ---")

    # Helper to generate biased random probabilities for demo
    def get_biased_prob(base_random_val, bias):
        # bias ranges from -1.0 (truth bias) to 1.0 (lie bias)
        # 0.0 means no bias (pure random)
        biased_val = base_random_val + bias * (0.5 - base_random_val) # Pushes towards 0.0 or 1.0 based on bias
        return max(0.01, min(0.99, biased_val)) # Clamp to ensure within 0-1 range

    # 1. Observe from each modality if available
    # For file uploads, we mark input as True, but still use simulated probs
    if video_input:
        vision_prob = get_biased_prob(random.random(), demo_bias_for_lie)
        modality_results.append({'lie': vision_prob, 'truth': 1-vision_prob})
        individual_probs['Video'] = vision_prob
        reasoning_trace.append(f"Vision analysis suggests lie probability {vision_prob:.2f}.")

    if audio_input:
        audio_prob = get_biased_prob(random.random(), demo_bias_for_lie)
        modality_results.append({'lie': audio_prob, 'truth': 1-audio_prob})
        individual_probs['Audio'] = audio_prob
        reasoning_trace.append(f"Audio analysis suggests lie probability {audio_prob:.2f}.")

    if text_input:
        probs = text_model_predict([text_input])
        lie_prob = float(probs[0][1]) # Assuming index 1 is 'Lie' probability
        truth_prob = float(probs[0][0]) # Assuming index 0 is 'Truth' probability
            
        modality_results.append({'lie': lie_prob, 'truth': truth_prob})
        individual_probs['Text'] = lie_prob
        reasoning_trace.append(f"Text analysis suggests lie probability {lie_prob:.2f} for '{text_input}'.")

    if physio_input:
        physio_prob = get_biased_prob(random.random(), demo_bias_for_lie)
        modality_results.append({'lie': physio_prob, 'truth': 1-physio_prob})
        individual_probs['Physiological'] = physio_prob
        reasoning_trace.append(f"Physiological analysis suggests lie probability {physio_prob:.2f}.")

    if not modality_results:
        return "No input provided", "Uncertain", 0.0, {}

    reasoning_trace.append("--- Reasoning ---")
    if len(modality_results) > 1:
        reasoning_trace.append("Combining all available modalities to form a conclusion.")
    else:
        reasoning_trace.append("Single modality provided, basing conclusion on that alone.")

    # 2. Act: fuse results to get final decision
    decision, confidence = fuse_outputs(modality_results)
    reasoning_trace.append(f"Final decision: {decision.upper()} (confidence {confidence:.2f}).")

    return "\n".join(reasoning_trace), decision, confidence, individual_probs

# --- Global Demo Bias Control ---
st.sidebar.header("Demo Controls")
st.sidebar.markdown("Use these controls to influence the demo's outcome for simulated modalities.")
demo_bias_for_lie = st.sidebar.slider(
    "Demo Bias (Simulated Modalities):",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Move towards -1.0 to bias results towards 'Truth', towards 1.0 for 'Lie'. 0.0 is random."
)
st.sidebar.info(f"Current Demo Bias: {'Truth' if demo_bias_for_lie < 0 else ('Lie' if demo_bias_for_lie > 0 else 'Neutral')}")


# --- Helper for displaying results ---
def display_results(trace, decision, confidence, individual_probs):
    st.subheader("Final Decision")
    if decision.lower() == "lie":
        st.error(f"## 🚨 {decision.upper()} ({confidence:.2f} Confidence)")
    elif decision.lower() == "truth":
        st.success(f"## ✅ {decision.upper()} ({confidence:.2f} Confidence)")
    else:
        st.warning(f"## ❔ {decision.upper()} ({confidence:.2f} Confidence)")

    st.subheader("Reasoning Trace")
    st.code(trace)

    if individual_probs:
        st.subheader("Individual Modality Probabilities (Lie)")
        # Create a Plotly bar chart
        modalities = list(individual_probs.keys())
        lie_probabilities = [individual_probs[m] for m in modalities]
        truth_probabilities = [1 - p for p in lie_probabilities]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=modalities, y=truth_probabilities, name='Truth Probability', marker_color='lightgreen'))
        fig.add_trace(go.Bar(x=modalities, y=lie_probabilities, name='Lie Probability', marker_color='salmon'))

        fig.update_layout(barmode='stack', title_text='Lie Probabilities by Modality',
                          yaxis_title='Probability', xaxis_title='Modality',
                          yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.info("Remember: This is a prototype. Results are not reliable for real-world use.")


# --- 5. Streamlit App Interface (Manual Input Section) ---

st.header("Analyze a Statement (Manual Input)")

col1_manual, col2_manual = st.columns([2, 1])

with col1_manual:
    user_text_manual = st.text_area("Enter a statement:", "I was at home all evening.", height=100, key="manual_text_area")

with col2_manual:
    st.subheader("Simulated Modalities")
    use_video_manual = st.checkbox("Simulate Video Input", value=True, help="Simulates presence of video data.", key="manual_video_checkbox")
    use_audio_manual = st.checkbox("Simulate Audio Input", value=True, help="Simulates presence of audio data.", key="manual_audio_checkbox")
    use_physio_manual = st.checkbox("Simulate Physiological Input", value=True, help="Simulates presence of physiological data.", key="manual_physio_checkbox")

if st.button("Analyze Manual Statement", type="primary", key="analyze_manual_button"):
    if not user_text_manual and not use_video_manual and not use_audio_manual and not use_physio_manual:
        st.warning("Please provide at least one input (text or simulated modality).")
    else:
        st.info("Running manual analysis...")
        
        video_input_for_agent = True if use_video_manual else None
        audio_input_for_agent = True if use_audio_manual else None
        physio_input_for_agent = True if use_physio_manual else None
        text_input_for_agent = user_text_manual if user_text_manual.strip() != "" else None

        trace, decision, confidence, individual_probs = react_agent_decision(
            video_input=video_input_for_agent,
            audio_input=audio_input_for_agent,
            text_input=text_input_for_agent,
            physio_input=physio_input_for_agent,
            demo_bias_for_lie=demo_bias_for_lie # Pass the demo bias
        )
        display_results(trace, decision, confidence, individual_probs)

st.divider()

# --- 6. Live Analysis Section (using streamlit-webrtc) ---
st.header("Live Analysis (Webcam & Microphone)")
st.warning("""
**⚠️ IMPORTANT:** Live analysis is **experimental** and uses simulated feature extraction.
Your browser will ask for camera and microphone permissions.
The live video stream is displayed, but actual analysis relies on random probabilities, just like the simulated inputs.
""")

class LiveVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # For this prototype, we're simply passing the frame back to display it.
        return frame

webrtc_ctx = webrtc_streamer(
    key="live-stream-detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=LiveVideoProcessor,
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("Live stream active. Data is being conceptually processed (simulated).")

    user_text_live = st.text_area("Enter a statement while live:", "I am telling the truth via webcam and microphone.", key="live_text_area")

    if st.button("Analyze Live Statement", type="primary", key="analyze_live_button"):
        st.info("Running live analysis (simulated feature extraction)...")

        trace, decision, confidence, individual_probs = react_agent_decision(
            video_input=webrtc_ctx.video_processor is not None,
            audio_input=webrtc_ctx.audio_receiver is not None,
            text_input=user_text_live if user_text_live.strip() != "" else None,
            physio_input=False,
            demo_bias_for_lie=demo_bias_for_lie # Pass the demo bias
        )
        display_results(trace, decision, confidence, individual_probs)

else:
    st.info("Click the 'Start' button above (within the live stream component) to activate your webcam and microphone.")


st.divider()

# --- 7. Upload File Analysis Section ---
st.header("Analyze from Uploaded Files")
st.warning("""
**⚠️ IMPORTANT:** File analysis uses **simulated feature extraction**.
Uploaded video/audio will be displayed, but the analysis relies on random probabilities for the prototype.
""")

col_upload_video, col_upload_audio = st.columns(2)

uploaded_video_file = col_upload_video.file_uploader("Upload a Video File (.mp4)", type=["mp4"], key="video_uploader")
uploaded_audio_file = col_upload_audio.file_uploader("Upload an Audio File (.wav)", type=["wav"], key="audio_uploader")

uploaded_text_input = st.text_area("Enter a statement for uploaded files:", "I am telling the truth in this recording.", key="upload_text_area")
uploaded_physio_checkbox = st.checkbox("Simulate Physiological Input for uploaded files", value=True, key="upload_physio_checkbox")

if st.button("Analyze Uploaded Files", type="primary", key="analyze_uploaded_button"):
    video_present = uploaded_video_file is not None
    audio_present = uploaded_audio_file is not None
    text_present = uploaded_text_input.strip() != ""
    physio_present = uploaded_physio_checkbox

    if not (video_present or audio_present or text_present or physio_present):
        st.warning("Please upload at least one file, enter text, or simulate physiological input.")
    else:
        st.info("Running analysis on uploaded files (simulated feature extraction)...")

        # Display uploaded video/audio for user feedback
        if uploaded_video_file:
            st.subheader("Uploaded Video")
            st.video(uploaded_video_file)
            # In a real system, you'd process the video file here
            # For now, just indicate its presence to react_agent_decision
        
        if uploaded_audio_file:
            st.subheader("Uploaded Audio")
            st.audio(uploaded_audio_file, format='audio/wav')
            # In a real system, you'd process the audio file here

        trace, decision, confidence, individual_probs = react_agent_decision(
            video_input=video_present,
            audio_input=audio_present,
            text_input=uploaded_text_input if text_present else None,
            physio_input=physio_present,
            demo_bias_for_lie=demo_bias_for_lie # Pass the demo bias
        )
        display_results(trace, decision, confidence, individual_probs)


st.divider()

# --- 8. Explainability Section (Simplified) ---
st.header("Text Model Explainability (LIME)")
st.write("This section demonstrates how LIME helps understand what words influence the text model's decision.")

lime_text_input = st.text_input("Enter text for LIME explanation:", "I completely forgot about that meeting.", key="lime_text_input")

if st.button("Generate LIME Explanation", key="generate_lime_button"):
    if lime_text_input.strip() == "":
        st.warning("Please enter some text for LIME explanation.")
    else:
        try:
            from lime.lime_text import LimeTextExplainer
            
            explainer = LimeTextExplainer(class_names=["Truth", "Lie"])

            def lime_predict_proba_wrapper(texts):
                raw_probabilities = text_model_predict(texts)
                return np.array(raw_probabilities)

            st.spinner("Generating LIME explanation...")
            exp = explainer.explain_instance(
                lime_text_input,
                lime_predict_proba_wrapper,
                num_features=5
            )
            
            st.subheader("Top Influences for Prediction:")
            explanation_list = exp.as_list()
            for word, score in explanation_list:
                st.write(f"- **{word}**: {score:.3f}")
            

        except ImportError:
            st.error("LIME library not found. Please install it (`pip install lime`) to use this feature.")
        except Exception as e:
            st.error(f"An error occurred during LIME explanation: {e}")
            st.write("Please ensure your `text_model_predict` function is working correctly and returns probabilities in the format `[[prob_truth, prob_lie]]`.")

st.markdown("---")
st.markdown("Developed with ❤️ for academic purposes.")
