import streamlit as st
import torch
import pickle
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import re
from datetime import datetime
from collections import Counter
import pandas as pd

# Load model, tokenizer, and emotion labels
@st.cache_resource
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained('Vipulydvv/BERTEMOTION')
    tokenizer = BertTokenizer.from_pretrained('Vipulydvv/BERTEMOTION')
    with open('bert_emotion_labels.pkl', 'rb') as f:
        emotion_labels = pickle.load(f)
    model.eval()
    return model, tokenizer, emotion_labels

model, tokenizer, emotion_labels = load_model_and_tokenizer()

# Intensity/emotional words
emotional_words = {
    'very': 0.1, 'really': 0.1, 'extremely': 0.2, 'absolutely': 0.2,
    'hate': 0.3, 'love': 0.3, 'terrible': 0.3, 'amazing': 0.3,
    'awful': 0.3, 'wonderful': 0.3, 'horrible': 0.3, 'fantastic': 0.3
}
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\u2702-\u27B0"
    u"\u24C2-\U0001F251"
    "]+", flags=re.UNICODE)

# Session state
if 'session_emotions' not in st.session_state:
    st.session_state['session_emotions'] = []

# --- Feature Functions ---
def analyze_intensity(text):
    intensity = 0.5  # Base
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.3:
        intensity += 0.2
    exclamation_count = text.count('!')
    intensity += min(exclamation_count * 0.1, 0.3)
    question_count = text.count('?')
    if question_count > 0:
        intensity += 0.1
    emojis = emoji_pattern.findall(text)
    if emojis:
        intensity += 0.15
    text_lower = text.lower()
    for word, boost in emotional_words.items():
        if word in text_lower:
            intensity += boost
    if detect_repetition(text):
        intensity += 0.1
    return min(intensity, 1.0)

def detect_repetition(text):
    words = text.lower().split()
    if len(words) < 3:
        return False
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    max_repetition = max(word_counts.values())
    return max_repetition > 2

def analyze_text_features(text):
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'punctuation_count': sum(1 for c in text if c in '!?.,;:'),
        'has_questions': '?' in text,
        'has_exclamations': '!' in text,
        'repetition': detect_repetition(text)
    }
    return features

def predict_emotions(text, threshold=0.3):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    predicted_emotions = []
    for i, prob in enumerate(probabilities):
        if prob > threshold:
            predicted_emotions.append((emotion_labels[i], float(prob)))
    predicted_emotions.sort(key=lambda x: x[1], reverse=True)
    confidence = float(np.max(probabilities))
    all_probs = {emotion_labels[i]: float(prob) for i, prob in enumerate(probabilities)}
    return predicted_emotions, confidence, all_probs

def get_session_insights():
    session = st.session_state['session_emotions']
    if not session:
        return None
    all_emotions = [entry['emotion'] for entry in session]
    counts = Counter(all_emotions)
    most_common = counts.most_common(1)[0] if counts else (None, 0)
    diversity = len(counts) / len(emotion_labels) if emotion_labels else 0
    avg_intensity = np.mean([entry['intensity'] for entry in session]) if session else 0
    return {
        'total': len(session),
        'most_common': most_common,
        'diversity': diversity,
        'avg_intensity': avg_intensity
    }

def get_suggestion(emotion):
    suggestions = {
        'joy': 'Keep spreading positivity! 😊',
        'love': 'Cherish your connections! 💖',
        'sadness': 'It’s okay to feel sad. Take care of yourself. 💙',
        'anger': 'Try some deep breaths or a walk. 🔴',
        'fear': 'You are not alone. 🟤',
        'surprise': 'Embrace the unexpected! 🟠',
        'neutral': 'Share more to discover your emotions!'
    }
    return suggestions.get(emotion, 'Thank you for sharing!')

# --- Detect theme and set CSS variables ---
theme = st.get_option("theme.base") or "light"
if theme == "dark":
    page_bg = "#181825"
    card_bg = "#23272f"
    card_grad = "linear-gradient(120deg, #23272f 60%, #374151 100%)"
    text_color = "#e0e7ef"
    accent = "#6366f1"
    border = "#374151"
    input_bg = "#23272f"
    input_text = "#e0e7ef"
else:
    page_bg = "#f8fafc"
    card_bg = "#f1f5f9"
    card_grad = "linear-gradient(120deg, #f1f5f9 60%, #e0e7ff 100%)"
    text_color = "#22223b"
    accent = "#6366f1"
    border = "#e0e7ef"
    input_bg = "#fff"
    input_text = "#22223b"

st.markdown(f"""
<style>
body, .stApp {{background: {page_bg} !important;}}
.big-title {{font-size:2.7rem; font-weight:800; color:{accent}; letter-spacing:-1px; margin-bottom:0.2em;}}
.footer {{text-align:center; color:gray; font-size:0.95rem; margin-top:2em;}}
.card {{background: {card_grad}; border-radius:16px; padding:1.7em 2em; margin-bottom:1.2em; box-shadow:0 4px 16px rgba(80,80,180,0.07); transition: box-shadow 0.2s; color:{text_color}; border: 1.5px solid {border};}}
.card:hover {{box-shadow:0 8px 32px rgba(80,80,180,0.13);}}
.metric {{font-size:1.2rem; font-weight:600; color:{accent};}}
.emoji {{font-size:2.7rem;}}
.suggestion {{font-size:1.1rem; color:#0ea5e9; font-weight:500;}}
hr {{border: none; border-top: 1.5px solid {border}; margin: 1.5em 0;}}
textarea, .stTextArea textarea, .stTextInput input, .stButton>button {{background-color: {input_bg} !important; color: {input_text} !important;}}
.stTextArea label, .stTextInput label {{color: {text_color} !important;}}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big-title">🎭 BERT Emotion Analyzer</div>', unsafe_allow_html=True)

# --- Main Tabs Layout ---
tabs = st.tabs(["Emotion Analysis", "Dashboard"])

with tabs[0]:
    st.write("Enter text below to analyze emotions. Only emotions with ≥10% probability are shown in the breakdown.")
    with st.form("emotion_form"):
        user_text = st.text_area("Enter your text:", height=100)
        submitted = st.form_submit_button("Analyze Emotion", use_container_width=True)

    if submitted and user_text.strip():
        predicted_emotions, confidence, all_probs = predict_emotions(user_text.strip(), threshold=0.3)
        intensity = analyze_intensity(user_text)
        features = analyze_text_features(user_text)
        if predicted_emotions:
            emotion, conf = predicted_emotions[0]
        else:
            emotion, conf = 'neutral', 0.0
        entry = {
            'text': user_text,
            'emotion': emotion,
            'confidence': conf,
            'intensity': intensity,
            'all_probabilities': all_probs,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': features
        }
        st.session_state['session_emotions'].append(entry)
        # --- Results Card ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cols = st.columns([1,2,2])
        emoji_map = {
            'joy': '🟡', 'love': '💖', 'sadness': '💙',
            'anger': '🔴', 'fear': '🟤', 'surprise': '🟠', 'neutral': '⚪️'
        }
        emoji = emoji_map.get(emotion, '❓')
        cols[0].markdown(f"<span class='emoji'>{emoji}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"<span class='metric'>Emotion:</span> <span style='font-size:1.5rem'>{emotion.upper()}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span class='suggestion'>Suggestion:<br>{get_suggestion(emotion)}</span>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Confidence**")
        st.progress(confidence)
        st.markdown("**Intensity**")
        st.progress(intensity)
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("Emotion Breakdown (≥10%)", expanded=True):
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            shown = False
            for emotion_name, prob in sorted_probs:
                if prob < 0.10:
                    continue
                bar = "█" * int(prob * 20)
                st.write(f"{emotion_name.capitalize():<10} {prob:>6.1%} {bar}")
                shown = True
            if not shown:
                st.write("No emotions above 10% probability.")
        with st.expander("Text Features", expanded=False):
            feat_cols = st.columns(3)
            for i, (feature, value) in enumerate(features.items()):
                feat_cols[i%3].write(f"- **{feature}**: {value}")
        st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    dashboard_option = st.radio(
        "Select dashboard view:",
        ("Overview", "Distribution", "Timeline", "Intensity Heatmap", "Download Session"),
        horizontal=True,
        key="dashboard_radio"
    )
    if st.button("Clear Session Data", key="clear_btn", use_container_width=True):
        st.session_state['session_emotions'] = []
        st.experimental_rerun()
    session = st.session_state['session_emotions']
    if dashboard_option == "Overview":
        st.subheader("Session Insights")
        insights = get_session_insights()
        if insights:
            st.write(f"**Total Entries:** {insights['total']}")
            st.write(f"**Most Frequent:** {insights['most_common'][0].capitalize()} ({insights['most_common'][1]})")
            st.write(f"**Diversity:** {insights['diversity']:.1%}")
            st.write(f"**Avg. Intensity:** {insights['avg_intensity']:.1%}")
        else:
            st.info("No data yet.")
    elif dashboard_option == "Distribution":
        st.subheader("Emotion Distribution (Pie Chart)")
        all_emotions = [entry['emotion'] for entry in session]
        if all_emotions:
            counts = Counter(all_emotions)
            labels = list(counts.keys())
            sizes = list(counts.values())
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax.set_title('Session Emotion Distribution')
            st.pyplot(fig)
        else:
            st.info("No emotion data to display yet.")
    elif dashboard_option == "Timeline":
        st.subheader("Emotion Timeline")
        if session:
            df = pd.DataFrame(session)
            st.line_chart(df.set_index('timestamp')['intensity'])
            for entry in session:
                st.write(f"[{entry['timestamp']}] {entry['emotion'].capitalize()} (Intensity: {entry['intensity']:.1%}) - {entry['text']}")
        else:
            st.info("No data yet.")
    elif dashboard_option == "Intensity Heatmap":
        st.subheader("Intensity Heatmap")
        if session:
            df = pd.DataFrame(session)
            fig, ax = plt.subplots(figsize=(8,2))
            ax.bar(range(len(df)), df['intensity'], color=accent)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([e['emotion'] for e in session], rotation=45)
            ax.set_ylabel('Intensity')
            ax.set_xlabel('Entry')
            st.pyplot(fig)
        else:
            st.info("No data yet.")
    elif dashboard_option == "Download Session":
        st.subheader("Download Session Data")
        if session:
            df = pd.DataFrame(session)
            st.download_button("Download as CSV", df.to_csv(index=False), file_name="emotion_session.csv")
        else:
            st.info("No data yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">Made with ❤️!</div>', unsafe_allow_html=True) 
