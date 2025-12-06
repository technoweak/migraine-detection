import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io, base64

# ===============================
# Load saved model and utilities
# ===============================
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# ===============================
# Migraine Info + Defaults
# ===============================
migraine_info = {
    "Basilar-Type Aura": [
        "Originates from the brainstem, involving disturbances like vertigo, dysarthria, or double vision.",
        "Commonly affects both sides of the body, unlike unilateral migraines.",
        "Often accompanied by visual and auditory hallucinations.",
        "Can lead to severe nausea and temporary loss of coordination.",
        "Considered one of the rare and complex migraine subtypes."
    ],
    "Familial Hemiplegic Migraine": [
        "A hereditary migraine type linked to genetic mutations (CACNA1A, ATP1A2, SCN1A genes).",
        "Causes temporary paralysis or weakness on one side of the body during an attack.",
        "Often associated with aura symptoms such as speech difficulties or vision loss.",
        "Can last from several hours to days, with gradual recovery.",
        "Typically diagnosed through family medical history and genetic testing."
    ],
    "Migraine Without Aura": [
        "The most prevalent migraine subtype worldwide.",
        "Characterized by pulsating headaches, nausea, photophobia, and phonophobia.",
        "Triggers include stress, hormonal changes, lack of sleep, or dietary factors.",
        "Lacks any preceding neurological warning signs (aura).",
        "Duration ranges from 4 to 72 hours if untreated."
    ],
    "Sporadic Hemiplegic Migraine": [
        "Shares similar clinical symptoms with familial hemiplegic migraine.",
        "Occurs without any familial or genetic background.",
        "Involves unilateral motor weakness or paralysis during an attack.",
        "Neurological symptoms usually resolve completely after the episode.",
        "Diagnosis is confirmed after excluding secondary neurological causes."
    ],
    "Typical Aura With Migraine": [
        "Begins with visual or sensory aura such as flashing lights or tingling sensations.",
        "Aura symptoms usually precede headache by 20–60 minutes.",
        "The headache phase follows, often on one side of the head.",
        "Common triggers include bright light exposure and emotional stress.",
        "Usually resolves within a few hours to a day."
    ],
    "Typical Aura Without Migraine": [
        "Presence of aura symptoms without subsequent headache.",
        "Often mistaken for transient ischemic attacks or other neurological events.",
        "Involves visual disturbances like zig-zag patterns or blind spots.",
        "Episodes last less than 60 minutes, with full recovery afterward.",
        "Occurs more commonly in individuals with a history of migraine with aura."
    ]
}

default_profiles = {
    "Basilar-Type Aura": [32, 2, 1, 1, 1, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "Familial Hemiplegic Migraine": [21, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    "Migraine Without Aura": [35, 2, 4, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Sporadic Hemiplegic Migraine": [21, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "Typical Aura With Migraine": [32, 1, 2, 1, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    "Typical Aura Without Migraine": [26, 1, 2, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0]
}

# ===============================
# Page Setup & Styling
# ===============================
st.set_page_config(page_title="Migraine Sense", layout="wide")

st.markdown("""
<style>
/* Navbar */
.navbar {
    background: linear-gradient(90deg, #0b0c10, #1f2833);
    padding: 1rem 2rem;
    border-radius: 12px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.nav-title {
    font-size: 1.6rem;
    font-weight: bold;
    color: #66fcf1;
}

/* Section styling with glow */
.section {
    background: rgba(102, 252, 241, 0.07);
    border: 1px solid rgba(102, 252, 241, 0.25);
    box-shadow: 0 0 25px rgba(102, 252, 241, 0.3);
    padding: 20px;
    border-radius: 16px;
    margin-top: 20px;
}

/* Sidebar Highlight */
[data-testid="stSidebar"] {
    background-color: #0b0c10 !important;
    border-right: 2px solid rgba(102,252,241,0.3);
}
.sidebar-highlight {
    background: linear-gradient(135deg, rgba(102,252,241,0.15), rgba(0,255,200,0.05));
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(102,252,241,0.3);
    border: 1px solid rgba(102,252,241,0.3);
}

/* Glowing Predict Button */
div.stButton > button {
    background: linear-gradient(90deg, #45a29e, #66fcf1);
    color: black;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    box-shadow: 0 0 20px rgba(102, 252, 241, 0.6);
    transition: all 0.3s ease-in-out;
    animation: glowPulse 1.5s infinite alternate;
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(102, 252, 241, 0.9);
}
@keyframes glowPulse {
    from { box-shadow: 0 0 10px rgba(102,252,241,0.4); }
    to { box-shadow: 0 0 25px rgba(102,252,241,0.9); }
}

/* Background */
body {
    background-color: #0b0c10;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Navbar with Default Type
# ===============================
col1, col2 = st.columns([4, 2])
with col1:
    st.markdown('<div class="navbar"><div class="nav-title">🧠 Migraine Sense</div></div>', unsafe_allow_html=True)
with col2:
    selected_default_type = st.selectbox("Select Default Type", list(default_profiles.keys()), index=0)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.markdown("<div class='sidebar-highlight'>", unsafe_allow_html=True)
    st.header("⚙️ Input Options")
    use_default = st.radio("Use Default Profile?", ("Yes", "No"))
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Input Section
# ===============================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("Enter Symptom Values")

if use_default == "Yes":
    migraine_type = selected_default_type
    user_input = default_profiles[migraine_type]
    st.success(f"✅ Using default values for **{migraine_type}**.")
else:
    migraine_type = "Manual Entry"
    user_input = []
    cols = st.columns(2)
    for i, feat in enumerate(selected_features):
        with cols[i % 2]:
            val = st.number_input(f"{feat}", min_value=0, max_value=100, value=1)
            user_input.append(val)
st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Prediction Section
# ===============================
if st.button("🔍 Predict Migraine Type"):
    input_scaled = scaler.transform([user_input])
    pred_encoded = best_model.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Prediction Results")
    st.write(f"**Input Profile Type:** {migraine_type}")
    st.write(f"**Predicted Migraine Type:** {pred_label}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # Feature Importance Modal Image
    # ===============================
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(importances)), importances[sorted_idx], color="teal")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([selected_features[i] for i in sorted_idx], rotation=45, ha="right")
        ax.set_ylabel("Importance Score")
        ax.set_title("Feature Importance (Random Forest)")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode()

        html_code = f"""
        <style>
        .thumbnail {{
            width: 280px;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s ease;
        }}
        .thumbnail:hover {{
            transform: scale(1.05);
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            backdrop-filter: blur(8px);
            background-color: rgba(0,0,0,0.6);
            justify-content: center;
            align-items: center;
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }} to {{ opacity: 1; }}
        }}
        .modal img {{
            max-width: 80%;
            max-height: 80%;
            border-radius: 12px;
            box-shadow: 0 0 25px rgba(255,255,255,0.3);
        }}
        </style>

        <div>
            <img src="data:image/png;base64,{image_base64}" class="thumbnail" id="thumbnailImg" />
        </div>

        <div class="modal" id="imageModal">
            <img src="data:image/png;base64,{image_base64}" />
        </div>

        <script>
        const thumb = document.getElementById("thumbnailImg");
        const modal = document.getElementById("imageModal");
        thumb.onclick = function() {{ modal.style.display = "flex"; }}
        modal.onclick = function() {{ modal.style.display = "none"; }}
        </script>
        """

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Feature Importance Chart")
        st.components.v1.html(html_code, height=400)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # Migraine Info Section
    # ===============================
    def normalize_label(s):
        return s.strip().lower().replace("-", "").replace(" ", "")

    norm_pred = normalize_label(pred_label)
    normalized_info_map = {normalize_label(k): v for k, v in migraine_info.items()}
    info_points = normalized_info_map.get(norm_pred, ["⚠️ No detailed info available."])

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader(f"Information About {pred_label}")
    for i, point in enumerate(info_points, 1):
        st.write(f"{i}. {point}")
    st.markdown("</div>", unsafe_allow_html=True)
