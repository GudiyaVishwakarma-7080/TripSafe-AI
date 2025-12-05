# ==============================================================================
# "TripSafe AI: Intelligent Hazard Detection System"
# Updated: Changed Background to a Safety/Care related soft image
# ==============================================================================

import streamlit as st
import numpy as np
import cv2
import os
import random
import time
import urllib.request
import base64
from PIL import Image
from io import BytesIO

# --- Try Importing gTTS for Audio Alerts ---
try:
    from gtts import gTTS
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# --- Page Config (Must be first) ---
st.set_page_config(
    page_title="TripSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Localization / Translations ---
LANGUAGES = {
    "English": {
        "title": "TripSafe AI",
        "tagline": "Advanced Indoor Safety",
        "guide_title": "🛡️ Quick Guide",
        "how_to_use": "### 🚀 Getting Started",
        "step1": "1. **Select Source:** Use 'Upload' for photos or 'Camera' for live scan.",
        "step2": "2. **Capture:** Ensure the floor area is visible and well-lit.",
        "step3": "3. **Analyze:** AI scans for trip hazards instantly.",
        "step4": "4. **Act:** Follow placement suggestions to secure the room.",
        "tips_title": "### 🧠 Safety Insights",
        "tips_list": [
            "**Cable Management:** Use velcro ties or cable clips along baseboards.",
            "**Rug Safety:** Apply double-sided anti-slip tape to all rug corners.",
            "**Stairways:** Keep stairs 100% clear. Install motion-sensor lights.",
            "**Lighting:** Ensure high-traffic areas have at least 300 lumens of light.",
            "**Storage:** Heavy items go down; light items go up."
        ],
        "daily_tip": "💡 **Daily Safety Tip:**",
        "placement_title": "#### ✅ Best Practices:",
        "place_text": """
        - 🎒 **Bags:** Designated hooks/closets.
        - 👟 **Shoes:** Shoe rack at entry.
        - 💻 **Electronics:** Desks with cable organizers.
        - ☕ **Dishes:** Sink or dishwasher immediately.
        """,
        "tab_home": "🏠 Home",
        "tab_scanner": "🔍 Scanner",
        "tab_settings": "⚙️ Config",
        "tab_info": "ℹ️ Info & Support",
        "home_hero_title": "Making Every Step Safe.",
        "home_hero_subtitle": "AI-powered vision to detect hazards and prevent falls before they happen.",
        "home_features_title": "Why Choose TripSafe?",
        "input_source": "### 📸 Input Channel",
        "select": "Mode:",
        "upload": "File Upload",
        "camera": "Live Camera",
        "high_risk": "CRITICAL RISK",
        "caution": "CAUTION ADVISED",
        "safe": "SAFE ENVIRONMENT",
        "high_risk_msg": "⚠️ Alert: {count} trip hazards detected. Immediate action required.",
        "caution_msg": "⚠️ Caution: Objects found on the floor. Proceed with care.",
        "safe_msg": "✅ Status: Area is clear and safe.",
        "status": "Safety Status",
        "hazards": "Hazards Found",
        "safe_zones": "Safe Zones",
        "suggestions": "🤖 AI Recommendations:",
        "download_report": "📥 Export Safety Report",
        "settings_config": "### ⚙️ System Parameters",
        "sensitivity": "**AI Sensitivity**",
        "alerts": "**Notifications**",
        "enable_audio": "Voice Alerts",
        "about_title": "### 🛡️ Our Mission",
        "about_text": "**TripSafe AI** is dedicated to reducing indoor accidents through cutting-edge computer vision. Designed for the elderly and visually impaired, it acts as a vigilant second pair of eyes.",
        "contact_title": "### 📞 Team & Contact",
        "contact_text": "We'd love to hear your feedback or collaboration ideas.",
        "contact_name": "Gudiya Vishwakarma",
        "contact_role": "Lead Developer & Researcher",
        "contact_email": "Akankshavish470@gmail.com"
    },
    "Hindi": {
        "title": "TripSafe AI",
        "tagline": "उन्नत घरेलू सुरक्षा",
        "guide_title": "🛡️ गाइड",
        "how_to_use": "### 🚀 शुरुआत कैसे करें",
        "step1": "1. **स्रोत चुनें:** 'अपलोड' या 'कैमरा' का उपयोग करें।",
        "step2": "2. **फोटो लें:** सुनिश्चित करें कि फर्श साफ दिखाई दे रहा है।",
        "step3": "3. **विश्लेषण:** AI तुरंत खतरों को स्कैन करता है।",
        "step4": "4. **कार्रवाई:** सुझावों का पालन करें और घर सुरक्षित रखें।",
        "tips_title": "### 🧠 सुरक्षा ज्ञान",
        "tips_list": [
            "**तार (Cables):** तारों को दीवारों के साथ क्लिप से बांधें।",
            "**कालीन (Rugs):** कोनों पर 'एंटी-स्लिप टेप' जरूर लगाएं।",
            "**सीढ़ियाँ:** सीढ़ियों को हमेशा खाली और साफ रखें।",
            "**रोशनी:** चलने वाले रास्तों में पर्याप्त रोशनी रखें।",
            "**स्टोरेज:** भारी सामान नीचे, हल्का सामान ऊपर।"
        ],
        "daily_tip": "💡 **आज का सुझाव:**",
        "placement_title": "#### ✅ सही आदतें:",
        "place_text": """
        - 🎒 **बैग:** हुक या अलमारी में।
        - 👟 **जूते:** शू रैक में।
        - 💻 **इलेक्ट्रॉनिक्स:** डेस्क पर।
        - ☕ **बर्तन:** सिंक में तुरंत डालें।
        """,
        "tab_home": "🏠 होम",
        "tab_scanner": "🔍 स्कैनर",
        "tab_settings": "⚙️ सेटिंग्स",
        "tab_info": "ℹ️ जानकारी और सहायता",
        "home_hero_title": "हर कदम सुरक्षित।",
        "home_hero_subtitle": "AI-संचालित तकनीक जो गिरने से पहले खतरों को पहचानती है।",
        "home_features_title": "TripSafe ही क्यों?",
        "input_source": "### 📸 इनपुट मोड",
        "select": "मोड:",
        "upload": "फाइल अपलोड",
        "camera": "लाइव कैमरा",
        "high_risk": "गंभीर जोखिम",
        "caution": "सावधानी बरतें",
        "safe": "सुरक्षित क्षेत्र",
        "high_risk_msg": "⚠️ चेतावनी: {count} खतरे मिले हैं। तुरंत हटाएं।",
        "caution_msg": "⚠️ ध्यान दें: फर्श पर सामान है। संभलकर चलें।",
        "safe_msg": "✅ स्थिति: क्षेत्र पूरी तरह सुरक्षित है।",
        "status": "सुरक्षा स्थिति",
        "hazards": "खतरे मिले",
        "safe_zones": "सुरक्षित जगहें",
        "suggestions": "🤖 AI सुझाव:",
        "download_report": "📥 रिपोर्ट डाउनलोड करें",
        "settings_config": "### ⚙️ सिस्टम सेटिंग्स",
        "sensitivity": "**AI संवेदनशीलता**",
        "alerts": "**सूचनाएं**",
        "enable_audio": "वॉयस अलर्ट",
        "about_title": "### 🛡️ हमारा मिशन",
        "about_text": "**TripSafe AI** कंप्यूटर विजन के माध्यम से घरेलू दुर्घटनाओं को कम करने के लिए समर्पित है। विशेष रूप से बुजुर्गों के लिए डिज़ाइन किया गया, यह एक अतिरिक्त सुरक्षा कवच है।",
        "contact_title": "### 📞 टीम संपर्क",
        "contact_name": "गुड़िया विश्वकर्मा",
        "contact_role": "लीड डेवलपर और रिसर्चर",
        "contact_email": "Akankshavish470@gmail.com"
    }
}

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
/* --- BACKGROUND IMAGE SETTING --- */
.stApp {
    background: linear-gradient(rgba(0, 0, 0, 0.70), rgba(0, 0, 0, 0.80)),
                url('https://images.pexels.com/photos/7218525/pexels-photo-7218525.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #f8fafc;
}
/* --- HEADER STYLING --- */
.custom-header {
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 10px 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 15px;
}
/* --- FOOTER STYLING --- */
.custom-footer {
    margin-top: 50px;
    padding: 30px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    color: #94a3b8;
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(5px);
}
.custom-footer a, .custom-footer summary {
    color: #38bdf8;
    text-decoration: none;
    margin: 0 10px;
    cursor: pointer;
    display: inline-block;
}
.custom-footer a:hover, .custom-footer summary:hover {
    text-decoration: underline;
    color: white;
}
/* --- POPUP CONTENT STYLING (ACCORDION FIX) --- */
details {
    display: inline-block;
    vertical-align: top;
    margin: 0 5px;
}
details > div {
    background: rgba(15, 23, 42, 0.98);
    padding: 15px;
    border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.3);
    margin-top: 10px; 
    text-align: left;
    color: #cbd5e1;
    font-size: 0.85rem;
    max-width: 300px;
    margin-left: auto;
    margin-right: auto;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
/* --- Navbar Styling (Compact & Inline) --- */
.header-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 15px;
    padding: 5px 0px; 
    background: transparent; 
    margin-bottom: 5px; 
}
/* Logo Image Style */
.header-logo {
    height: 55px;
    width: auto;
    object-fit: contain;
    filter: drop-shadow(0 0 5px rgba(56, 189, 248, 0.6));
    transition: transform 0.3s ease;
    margin-bottom: 0px;
}
.header-logo:hover {
    transform: scale(1.1) rotate(5deg);
}
/* Title Text */
.header-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.header-tagline {
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0;
}
/* --- Tabs Styling --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    padding: 0;
    flex-wrap: wrap; 
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-top: 0px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    min-width: 80px; 
    border-radius: 8px 8px 0 0;
    border: none;
    color: #cbd5e0;
    font-weight: 600;
    background-color: transparent;
    padding: 0 15px;
    font-size: 0.9rem;
    white-space: nowrap; 
    flex-grow: 0;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(255, 255, 255, 0.05);
    color: white;
}
.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8;
}
/* --- Cards & Containers --- */
.feature-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 20px; 
    transition: all 0.3s ease;
    min-height: 150px; 
    display: flex;
    flex-direction: column;
    justify-content: flex-start; 
    backdrop-filter: blur(10px);
}
.feature-card:hover {
    transform: translateY(-5px);
    border-color: #38bdf8;
    box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.3);
}
.feature-card h3 {
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    font-size: 1.3rem !important;
    white-space: nowrap !important; /* Force Single Line */
}
.metric-container {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
    animation: fadeIn 0.5s ease-out;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.status-text { font-size: 2rem; font-weight: 800; margin: 10px 0; text-shadow: 0 0 20px rgba(255,255,255,0.1); }
.status-label { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; }
/* --- Buttons --- */
.stButton button {
    width: 100%;
    border-radius: 12px;
    font-weight: 600;
    height: 50px;
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    transition: all 0.3s;
    box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
}
.stButton button:hover {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    transform: translateY(-2px);
    box-shadow: 0 10px 15px rgba(37, 99, 235, 0.4);
}
/* --- Images --- */
.hero-image {
    border-radius: 24px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    width: 100%;
    object-fit: cover;
    border: 1px solid rgba(255,255,255,0.1);
}
.interactive-logo {
    width: 100%;
    max-width: 120px;
    filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.4));
    /* Continuous Animation Applied Here Too */
    animation: always-moving 6s ease-in-out infinite;
    background: transparent;
    cursor: pointer;
}
.interactive-logo:hover {
    filter: drop-shadow(0 0 25px rgba(56, 189, 248, 0.8));
}
/* Animation Keyframes for Logo */
@keyframes always-moving {
    0% { transform: translateY(0px) rotate(0deg) scale(1); }
    25% { transform: translateY(-5px) rotate(2deg) scale(1.02); }
    50% { transform: translateY(0px) rotate(0deg) scale(1); }
    75% { transform: translateY(5px) rotate(-2deg) scale(1.02); }
    100% { transform: translateY(0px) rotate(0deg) scale(1); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
/* Text Colors */
h1, h2, h3 { color: #f8fafc !important; }
p, li { color: #cbd5e1 !important; }
strong { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. Model & Asset Management
# ==============================================================================
def get_local_logo_base64(file_path="triphazard.png"):
    """Reads local logo or falls back to reliable online 3D logo."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
            b64_data = base64.b64encode(data).decode()
            mime = "image/jpeg" if file_path.lower().endswith(".jpg") else "image/png"
            return f"data:{mime};base64,{b64_data}"
    except: pass
    # Fallback to the reliable online 3D logo
    return "http://googleusercontent.com/image_generation_content/3"

def repair_model_files():
    files = {
        "yolov3-tiny.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    try:
        for filename, url in files.items():
            if os.path.exists(filename): os.remove(filename)
            urllib.request.urlretrieve(url, filename)
        return True
    except: return False

@st.cache_resource
def load_yolo_model():
    cfg, weights, names = "yolov3-tiny.cfg", "yolov3-tiny.weights", "coco.names"
    if not os.path.exists(weights) or os.path.getsize(weights) < 1000000: repair_model_files()
    try:
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        try: output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
        except: output_layers = [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        classes = []
        if os.path.exists(names):
            with open(names, "r") as f: classes = [line.strip() for line in f.readlines()]
        return net, output_layers, classes
    except:
        if repair_model_files(): return load_yolo_model()
        return None, None, None

net, output_layers, classes = load_yolo_model()

# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def text_to_speech_autoplay(text):
    if not AUDIO_AVAILABLE: return
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        st.markdown(f'<audio controls autoplay style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
    except: pass

def detect_hazards_and_zones(image, net, output_layers, classes, conf_threshold, nms_threshold):
    HIGH_RISK_ITEMS = ['sports ball', 'bottle', 'cup', 'wine glass', 'bowl', 'knife', 'spoon', 'fork', 'scissors', 'mouse', 'remote', 'cell phone', 'keyboard', 'book', 'laptop', 'backpack', 'suitcase', 'handbag', 'umbrella', 'teddy bear']
    SAFE_ZONES = ['dining table', 'desk', 'sofa', 'bed', 'cabinet', 'refrigerator', 'shelf']
    
    img = np.array(image.convert('RGB')) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y = int(detection[0]*w_img), int(detection[1]*h_img)
                w, h = int(detection[2]*w_img), int(detection[3]*h_img)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    hazards, safe_zones_found = [], []
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 0, 255) if label in HIGH_RISK_ITEMS else ((0, 255, 0) if label in SAFE_ZONES else (0, 165, 255))
            if label in HIGH_RISK_ITEMS: hazards.append(label)
            elif label in SAFE_ZONES: safe_zones_found.append(label)
            
            # Draw Styled Box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Draw Styled Label Background
            label_text = f"{label.title()}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(img, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), hazards, safe_zones_found, HIGH_RISK_ITEMS

def get_placement_suggestions(hazards, safe_zones, lang_code="English"):
    suggestions = []
    furniture = list(set(safe_zones))
    is_hindi = lang_code == "Hindi"
    
    for item in set(hazards):
        sug = "Clear from floor." if not is_hindi else "फर्श से हटाएं।"
        if item == 'bottle':
            sug = "If water bottle: **Kitchen/Table**. If medicine: **Cabinet**." if not is_hindi else "यदि पानी की बोतल है: **किचन/टेबल**। यदि दवा है: **अलमारी**।"
        elif item in ['cup', 'bowl']:
            sug = "Move to **Kitchen** or **Dining Table**." if not is_hindi else "**किचन** या **डाइनिंग टेबल** पर रखें।"
        elif item in ['book', 'laptop', 'mouse']:
            if 'desk' in furniture: sug = "Place on **Desk**." if not is_hindi else "**डेस्क** पर रखें।"
            else: sug = "Store on shelf." if not is_hindi else "शेल्फ पर रखें।"
        elif item in ['backpack', 'handbag']:
            if 'sofa' in furniture: sug = "Place on **Sofa**." if not is_hindi else "**सोफा** पर रखें।"
            else: sug = "Hang in closet." if not is_hindi else "अलमारी में रखें।"
        suggestions.append(f"🔸 **{item.title()}**: {sug}")
    return suggestions

def generate_report(items, suggestions, risk):
    txt = f"TRIPSAFE AI REPORT\nDate: {time.strftime('%c')}\nStatus: {risk}\n\nITEMS FOUND:\n" + "\n".join([f"- {i}" for i in set(items)])
    txt += "\n\nRECOMMENDED ACTIONS:\n" + "\n".join([s.replace('**','') for s in suggestions])
    return txt

# ==============================================================================
# 3. Sidebar
# ==============================================================================
with st.sidebar:
    # ⬇️ USING triphazard.png
    LOGO_FILENAME = "triphazard.png" 
    
    logo_src = get_local_logo_base64(LOGO_FILENAME)
    # Reduced size to 80px as requested
    st.markdown(f'<div style="text-align: center; margin-bottom: 5px;"><img src="{logo_src}" class="interactive-logo" style="width: 80px; max-width: 80px;"></div>', unsafe_allow_html=True)
    
    lang = st.selectbox("🌐 Language / भाषा", ["English", "Hindi"])
    txt = LANGUAGES[lang]
    
    st.markdown(f"## {txt['guide_title']}")
    st.info(f"{txt['step1']}\n{txt['step2']}\n{txt['step3']}\n{txt['step4']}")
    st.markdown("---")
    st.markdown(txt['tips_title'])
    st.success(f"{txt['daily_tip']}\n\n{random.choice(txt['tips_list'])}")
    
    # --- REPLACING CONTACTS WITH PROJECT TECH INFO ---
    st.markdown("---")
    st.markdown("### 🛠️ Powered By")
    
    st.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span>🧠 Model</span>
<span style="color: #38bdf8; font-weight: bold;">YOLOv3 Tiny</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span>👁️ Vision</span>
<span style="color: #38bdf8; font-weight: bold;">OpenCV</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span>💻 Framework</span>
<span style="color: #38bdf8; font-weight: bold;">Streamlit</span>
</div>
<div style="margin-top: 15px; font-size: 0.8rem; color: #94a3b8; text-align: center;">
Running on <strong>TripSafe Engine v2.1</strong>
</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. Main UI (Navbar Layout: Logo Left, Tabs Right)
# ==============================================================================

# --- HEADER SECTION (New) ---
st.markdown("""
<div class="custom-header">
<h1 style="margin:0; font-size: 1.5rem; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🛡️ TripSafe AI</h1>
<div style="flex-grow:1;"></div>
<span style="color: #94a3b8; font-size: 0.9rem;">Intelligent Indoor Safety System</span>
</div>
""", unsafe_allow_html=True)

# Create 2 Columns: Col 1 for Logo, Col 2 for Tabs
col_logo, col_tabs = st.columns([1, 6])

with col_logo:
    # Logo Display (Fixed to use logo_src)
    logo_src = get_local_logo_base64(LOGO_FILENAME)
    
    st.markdown(f"""
<div style="display: flex; align-items: center; height: 60px;">
<img src="{logo_src}" class="interactive-logo" style="width: 80px;">
</div>
""", unsafe_allow_html=True)

with col_tabs:
    # Optimized Tabs: 4 Main Tabs
    tab_home, tab_scanner, tab_settings, tab_info = st.tabs([
        txt['tab_home'], txt['tab_scanner'], txt['tab_settings'], txt['tab_info']
    ])

# --- TAB 1: HOME ---
with tab_home:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(f"<h1 style='font-size: 2.5rem; margin-bottom: 10px; color: white;'>{txt['home_hero_title']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #cbd5e1; margin-bottom: 30px;'>{txt['home_hero_subtitle']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"### {txt['home_features_title']}")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("""
<div class="feature-card">
<h3>⚡ Real-Time</h3>
<p>Instant hazard detection using advanced YOLO models.</p>
</div>
""", unsafe_allow_html=True)
        with col_f2:
            st.markdown("""
<div class="feature-card">
<h3>🧠 Smart AI</h3>
<p>Context-aware suggestions for a safer home.</p>
</div>
""", unsafe_allow_html=True)
            
    with c2:
        # High Quality Home Image
        st.markdown('<img src="https://images.pexels.com/photos/1643383/pexels-photo-1643383.jpeg?auto=compress&cs=tinysrgb&w=800" class="hero-image">', unsafe_allow_html=True)

# --- TAB 2: SCANNER ---
with tab_scanner:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(txt['input_source'])
        src = st.radio(txt['select'], [txt['upload'], txt['camera']], label_visibility="collapsed")
        img_file = st.file_uploader(txt['upload'], type=['jpg','png']) if src == txt['upload'] else st.camera_input(txt['camera'])

    with c2:
        if img_file and net:
            img = Image.open(img_file)
            with st.spinner("Scanning..."):
                time.sleep(0.5) 
                res_img, hazards, zones, risk_list = detect_hazards_and_zones(img, net, output_layers, classes, 0.25, 0.4)
            
            st.image(res_img, caption="AI Analysis Result", use_container_width=True)
            
            risk_count = sum(1 for i in hazards if i in risk_list)
            if risk_count > 0:
                status, color, msg = txt['high_risk'], "#fc8181", txt['high_risk_msg'].format(count=risk_count)
            elif hazards:
                status, color, msg = txt['caution'], "#f6e05e", txt['caution_msg']
            else:
                status, color, msg = txt['safe'], "#68d391", txt['safe_msg']
            
            if "audio_on" not in st.session_state: st.session_state.audio_on = True
            if st.session_state.audio_on and status != txt['safe']: text_to_speech_autoplay(msg)
            
            # Metrics
            st.markdown(f"""
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 20px;">
<div class="metric-container" style="border-bottom: 4px solid {color};">
<div class="status-label">{txt['status']}</div>
<div class="status-text" style="color: {color};">{status}</div>
</div>
<div class="metric-container">
<div class="status-label">{txt['hazards']}</div>
<div class="status-text">{len(hazards)}</div>
</div>
<div class="metric-container">
<div class="status-label">{txt['safe_zones']}</div>
<div class="status-text">{len(zones)}</div>
</div>
</div>
""", unsafe_allow_html=True)
            
            sugs = get_placement_suggestions(hazards, zones, lang)
            if sugs:
                st.markdown(f"""
<div style="background: rgba(6, 182, 212, 0.1); border-left: 4px solid #06b6d4; padding: 20px; border-radius: 12px; margin-top: 20px;">
<h4 style="margin-top:0;">{txt["suggestions"]}</h4>
{'<br>'.join(sugs)}
</div>
""", unsafe_allow_html=True)
            
            st.download_button(txt['download_report'], generate_report(hazards, sugs, status), "report.txt")

# --- TAB 3: SETTINGS ---
with tab_settings:
    st.markdown(txt['settings_config'])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(txt['sensitivity'])
        st.slider("Confidence", 0.0, 1.0, 0.25, key="conf")
        st.slider("NMS Threshold", 0.0, 1.0, 0.4, key="nms")
    with c2:
        st.markdown(txt['alerts'])
        st.toggle(txt['enable_audio'], value=True, key="audio_on")

# --- TAB 4: INFO & SUPPORT (Merged) ---
with tab_info:
    # 1. About Section (Full Width)
    st.markdown(f"### {txt['about_title']}")
    st.info(txt['about_text'])
    
    st.markdown("---")
    
    # 2. Contact Section
    st.markdown(f"### {txt['contact_title']}")
    
    # Lead Developer (Prominent Card)
    st.markdown(f"""
<div class="feature-card" style="margin-bottom: 20px; padding: 20px; text-align: center;">
<h3 style="margin:0; font-size: 1.4rem;">👤 {txt['contact_name']}</h3>
<p style="color: #38bdf8; font-weight: bold; margin: 5px 0;">{txt['contact_role']}</p>
<p style="margin:0;">📧 <a href="mailto:{txt['contact_email']}" style="color: white;">{txt['contact_email']}</a></p>
</div>
""", unsafe_allow_html=True)

    # Team Members (Anuradha & Nisha) - SIDE BY SIDE
    team_members = [
        {"name": "Anuradha Singh", "email": "anuradhakipm@gmail.com"},
        {"name": "Nisha Maddhesia", "email": "nshmddsh@gmail.com"}
    ]
    
    # Use st.columns to put them in one row
    t_col1, t_col2 = st.columns(2)
    
    for i, member in enumerate(team_members):
        col = t_col1 if i == 0 else t_col2
        with col:
            st.markdown(f"""
<div class="feature-card" style="margin-bottom: 10px; padding: 15px; text-align: center;">
<h3 style="margin:0; font-size: 1.1rem;">👤 {member['name']}</h3>
<p style="color: #94a3b8; margin: 5px 0;">Team Member</p>
<p style="margin:0; font-size: 0.9rem;">📧 <a href="mailto:{member['email']}" style="color: #38bdf8;">{member['email']}</a></p>
</div>
""", unsafe_allow_html=True)

# --- FOOTER SECTION (New) ---
st.markdown("""
<div class="custom-footer">
<p>© 2025 TripSafe AI. All Rights Reserved.</p>
<p>Developed with ❤️ by <span style="color: #38bdf8;">Team TripSafe</span></p>
<div style="margin-top: 10px; font-size: 0.9rem;">
<!-- Privacy Policy Dropdown -->
<details>
<summary>📄 Privacy Policy</summary>
<div>
<strong style="color: #38bdf8;">Data & Privacy</strong><br>
<small>
1. <strong>Local Processing:</strong> All video feeds are processed locally. No upload.<br>
2. <strong>Usage:</strong> Camera access is strictly for hazard detection.<br>
3. <strong>Consent:</strong> By using this app, you agree to camera usage.
</small>
</div>
</details>
<span style="color: #64748b;">•</span>
<!-- Terms of Use Dropdown -->
<details>
<summary>⚖️ Terms of Use</summary>
<div>
<strong style="color: #38bdf8;">Disclaimer</strong><br>
<small>
1. <strong>Education:</strong> This is an assistive tool, not for critical safety.<br>
2. <strong>Accuracy:</strong> Depends on lighting. Verify manually.<br>
3. <strong>Liability:</strong> Developers are not liable for errors.
</small>
</div>
</details>
<span style="color: #64748b;">•</span>
<a href="https://github.com/GudiyaVishwakarma-7080/TripSafe-AI" target="_blank">🐱 GitHub</a>
</div>
</div>
""", unsafe_allow_html=True)
