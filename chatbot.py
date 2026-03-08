import json
import sys
import os
import uuid
import requests # 🔴 ใช้ตัวนี้ยิง API ออกนอกเซิร์ฟเวอร์
from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity 
import mysql.connector  
import json   
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

# ==========================================
# ⚙️ 1. ตั้งค่าระบบฐานข้อมูล (XAMPP / MySQL)
# ==========================================
print("🔍 [ระบบสืบสวน] กำลังสแกนตัวแปรทั้งหมดที่ Railway ส่งมาให้...")

# กวาดชื่อตัวแปรทั้งหมดที่มีคำว่า "MYSQL" หรือ "DB" มาปริ้นต์ดู (เอาแค่ชื่อ ไม่เอารหัสผ่าน เพื่อความปลอดภัย)
available_vars = [key for key in os.environ.keys() if "MYSQL" in key.upper() or "DB" in key.upper()]
print(f"📋 รายชื่อตัวแปรที่พบในระบบ: {available_vars}")

# ดึงข้อมูล โดยดักจับทั้งชื่อแบบไม่มีขีด และแบบมีขีด (รองรับหลายมาตรฐาน)
DB_HOST = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST")
port_env = os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT")
DB_PORT = int(port_env) if port_env and port_env.isdigit() else 3306 
DB_NAME = os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE")
DB_USER = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER")
DB_PASS = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_ROOT_PASSWORD")

# 🛑 ดักจับ Error หากยังหาไม่เจออีก
if not DB_HOST:
    print("❌ Error: API หาตัวแปร Host ไม่เจอเลย แม้จะลองสแกนดูแล้วก็ตาม!")
    print("👉 หาก 'รายชื่อตัวแปรที่พบ' ข้างบนว่างเปล่า ([]) แปลว่า Railway ไม่ได้ส่งอะไรมาเลยจริงๆ ต้องเช็คการตั้งค่า Reference ใหม่ครับ")
    sys.exit(1) 

print(f"✅ ข้อมูลฐานข้อมูลพร้อม! เตรียมเชื่อมต่อ {DB_HOST}:{DB_PORT}")
# ==========================================
# 🔑 2. ตั้งค่า Hugging Face API (สมอง AI)
# ==========================================
#  1. เอา API Key จากเว็บ Hugging Face มาใส่ตรงนี้ (ขึ้นต้นด้วย hf_...)
HF_TOKEN = os.getenv("HF_TOKEN") # ใส่ Hugging Face API Token ของคุณที่นี่
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "typhoon-ai/llama3.1-typhoon2-8b-instruct:featherless-ai"

def call_huggingface_llm(prompt_text):

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        print("Status:", response.status_code)
        print("Text:", response.text[:300])

        if response.status_code != 200:
            return ""

        result = response.json()

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("❌ HF Request Failed:", e)
        return ""

# ==========================================
# 📂 3. ตั้งค่า Path ไฟล์ข้อมูลความรู้ (RAG)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "clean_data")
GEMINI_DATA_FILE = os.path.join(BASE_DIR, "train_iot_premium.json")
VECTOR_FILE = os.path.join(BASE_DIR, "knowledge_vectors.npy")
TEXT_FILE = os.path.join(BASE_DIR, "knowledge_texts.json")

def generate_session_id():
    return str(uuid.uuid4())

def init_db():
    print("🛠️ กำลังตรวจสอบและสร้างตารางฐานข้อมูล...")
    try:
        # เชื่อมต่อครั้งเดียว โดยระบุพอร์ตให้ชัดเจน
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            port=DB_PORT, 
            charset="utf8mb4"
        )
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255),
                        role VARCHAR(50),
                        source VARCHAR(50),
                        content TEXT,
                        category VARCHAR(100),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     )''')
        conn.commit()
        conn.close()
        print("✅ ฐานข้อมูลและตารางพร้อมใช้งาน!")
    except Exception as e:
        print(f"❌ DB Init Failed: {e}")

def categorize_question(user_message):
    categories = {
        "หลักสูตร": ["หลักสูตร", "เรียนกี่ปี", "สาขา", "วุฒิ", "ปริญญา"],
        "อาจารย์": ["อาจารย์", "อ.", "ดร.", "หัวหน้าภาค", "ผู้สอน"],
        "รายวิชา": ["วิชา", "เรียนอะไร", "หน่วยกิต", "แคลคูลัส", "ฟิสิกส์", "โปรแกรม"],
        "การรับสมัคร": ["tcas", "รับสมัคร", "โควตา", "สอบเข้า", "พอร์ต"],
        "ค่าใช้จ่าย": ["ค่าเทอม", "กู้", "กยศ", "ทุน", "จ่าย"]
    }
    text = user_message.lower()
    for cat_name, keywords in categories.items():
        if any(kw in text for kw in keywords):
            return cat_name
    return "ทั่วไป/ไม่ทราบ"

def save_message_mysql(session_id, role, source, content, category="-"):
    try:
        conn = mysql.connector.connect(host=DB_HOST,user=DB_USER,password=DB_PASS,database=DB_NAME,port=DB_PORT,charset="utf8mb4")
        c = conn.cursor()

        c.execute("""
        INSERT INTO chat_history 
        (session_id, role, source, content, category)
        VALUES (%s,%s,%s,%s,%s)
        """, (session_id, role, source, content, category))

        conn.commit()
        conn.close()

        print("💾 Save:", role, content[:50])

    except Exception as e:
        print(f"⚠️ บันทึก DB พลาด: {e}")

# ==========================================
# 📚 4. โหลด Vector Database (RAG) เข้า RAM
# ==========================================
print("กำลังโหลดโมเดล Vector (MiniLM)... ⏳")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

knowledge_base = []
seen_contents = set()
knowledge_vectors = None

VECTOR_FILE = os.path.join(BASE_DIR, "knowledge_vectors.npy")
TEXT_FILE = os.path.join(BASE_DIR, "knowledge_texts.json")

def add_to_knowledge(question, answer):

    content = f"คำถาม: {question} คำตอบ: {answer}"
    content = content.strip()

    if content and content not in seen_contents:
        seen_contents.add(content)
        knowledge_base.append(content)


# ======================================
# ⚡ โหลด Vector DB ถ้ามีอยู่แล้ว
# ======================================

if os.path.exists(VECTOR_FILE) and os.path.exists(TEXT_FILE):

    print("⚡ โหลด Vector Database จากไฟล์ (เร็วมาก)")

    knowledge_vectors = np.load(VECTOR_FILE)

    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    print(f"✅ โหลดข้อมูล {len(knowledge_base)} ก้อน")

else:

    print("📄 กำลังโหลดไฟล์ข้อมูล...")

    if os.path.exists(CLEAN_DATA_DIR):
        for filename in os.listdir(CLEAN_DATA_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(CLEAN_DATA_DIR, filename), 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                         question = item.get("question", "")
                         answer = item.get("output", "")
                         add_to_knowledge(question, answer)

    if os.path.exists(GEMINI_DATA_FILE):
        print("📄 โหลดไฟล์ข้อมูลสังเคราะห์...")
        with open(GEMINI_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                     question = item.get("question", "")
                     answer = item.get("output", "")
                     add_to_knowledge(question, answer)

    print(f"🎉 รวบรวมข้อมูล {len(knowledge_base)} ก้อน")

    print("🧠 กำลังสร้าง Vector Database...")

    knowledge_vectors = embedder.encode(
        knowledge_base,
        batch_size=32,
        show_progress_bar=True
    )

    np.save(VECTOR_FILE, knowledge_vectors)

    with open(TEXT_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False)

    print("💾 บันทึก Vector Database เรียบร้อย")

print("🚀 ระบบ Vector Search พร้อมใช้งาน!\n")

# ==========================================
# 🤖 5. ฟังก์ชันเตรียม AI & กฎเหล็ก
# ==========================================
ADMISSION_KEYWORDS = ["รับสมัคร", "tcas", "เกณฑ์", "คะแนน", "โควตา", "จำนวนรับ", "สมัครเรียน"]
BAD_WORDS = ["ควย", "เฮี้ย", "กู", "เหี้ย", "สัส", "แม่ง", "เสียว"]
FIXED_ADMISSION_REPLY = "ขอให้ติดตามข่าวสารและประกาศรับสมัครล่าสุดได้ที่เว็บไซต์ของภาควิชาฯ (www.iote.kmitl.ac.th) ครับ"
BAD_WORD_REPLY = "ขอโทษครับ ผมเป็นคนดีเกินไปที่จะตอบคำถามแบบนั้นได้"

def is_admission_question(text):
    return any(k in text.lower() for k in ADMISSION_KEYWORDS)

def contains_bad_words(text):
    return any(b in text.lower() for b in BAD_WORDS)

def build_instruction_rules():
    return (
        "กฎการตอบคำถาม:\n"
        "1. คุณคือผู้ช่วย AI ของภาควิชาวิศวกรรมระบบไอโอทีและสารสนเทศ สจล.\n"
        "2. ใช้ข้อมูลใน 'บริบทข้อมูล' เป็นหลัก และสามารถสรุปหรือวิเคราะห์ต่อได้ "
        "แต่ห้ามสร้างข้อมูลใหม่ที่ไม่มีอยู่ในบริบท\n"
        "3. อธิบายเชื่อมโยงกับบริบทของภาควิชาในเชิงภาพรวมเท่านั้น\n"
        "4. หากบริบทข้อมูลระบุว่าไม่พบข้อมูล ให้ตอบอย่างสุภาพว่า "
        "'ขออภัยครับ ข้อมูลส่วนนี้ยังไม่ได้ระบุไว้ในระบบของผมครับ'\n"
        "5. ห้ามตอบว่า 'นอกเรื่อง' หรือปัดคำถามทิ้ง\n"
        "6. ความยาวคำตอบ 15–50 คำ ถ้าข้อมูลเยอะให้สรุป\n"
        "7. ภาษาธรรมชาติ ผิดเล็กน้อยได้ ไม่ต้องเป็นทางการเกินไป\n"
        "8. ตอบเป็นภาษาไทย ห้ามใช้จุด (.) และห้ามเว้นวรรคแปลก ๆ ระหว่างประโยค\n"
        "9.ห้ามใช้คำสรรพนามลอยๆ (เช่น เขา, อาจารย์ท่านนี้, ความเชี่ยวชาญนี้) ให้ระบุ ชื่อ-นามสกุล หรือชื่อวิชา ลงไปในคำถามและคำตอบทุกครั้ง"
        "10.ตอบคำถามให้เข้าใจง่าย และถ้าเป็นคำถามเกี่ยวกับอาชีพให้ระบุตำแหน่งงานเป็นรายการ เช่น IoT Engineer,Embedded Engineer,programmer,Data Analyst เป็นต้น และ ถ้าไม่เจาะจงจงตอบสายงานเกี่ยวกับไอโอทีและสารสนเทศเป็นหลัก")


def expand_query_with_llm(user_query):
    print(f"\n🧠 [API 1] กำลังให้ Hugging Face ช่วยขยายคำถาม: '{user_query}'...")
    prompt = f"""หน้าที่ของคุณคือวิเคราะห์คำถามของผู้ใช้ และเขียนใหม่ให้เป็น 'ประโยคค้นหาที่สมบูรณ์' (Search Sentence)
    กฎเหล็ก:
    1. เขียนเป็นประโยคยาวๆ ที่อ่านเป็นธรรมชาติ ห้ามคั่นด้วยลูกน้ำ (,) เด็ดขาด
    2. ห้ามใช้คำว่า "หลักสูตร" โดดๆ ให้ใช้คำว่า "เนื้อหาการเรียน" หรือ "รายวิชา" แทน (เพื่อป้องกันการไปค้นเจอตำแหน่ง 'อาจารย์ประจำหลักสูตร')
    3. ห้ามเติมคำศัพท์เฉพาะทางเทคโนโลยีที่ผู้ใช้ไม่ได้ถาม
    4. [กฎเหล็กสำคัญ] ถ้าผู้ใช้ถามเรื่อง 'การรับสมัคร', 'TCAS', 'เกณฑ์คะแนน', 'โควตา' หรือ 'จำนวนรับ' ให้คุณตอบแค่ประโยคนี้เท่านั้น: 
    "'ขอให้ติดตามข่าวสารและประกาศรับสมัครล่าสุดได้ที่เว็บไซต์ของภาควิชาฯ (www.iote.kmitl.ac.th) ครับ' "
    "โดยห้ามอธิบายเรื่องเกณฑ์คะแนนหรือจำนวนคนรับเด็ดขาด เพื่อป้องกันความผิดพลาดของข้อมูล"
    
    ตัวอย่างที่ 1:
    คำถาม: ภาคไอโอทีอยู่ที่ไหน
    ประโยคค้นหา: ข้อมูลสถานที่ตั้ง อาคารที่อยู่ และช่องทางการติดต่อของภาควิชาวิศวกรรมระบบไอโอทีและสารสนเทศ
    
    ตัวอย่างที่ 2:
    คำถาม: สาขานี้เรียนเกี่ยวกับอะไร
    ประโยคค้นหา: ข้อมูลเนื้อหาการเรียนการสอน รายวิชาที่เรียนเกี่ยวกับการพัฒนาซอฟต์แวร์ ฮาร์ดแวร์ และระบบไอโอที
    
    คำถาม: {user_query}
    ประโยคค้นหา:"""
    expanded = call_huggingface_llm(prompt)
    if expanded:
        expanded = expanded.replace("ประโยคค้นหา:", "").strip()
        print(f"✨ ประโยคที่ใช้ค้นหาจริง: {expanded}")
        return expanded
    return user_query

def get_semantic_knowledge(user_query):
    if not knowledge_base or knowledge_vectors is None:
        return "ไม่พบข้อมูลในระบบ"
    
    smart_query = expand_query_with_llm(user_query)
    query_vector = embedder.encode([smart_query])
    similarities = cosine_similarity(query_vector, knowledge_vectors)[0]
    
    top_3_indices = np.argsort(similarities)[::-1][:3]
    knowledge_pieces = [knowledge_base[idx] for idx in top_3_indices if similarities[idx] > 0.75]
    
    return "ข้อมูลที่เกี่ยวข้องพบดังนี้:\n" + "\n---\n".join(knowledge_pieces) if knowledge_pieces else "ไม่พบข้อมูลที่เกี่ยวข้อง"

# ==========================================================
# 🌐 6. จุดรับคำสั่งจากผู้ใช้ (Flask API)
# ==========================================================
@app.route('/')
@app.route('/chatbot.html')
def serve_html():
    return send_file('chatbot.html')
@app.route('/ask', methods=['POST'])
def ask_ollama():
    try:
        data = request.json
        user_message = data.get("question", "").strip()
        
        session_id = data.get("session_id")
        if not session_id or session_id == "default":
            session_id = generate_session_id()

        category = categorize_question(user_message)
        save_message_mysql(session_id, "user", "user", user_message, category)
        
        answer = ""

        # 🔒 Hard rule
        if contains_bad_words(user_message):
            answer = BAD_WORD_REPLY
        elif is_admission_question(user_message):
            answer = FIXED_ADMISSION_REPLY
        else:
            knowledge = get_semantic_knowledge(user_message)
            final_prompt = (
                f"{build_instruction_rules()}\n\n"
                f"บริบทข้อมูล:\n{knowledge}\n\n"
                f"คำถามผู้ใช้: {user_message}\nคำตอบ:"
            )

            print("🧠 [API 2] กำลังส่งข้อมูลให้ Hugging Face คิดคำตอบสุดท้าย...")
            ai_response = call_huggingface_llm(final_prompt)
            
            # ถ้า API มีปัญหา (เช่น rate limit หรือ timeout) ให้ตอบกลับแบบปลอดภัย
            answer = ai_response if ai_response else "ขออภัยครับ ตอนนี้เซิร์ฟเวอร์ AI ของเราตอบสนองช้าชั่วคราว ลองพิมพ์ถามใหม่อีกครั้งนะครับ"

        save_message_mysql(session_id, "assistant", "ai", answer, category)
        return jsonify({"answer": answer, "session_id": session_id})
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

init_db() 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False,threaded=True)





















