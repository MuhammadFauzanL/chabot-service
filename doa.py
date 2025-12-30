"""
=================================================================
COMPLETE ISLAMIC CHATBOT BACKEND - FIXED & IMPROVED
=================================================================
Features:
- Smart keyword matching (better than TF-IDF for Indonesian)
- Intent detection with scoring
- Deduplication by ID
- Context-aware responses
- Multiple result types (doa/hadis)
- Helpful suggestions
- Time-based greetings using location (via Aladhan API for timezone)
- Limited to 3 results per query (user-friendly)
- Offline-safe with proper timeout handling
=================================================================
"""

import json, os, re
import requests
from datetime import datetime
import pytz
from flask import Flask, request, jsonify
from flask_cors import CORS
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
CORS(app)

DOA_FILE = "doa_dataset.json"
HADIS_FILE = "hadist_dataset.json"
INTENT_FILE = "intent_rules.json"

stemmer = StemmerFactory().create_stemmer()

# ================= LOAD DATA =================
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

DOA_DATA = load_json(DOA_FILE)
HADIS_DATA = load_json(HADIS_FILE)
INTENTS = load_json(INTENT_FILE)

# ================= STOPWORDS (Indonesian) =================
STOPWORDS = {
    'yang', 'untuk', 'pada', 'adalah', 'dari', 'di', 'ke', 'dalam',
    'dengan', 'oleh', 'ini', 'itu', 'dan', 'atau', 'akan', 'ada',
    'bisa', 'dapat', 'juga', 'lebih', 'saat', 'ketika', 'tentang',
    'apa', 'siapa', 'mana', 'bagaimana', 'kenapa', 'kapan', 'dimana'
}

# ================= NLP FUNCTIONS =================
def normalize(text):
    """Normalize text: lowercase, remove extra spaces"""
    return re.sub(r"\s+", " ", text.lower().strip())

def preprocess(text):
    """Clean and stem text"""
    # Remove special chars, keep letters and numbers
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    # Stem
    text = stemmer.stem(text)
    return text

def get_keywords(text):
    """Extract meaningful keywords from text"""
    words = preprocess(text).split()
    # Remove stopwords and single characters
    keywords = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    return keywords

# ================= SMART MATCHING =================
def calculate_match_score(query_keywords, target_text, target_keywords=None):
    """
    Calculate match score using multiple signals:
    1. Word overlap in main text
    2. Keyword match bonus
    3. Exact phrase match bonus
    4. Partial match consideration
    """
    target_words = set(preprocess(target_text).split())
    query_set = set(query_keywords)
    
    # Base score: word overlap
    overlap = len(query_set & target_words)
    if len(query_set) == 0:
        return 0.0
    
    base_score = overlap / len(query_set)
    
    # Bonus for keyword matches
    keyword_bonus = 0.0
    if target_keywords:
        target_kw_set = set()
        for kw in target_keywords:
            target_kw_set.update(get_keywords(kw))
        
        kw_overlap = len(query_set & target_kw_set)
        keyword_bonus = kw_overlap * 0.2  # Each keyword match adds 0.2
    
    # Bonus for exact phrase match
    phrase_bonus = 0.0
    original_query = " ".join(query_keywords)
    if original_query in preprocess(target_text):
        phrase_bonus = 0.3
    
    # Partial word match bonus (for words like "sabar" matching "kesabaran")
    partial_bonus = 0.0
    for qw in query_set:
        for tw in target_words:
            if len(qw) >= 4 and len(tw) >= 4:  # Only for words 4+ chars
                if qw in tw or tw in qw:
                    partial_bonus += 0.1
                    break
    
    total_score = min(base_score + keyword_bonus + phrase_bonus + partial_bonus, 1.0)
    return round(total_score, 3)

# ================= INTENT DETECTION =================
def detect_intent(query):
    """
    Detect user intent with confidence scoring
    Returns: dict with intent info or None
    """
    query_lower = query.lower()
    query_keywords = set(get_keywords(query))
    
    best_intent = None
    best_score = 0
    
    for intent_name, intent_data in INTENTS.items():
        score = 0
        matched_kws = []
        
        for keyword in intent_data["keywords"]:
            kw_lower = keyword.lower()
            
            # Exact match in original query (highest priority)
            if kw_lower in query_lower:
                score += 1.0
                matched_kws.append(keyword)
            # Stemmed keyword match
            elif any(kw_word in query_keywords for kw_word in get_keywords(keyword)):
                score += 0.5
                matched_kws.append(keyword)
        
        if score > best_score:
            best_score = score
            best_intent = {
                "name": intent_name,
                "type": intent_data["type"],
                "canonical_query": intent_data["canonical_query"],
                "score": score,
                "matched_keywords": matched_kws
            }
    
    # Return intent only if confidence is high enough
    if best_score >= 0.5:
        return best_intent
    return None

# ================= SEARCH FUNCTIONS =================
def search_doa(query_keywords, top_k=10):
    """Search doa dataset"""
    results = []
    
    for doa in DOA_DATA:
        # Combine searchable fields
        searchable = f"{doa['judul']} {doa['arti']} {doa.get('latin', '')}"
        
        score = calculate_match_score(query_keywords, searchable)
        
        if score > 0.1:  # Low threshold to catch more results
            results.append({
                "score": float(score),
                "data": {**doa, "source_type": "doa"}
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

def search_hadis(query_keywords, top_k=10):
    """Search hadis dataset"""
    results = []
    
    for hadis in HADIS_DATA:
        # Combine searchable fields
        searchable = f"{hadis['tema']} {hadis['arti']}"
        kata_kunci = hadis.get('kata_kunci', [])
        
        score = calculate_match_score(query_keywords, searchable, kata_kunci)
        
        if score > 0.1:
            results.append({
                "score": float(score),
                "data": {**hadis, "source_type": "hadis"}
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# ================= DEDUPLICATION =================
def deduplicate_by_id(results):
    """Remove duplicates based on ID field"""
    seen_ids = set()
    unique = []
    
    for result in results:
        item_id = result["data"].get("id")
        if item_id and item_id not in seen_ids:
            seen_ids.add(item_id)
            unique.append(result)
    
    return unique

# ================= RESPONSE FORMATTING =================
def format_response(results, query, intent=None):
    """Format chatbot response with helpful context"""
    
    if not results:
        return {
            "status": "ASK",
            "message": "Maaf, belum menemukan hasil yang sesuai 😔\n\nCoba dengan kata kunci lain ya!",
            "suggestions": [
                "Gunakan kata kunci lebih spesifik",
                "Contoh: 'doa sebelum makan', 'hadis tentang sabar'"
            ],
            "examples": [
                "doa keluar rumah",
                "hadis tentang ilmu",
                "doa untuk orang sakit"
            ]
        }
    
    # Separate by type
    doa_results = [r for r in results if r["data"]["source_type"] == "doa"]
    hadis_results = [r for r in results if r["data"]["source_type"] == "hadis"]
    
    total = len(results)
    doa_count = len(doa_results)
    hadis_count = len(hadis_results)
    
    # Create message with indication of limited results
    if intent:
        message = f"✨ Ditemukan untuk '{intent['name']}'"
    else:
        if doa_results and hadis_results:
            message = f"✨ Ditemukan {doa_count} doa dan {hadis_count} hadis"
        elif doa_results:
            message = f"✨ Ditemukan {doa_count} doa"
        else:
            message = f"✨ Ditemukan {hadis_count} hadis"
    
    # Add info if there are more results
    if total > 3:
        message += f" (menampilkan 3 teratas dari {total} hasil)"
    
    return {
        "status": "OK",
        "message": message,
        "data": results[:3],  # ✅ LIMIT TO 3 RESULTS
        "summary": {
            "total": total,
            "doa_count": doa_count,
            "hadis_count": hadis_count,
            "showing": min(3, total)  # ✅ SHOW MAX 3
        }
    }

# ================= TIME-BASED GREETING (OFFLINE-SAFE) =================
def get_time_based_greeting(lat=None, lng=None):
    """Get greeting based on current time in user's location"""
    timezone_name = 'Asia/Jakarta'  # Default to Jakarta
    
    # ✅ OFFLINE-SAFE: Try to get timezone, but don't block if it fails
    if lat is not None and lng is not None:
        try:
            url = f'https://api.aladhan.com/v1/timings?latitude={lat}&longitude={lng}&method=11'
            response = requests.get(url, timeout=3)  # ✅ SHORT TIMEOUT
            if response.status_code == 200:
                data = response.json()
                timezone_name = data['data']['meta']['timezone']
        except Exception as e:
            print(f"⚠️ Timezone fetch failed (offline?): {str(e)}")
            # Fallback to default - no problem!
    
    try:
        tz = pytz.timezone(timezone_name)
        current_hour = datetime.now(tz).hour
        
        if 5 <= current_hour < 11:
            return "Selamat pagi"
        elif 11 <= current_hour < 15:
            return "Selamat siang"
        elif 15 <= current_hour < 18:
            return "Selamat sore"
        else:
            return "Selamat malam"
    except Exception as e:
        print(f"⚠️ Time greeting failed: {str(e)}")
        return "Selamat datang"

# ================= MAIN CHAT ENDPOINT =================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        query = normalize(data.get("query", ""))
        lat = data.get("lat")
        lng = data.get("lng")
        
        # Empty query
        if not query:
            return jsonify({
                "status": "ASK",
                "message": "Silakan ketik kebutuhan doa atau hadis 😊",
                "examples": [
                    "doa sebelum makan",
                    "hadis tentang sabar",
                    "doa naik kendaraan"
                ]
            })
        
        # Greetings
        greetings = ["halo", "hai", "mulai", "assalamualaikum", "salam", "test", "tes"]
        if query in greetings:
            time_greeting = get_time_based_greeting(lat, lng)
            salam = "Wa'alaikumsalam" if query in ["assalamualaikum", "salam"] else "Halo"
            return jsonify({
                "status": "ASK",
                "message": f"{salam}, {time_greeting} 😊\n\nSaya Asisten Islami. Silakan tanyakan doa atau hadis yang kamu butuhkan!",
                "examples": [
                    "doa keluar rumah",
                    "hadis tentang ilmu",
                    "doa ketika sakit"
                ]
            })
        
        # Extract keywords
        query_keywords = get_keywords(query)
        
        if not query_keywords:
            return jsonify({
                "status": "ASK",
                "message": "Kata kunci terlalu umum. Coba lebih spesifik ya! 😊",
                "examples": [
                    "doa sebelum tidur",
                    "hadis tentang akhlak",
                    "doa memohon rezeki"
                ]
            })
        
        # Detect intent
        intent = detect_intent(query)
        
        results = []
        
        # Intent-based search (if strong intent detected)
        if intent and intent["score"] >= 1.0:
            canonical_keywords = get_keywords(intent["canonical_query"])
            
            if intent["type"] == "doa":
                results = search_doa(canonical_keywords)
                # Add some hadis too if results are few
                if len(results) < 3:
                    results.extend(search_hadis(query_keywords))
            else:
                results = search_hadis(canonical_keywords)
                # Add some doa too if results are few
                if len(results) < 3:
                    results.extend(search_doa(query_keywords))
        
        # General search (no strong intent or need more results)
        if not results or len(results) < 2:
            doa_results = search_doa(query_keywords)
            hadis_results = search_hadis(query_keywords)
            results = doa_results + hadis_results
        
        # Deduplicate and re-sort
        results = deduplicate_by_id(results)
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return jsonify(format_response(results, query, intent))
    
    except Exception as e:
        print(f"❌ Error in /chat: {str(e)}")
        return jsonify({
            "status": "ERROR",
            "message": "Maaf, terjadi kesalahan server 😔"
        }), 500

# ================= SUGGESTION ENDPOINT =================
@app.route("/suggest", methods=["GET"])
def suggest():
    """Get popular suggestions"""
    return jsonify({
        "status": "OK",
        "categories": {
            "doa_populer": [
                "Doa Sebelum Makan",
                "Doa Naik Kendaraan",
                "Doa Sebelum Tidur",
                "Doa Ketika Sakit",
                "Doa Keluar Rumah"
            ],
            "hadis_populer": [
                "Hadis tentang Akhlak",
                "Hadis tentang Ilmu",
                "Hadis tentang Sabar",
                "Hadis tentang Sosial",
                "Hadis tentang Ibadah"
            ]
        },
        "quick_searches": [
            "doa pagi",
            "doa malam",
            "hadis sabar",
            "doa rezeki",
            "hadis berbuat baik"
        ]
    })

# ================= BROWSE ENDPOINT =================
@app.route("/browse/<category>", methods=["GET"])
def browse(category):
    """Browse doa or hadis by category"""
    try:
        if category == "doa":
            return jsonify({
                "status": "OK",
                "category": "doa",
                "total": len(DOA_DATA),
                "data": [{"data": {**d, "source_type": "doa"}, "score": 1.0} for d in DOA_DATA]
            })
        elif category == "hadis":
            return jsonify({
                "status": "OK",
                "category": "hadis",
                "total": len(HADIS_DATA),
                "data": [{"data": {**h, "source_type": "hadis"}, "score": 1.0} for h in HADIS_DATA]
            })
        else:
            return jsonify({
                "status": "ERROR",
                "message": "Kategori tidak valid. Gunakan 'doa' atau 'hadis'."
            }), 400
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 500

# ================= HEALTH CHECK =================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "service": "Islamic Chatbot API",
        "version": "2.1",
        "data_stats": {
            "doa_count": len(DOA_DATA),
            "hadis_count": len(HADIS_DATA),
            "intents": list(INTENTS.keys())
        },
        "features": {
            "max_results_per_query": 3,
            "offline_safe": True,
            "timeout_seconds": 3
        }
    })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Islamic Chatbot Backend v2.1")
    print(f"📍 Running on port {port}")
    print(f"📚 Data loaded: {len(DOA_DATA)} doa, {len(HADIS_DATA)} hadis")
    print(f"🎯 Max results per query: 3")
    print(f"✅ Offline-safe mode: enabled")
    app.run(host="0.0.0.0", port=port)
