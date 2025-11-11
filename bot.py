import os
import json
import hashlib
import re
import zipfile
import time
import random
from datetime import datetime
from collections import defaultdict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from google.cloud import texttospeech
from google.oauth2 import service_account
import asyncio
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment variables")
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in environment variables")
if not GOOGLE_CREDENTIALS_JSON:
    raise ValueError("Missing GOOGLE_CREDENTIALS_JSON in environment variables")

class Config:
    MAX_TOPIC_LENGTH = 100
    MAX_VOCAB_ITEMS = 15
    TTS_TIMEOUT = 30
    API_RETRY_ATTEMPTS = 3
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 3600
    MAX_FILE_SIZE = 50 * 1024 * 1024

config = Config()

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

class RateLimiter:
    def __init__(self, max_requests=5, window=3600):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.window = window
    
    def is_allowed(self, user_id):
        now = time.time()
        user_requests = self.requests[user_id]
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < self.window]
        if len(user_requests) >= self.max_requests:
            return False
        user_requests.append(now)
        return True
    
    def get_reset_time(self, user_id):
        if not self.requests[user_id]:
            return 0
        oldest_request = min(self.requests[user_id])
        reset_time = oldest_request + self.window - time.time()
        return max(0, int(reset_time))

rate_limiter = RateLimiter(
    max_requests=config.RATE_LIMIT_REQUESTS,
    window=config.RATE_LIMIT_WINDOW
)

def get_google_tts_client():
    credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return texttospeech.TextToSpeechClient(credentials=credentials)

def validate_topic(topic):
    topic = re.sub(r'\s+', ' ', topic.strip())
    if re.search(r'[<>"|&;`$()]', topic):
        raise ValueError("Topic contains invalid characters")
    inappropriate_patterns = [r'\b(porn|sex|violence|hate|kill|death)\b']
    for pattern in inappropriate_patterns:
        if re.search(pattern, topic, re.IGNORECASE):
            raise ValueError("Topic contains inappropriate content")
    if len(topic) > config.MAX_TOPIC_LENGTH:
        topic = topic[:config.MAX_TOPIC_LENGTH]
    if not topic:
        raise ValueError("Topic cannot be empty")
    return topic

def split_text_into_sentences(text, max_length=200):
    sentences = re.split(r'([.!?])\s+', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    final_result = []
    for sentence in result:
        if len(sentence) > max_length:
            parts = re.split(r'([,;])\s+', sentence)
            temp = ""
            for part in parts:
                if len(temp + part) > max_length and temp:
                    final_result.append(temp)
                    temp = part
                else:
                    temp += part
            if temp:
                final_result.append(temp)
        else:
            final_result.append(sentence)
    return [s.strip() for s in final_result if s.strip()]

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(Exception)
)
def generate_tts_english_sync(text, voice_name):
    try:
        client = get_google_tts_client()
        sentences = split_text_into_sentences(text, max_length=200)
        all_audio = b""
        for sentence in sentences:
            synthesis_input = texttospeech.SynthesisInput(text=sentence)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            all_audio += response.audio_content
        return all_audio
    except Exception as e:
        print(f"Chirp3 TTS Error: {str(e)}")
        return None

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(Exception)
)
def generate_tts_standard_sync(text):
    try:
        client = get_google_tts_client()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-C",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.9,
        )
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        print(f"Standard TTS Error: {str(e)}")
        return None

async def generate_tts_async(text, voice_name):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_tts_english_sync, text, voice_name)

def safe_filename(filename):
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    filename = os.path.basename(filename)
    filename = filename[:100]
    return filename.strip('_')

def validate_deepseek_response(content):
    required_keys = ["main_text", "collocations", "opinion_texts", "discussion_questions"]
    if not all(k in content for k in required_keys):
        missing = [k for k in required_keys if k not in content]
        raise ValueError(f"Missing required keys: {missing}")
    if not isinstance(content['collocations'], list):
        raise ValueError("collocations must be a list")
    if len(content['collocations']) > config.MAX_VOCAB_ITEMS:
        content['collocations'] = content['collocations'][:config.MAX_VOCAB_ITEMS]
    for item in content['collocations']:
        if not all(k in item for k in ['english', 'russian']):
            raise ValueError("Each collocation must have 'english', 'russian'")
    if not all(k in content['opinion_texts'] for k in ['positive', 'negative', 'mixed']):
        raise ValueError("opinion_texts must have 'positive', 'negative', 'mixed'")
    if not isinstance(content['discussion_questions'], list):
        raise ValueError("discussion_questions must be a list")
    return True

@retry(
    stop=stop_after_attempt(config.API_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: print(f"Retry {retry_state.attempt_number}: {retry_state.outcome.exception()}")
)
def generate_content_with_deepseek(topic):
    prompt = f"""You are an English language teaching assistant. Create learning materials about the topic: "{topic}"

Please generate a JSON response with the following structure:
{{
  "main_text": "An engaging English text at CEFR B2/weak C1 level about {topic}. Should be 200-250 words long, natural and informative. MUST contain 3-5 phrasal verbs that are either typical for this context OR generically useful. Include the objects with phrasal verbs (e.g., 'pick up a language', 'look after children').",
  "collocations": [
    {{"english": "collocation/phrasal verb with object from text", "russian": "Russian translation"}},
    // Exactly 15 items total
    // MUST include all 3-5 phrasal verbs (with their objects as they appear in the text)
    // Remaining items should be useful collocations, expressions, verb+noun, or adjective+noun pairs from the text
    // All collocations must come directly from the main_text
  ],
  "opinion_texts": {{
    "positive": "A natural English response (B2/C1 level, 80-120 words) giving a positive reaction to the main topic. Should incorporate some vocabulary from the collocations list naturally.",
    "negative": "A natural English response (B2/C1 level, 80-120 words) giving a critical/negative reaction to the main topic. Should incorporate some vocabulary from the collocations list naturally.",
    "mixed": "A natural English response (B2/C1 level, 80-120 words) giving a balanced/mixed reaction to the main topic. Should incorporate some vocabulary from the collocations list naturally."
  }},
  "discussion_questions": [
    "Question 1 in English (B2/C1 level) - should encourage critical thinking",
    "Question 2 in English (B2/C1 level) - should encourage personal reflection",
    "Question 3 in English (B2/C1 level) - should prompt discussion",
    "Question 4 in English (B2/C1 level) - should be open-ended",
    "Question 5 in English (B2/C1 level) - should stimulate debate"
  ]
}}

CRITICAL REQUIREMENTS:
1. Main text MUST contain 3-5 phrasal verbs with their objects
2. ALL collocations must come from the main_text
3. The first 3-5 collocations MUST be the phrasal verbs
4. Remaining collocations should be useful expressions from the text
5. Opinion texts should naturally use some collocations but sound conversational
6. Discussion questions should be thought-provoking
7. Return ONLY valid JSON, no additional text"""

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert English language teacher who creates engaging, natural content at CEFR B2/C1 level with a focus on phrasal verbs and useful collocations. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        timeout=45.0
    )
    content_text = response.choices[0].message.content
    json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
    if json_match:
        content_text = json_match.group()
    content = json.loads(content_text)
    validate_deepseek_response(content)
    return content

async def create_vocabulary_file_with_tts(collocations, topic, progress_callback=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic_name = safe_filename(topic)
    filename = f"{safe_topic_name}_{timestamp}_collocations.txt"
    content = ""
    audio_files = {}
    total_items = len(collocations)
    tts_tasks = []
    for item in collocations:
        tts_tasks.append(generate_tts_async(item['english'], voice_name="en-US-Standard-C"))
    audio_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    for idx, (item, audio_data) in enumerate(zip(collocations, audio_results)):
        if progress_callback:
            await progress_callback(idx + 1, total_items)
        if isinstance(audio_data, Exception) or not audio_data:
            content += f"{item['russian']}\t{item['english']}\n"
        else:
            hash_object = hashlib.md5(item['english'].encode())
            audio_filename = f"tts_{hash_object.hexdigest()}.mp3"
            audio_filename = safe_filename(audio_filename)
            audio_files[audio_filename] = audio_data
            anki_tag = f"[sound:{audio_filename}]"
            content += f"{item['russian']}\t{item['english']}\t{anki_tag}\n"
    return filename, content, audio_files

def create_zip_package(vocab_filename, vocab_content, audio_files, html_filename, html_content, topic, timestamp):
    safe_topic_name = safe_filename(topic)
    zip_filename = f"{safe_topic_name}_{timestamp}_complete_package.zip"
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        safe_vocab = safe_filename(vocab_filename)
        zip_file.writestr(safe_vocab, vocab_content.encode('utf-8'))
        for audio_filename, audio_data in audio_files.items():
            safe_audio = safe_filename(audio_filename)
            zip_file.writestr(safe_audio, audio_data)
        safe_html = safe_filename(html_filename)
        zip_file.writestr(safe_html, html_content.encode('utf-8'))
    zip_buffer.seek(0)
    file_size = zip_buffer.getbuffer().nbytes
    if file_size > config.MAX_FILE_SIZE:
        raise ValueError(f"ZIP too large: {file_size / 1024 / 1024:.1f}MB")
    return zip_filename, zip_buffer

def create_html_document(topic, content, timestamp):
    safe_topic = safe_filename(topic)
    html_filename = f"{safe_topic}_{timestamp}_materials.html"
    vocab_rows = ""
    for i, item in enumerate(content['collocations'], 1):
        vocab_rows += f"""
        <tr>
            <td>{i}</td>
            <td class="english">{item['english']}</td>
            <td class="russian">{item['russian']}</td>
        </tr>
        """
    questions_html = ""
    for i, question in enumerate(content['discussion_questions'], 1):
        questions_html += f"""
        <div class="question">
            <span class="question-number">{i}</span>
            <span class="question-text">{question}</span>
        </div>
        """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>English Learning Materials: {topic}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.8;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .header .subtitle {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-icon {{
            font-size: 1.2em;
        }}
        .main-text {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 15px;
            font-size: 1.15em;
            line-height: 1.9;
            color: #2c3e50;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .english {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }}
        .russian {{
            color: #7f8c8d;
            font-style: italic;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        tbody tr:hover {{
            background: #e9ecef;
            transition: background 0.3s;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .opinion-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid;
        }}
        .opinion-positive {{
            border-left-color: #2ecc71;
        }}
        .opinion-negative {{
            border-left-color: #e74c3c;
        }}
        .opinion-mixed {{
            border-left-color: #f39c12;
        }}
        .opinion-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            font-size: 1.3em;
            font-weight: 600;
        }}
        .opinion-text {{
            font-size: 1.05em;
            line-height: 1.8;
            color: #2c3e50;
        }}
        .question {{
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 10px;
            display: flex;
            gap: 15px;
            align-items: start;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }}
        .question-number {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }}
        .question-text {{
            font-size: 1.05em;
            line-height: 1.7;
            color: #2c3e50;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
        @media (max-width: 768px) {{
            .content {{
                padding: 20px;
            }}
            .header {{
                padding: 30px 20px;
            }}
            .main-text {{
                font-size: 1em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 English Learning Materials</h1>
            <div class="subtitle">Topic: {topic}</div>
            <div class="subtitle">Level: CEFR B2 / Weak C1</div>
            <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </div>
        <div class="content">
            <!-- Collocations -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">📚</span>
                    Collocations & Phrasal Verbs
                </h2>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>English</th>
                            <th>Russian (Русский)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {vocab_rows}
                    </tbody>
                </table>
            </div>
            <!-- Main Text -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">📖</span>
                    Main Text
                </h2>
                <div class="main-text">{content['main_text']}</div>
            </div>
            <!-- Opinion Texts -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">💭</span>
                    Different Reactions
                </h2>
                <div class="opinion-card opinion-positive">
                    <div class="opinion-header">
                        <span>😊</span>
                        <span>Positive Reaction</span>
                    </div>
                    <div class="opinion-text">{content['opinion_texts']['positive']}</div>
                </div>
                <div class="opinion-card opinion-negative">
                    <div class="opinion-header">
                        <span>🤔</span>
                        <span>Critical Reaction</span>
                    </div>
                    <div class="opinion-text">{content['opinion_texts']['negative']}</div>
                </div>
                <div class="opinion-card opinion-mixed">
                    <div class="opinion-header">
                        <span>⚖️</span>
                        <span>Balanced Reaction</span>
                    </div>
                    <div class="opinion-text">{content['opinion_texts']['mixed']}</div>
                </div>
            </div>
            <!-- Discussion Questions -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">💬</span>
                    Discussion Questions
                </h2>
                {questions_html}
            </div>
        </div>
        <div class="footer">
            <p>Generated by English Learning Bot 🤖</p>
            <p>CEFR B2 / Weak C1 Level Materials</p>
        </div>
    </div>
</body>
</html>"""
    return html_filename, html_content

CHIRP_VOICES = [
    "en-US-Chirp3-HD-Achird",
    "en-US-Chirp3-HD-Callirrhoe",
    "en-US-Chirp3-HD-Achernar",
    "en-US-Chirp3-HD-Algenib",
    "en-US-Chirp3-HD-Erinome",
    "en-US-Chirp3-HD-Schedar",
    "en-US-Chirp3-HD-Kore"
]

async def handle_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    topic_raw = update.message.text.strip()
    
    if not rate_limiter.is_allowed(user_id):
        reset_time = rate_limiter.get_reset_time(user_id)
        await update.message.reply_text(
            f"⏱️ Rate limit reached!\n\n"
            f"You've used your 5 requests for this hour.\n"
            f"Please try again in {reset_time // 60} minutes."
        )
        return
    
    try:
        topic = validate_topic(topic_raw)
    except ValueError as e:
        await update.message.reply_text(f"❌ Invalid topic: {str(e)}\n\nPlease try a different topic.")
        return

    await update.message.chat.send_action(action="typing")
    progress_msg = await update.message.reply_text(
        f"📚 Materials for your '{topic[:20]}...'...\n\n"
        f"⏳ Progress: 0/5\n"
        f"⬜⬜⬜⬜⬜\n"
        f"Initializing..."
    )
    
    async def update_progress(step, message):
        progress_bar = "🟩" * step + "⬜" * (5 - step)
        try:
            await progress_msg.edit_text(
                f"📚 Materials for your '{topic[:20]}...'...\n\n"
                f"⏳ Progress: {step}/5\n"
                f"{progress_bar}\n"
                f"{message}"
            )
        except:
            pass
    
    try:
        await update_progress(1, "🤖 Generating content with AI...")
        await update.message.chat.send_action(action="typing")
        content = generate_content_with_deepseek(topic)
        if not content:
            await update.message.reply_text("❌ Failed to generate content. Please try again.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = safe_filename(topic)
        
        await update_progress(2, "📄 Creating HTML document...")
        html_filename, html_content = create_html_document(topic, content, timestamp)
        html_file = BytesIO(html_content.encode('utf-8'))
        html_file.name = html_filename
        await update.message.reply_document(document=html_file, filename=html_filename)
        
        await update_progress(3, "🎵 Generating TTS for collocations...")
        await update.message.chat.send_action(action="record_voice")
        
        async def vocab_progress(current, total):
            if current % 3 == 0:
                await update_progress(3, f"🎵 Generating TTS... ({current}/{total})")
        
        vocab_filename, vocab_content, audio_files = await create_vocabulary_file_with_tts(
            content['collocations'], safe_topic, progress_callback=vocab_progress
        )
        
        # Step: Generate TTS for the 4 main texts using random Chirp voices
        await update_progress(4, "🎧 Generating narration audio...")
        await update.message.chat.send_action(action="record_voice")
        
        text_mapping = {
            "Main_Text.mp3": content['main_text'],
            "Positive_Reaction.mp3": content['opinion_texts']['positive'],
            "Critical_Reaction.mp3": content['opinion_texts']['negative'],
            "Balanced_Reaction.mp3": content['opinion_texts']['mixed']
        }
        
        selected_voices = random.sample(CHIRP_VOICES, 4)  # pick 4 distinct voices
        audio_tasks = []
        filenames_and_texts = list(text_mapping.items())
        
        for i, (filename, text) in enumerate(filenames_and_texts):
            voice = selected_voices[i]
            audio_tasks.append(generate_tts_async(text, voice))
        
        audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)
        narration_files = []
        
        for i, (filename, _) in enumerate(filenames_and_texts):
            audio_data = audio_results[i]
            if not isinstance(audio_data, Exception) and audio_
                audio_buffer = BytesIO(audio_data)
                audio_buffer.name = filename
                narration_files.append((filename, audio_buffer))
            else:
                print(f"TTS failed for {filename}: {audio_data}")
        
        # Step: Send ZIP (unchanged)
        await update_progress(5, "📤 Sending package...")
        zip_filename, zip_buffer = create_zip_package(
            vocab_filename, vocab_content, audio_files, html_filename, html_content, topic, timestamp
        )
        zip_file_obj = BytesIO(zip_buffer.getvalue())
        zip_file_obj.name = zip_filename
        await update.message.reply_document(document=zip_file_obj, filename=zip_filename)
        
        # Send Anki .txt separately
        anki_file = BytesIO(vocab_content.encode('utf-8'))
        anki_file.name = "anki_import.txt"
        await update.message.reply_document(document=anki_file, filename="anki_import.txt")
        
        # Send each narration MP3
        if narration_files:
            await update.message.reply_text("🎧 Listen to the texts:")
            for filename, audio_buffer in narration_files:
                await update.message.reply_audio(audio=audio_buffer, filename=filename)
        
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)[:200]}"
        print(f"[Bot] Full error: {e}")
        await update.message.reply_text(error_msg)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """Welcome to the English Learning Bot! 🎯

Please give me a topic you want to discuss:

Be specific e.g.:

NOT - How can we use AI in business ( = too general)

GOOD = How can non-coders working in an IT company use AI?

Some examples of topics:
- "How has X been changing"
- "What is happening in late 2025 with ..."
- "Is X better than Y"
- "Predictions for X in 2026"
- "How to ..."
- "Why do people...?"
"""
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reset_time = rate_limiter.get_reset_time(user_id)
    help_text = (
        "📖 **How to Use:**\n\n"
        "1. Send me a topic (max 100 chars)\n"
        "2. Receive:\n"
        "   • HTML document\n"
        "   • ZIP package\n"
        "   • Separate Anki .txt file\n"
        "   • 4 audio files (Main + 3 reactions)\n"
        "   • Each with unique Chirp3 HD voice\n\n"
        "⚡ **Rate Limit:** 5 requests/hour"
    )
    if reset_time > 0:
        help_text += f"\n⏱️ Resets in {reset_time // 60} min"
    await update.message.reply_text(help_text, parse_mode='Markdown')

if __name__ == "__main__":
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_topic))
    print("Bot is running...")
    application.run_polling()
