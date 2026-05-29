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
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment variables")
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in environment variables")

class Config:
    MAX_TOPIC_LENGTH = 100
    MAX_VOCAB_ITEMS = 15
    TTS_TIMEOUT = 30
    API_RETRY_ATTEMPTS = 3
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 3600
    MAX_FILE_SIZE = 50 * 1024 * 1024
    TRACKING_SHEET_ID = os.getenv("TRACKING_SHEET_ID")

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
    credentials = service_account.Credentials.from_service_account_file(
        "google-creds.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return texttospeech.TextToSpeechClient(credentials=credentials)

def get_sheets_client():
    """Initialize Google Sheets client"""
    credentials = service_account.Credentials.from_service_account_file(
        "google-creds.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build('sheets', 'v4', credentials=credentials)

async def track_usage_google_sheets(user_id, username, first_name, last_name, topic):
    """Track student usage in Google Sheets"""
    try:
        if not config.TRACKING_SHEET_ID:
            logger.warning("[Tracking] No TRACKING_SHEET_ID configured, skipping")
            return

        sheets_client = get_sheets_client()

        # Prepare data row
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_name = f"{first_name or ''} {last_name or ''}".strip() or "Unknown"

        row_data = [[
            timestamp,
            user_id,
            username or "No username",
            full_name,
            topic[:50]  # Truncate long topics
        ]]

        # Append to sheet
        sheets_client.spreadsheets().values().append(
            spreadsheetId=config.TRACKING_SHEET_ID,
            range="A:E",
            valueInputOption="RAW",
            body={"values": row_data}
        ).execute()

        logger.info(f"[Tracking] ✅ Logged to Google Sheets: {full_name} ({username}) - '{topic[:30]}'")
    except Exception as e:
        logger.error(f"[Tracking] ❌ Failed to log to Google Sheets: {e}")

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
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i+1] in '.!?':
            result.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            if sentences[i].strip():
                result.append(sentences[i])
            i += 1
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
def generate_tts_chirp3_sync(text, voice_name):
    """Generate TTS using Chirp3 HD voices for main texts"""
    try:
        logger.info(f"[Chirp3 TTS] Generating for voice '{voice_name}', text length: {len(text)}")
        client = get_google_tts_client()
        sentences = split_text_into_sentences(text, max_length=200)
        logger.info(f"[Chirp3 TTS] Split into {len(sentences)} sentences")

        all_audio = b""
        for idx, sentence in enumerate(sentences):
            synthesis_input = texttospeech.SynthesisInput(text=sentence)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            all_audio += response.audio_content
            logger.info(f"[Chirp3 TTS] Sentence {idx+1}/{len(sentences)} completed")

        logger.info(f"[Chirp3 TTS] ✅ Success: {len(all_audio)} bytes")
        return all_audio
    except Exception as e:
        logger.error(f"[Chirp3 TTS] ❌ Error: {type(e).__name__}: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(Exception)
)
def generate_tts_wavenet_sync(text, voice_name="en-US-Wavenet-H"):
    """Generate TTS using Wavenet voices for Anki cards"""
    try:
        logger.info(f"[Wavenet TTS] Generating for '{text[:50]}...' with voice '{voice_name}'")
        client = get_google_tts_client()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        audio_size = len(response.audio_content)
        logger.info(f"[Wavenet TTS] ✅ Success: {audio_size} bytes for '{text[:30]}'")
        return response.audio_content
    except Exception as e:
        logger.error(f"[Wavenet TTS] ❌ Failed for '{text[:50]}': {type(e).__name__}: {str(e)}")
        raise

async def generate_tts_chirp3_async(text, voice_name):
    """Async wrapper for Chirp3 TTS"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_tts_chirp3_sync, text, voice_name)

async def generate_tts_wavenet_async(text, voice_name="en-US-Wavenet-H"):
    """Async wrapper for Wavenet TTS"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_tts_wavenet_sync, text, voice_name)

def safe_filename(filename):
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    filename = os.path.basename(filename)
    filename = filename[:100]
    return filename.strip('_')

def validate_deepseek_response(content):
    required_keys = ["main_text", "collocations", "opinion_texts", "speaking_questions"]
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
    if not isinstance(content['speaking_questions'], list):
        raise ValueError("speaking_questions must be a list")
    return True

@retry(
    stop=stop_after_attempt(config.API_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(f"Retry {retry_state.attempt_number}: {retry_state.outcome.exception()}")
)
def generate_content_with_deepseek(topic):
    logger.info(f"[DeepSeek] Generating content for: '{topic}'")

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
  "speaking_questions": [
    {{"question": "Question 1 reacting to a specific idea from the main_text, using 1-2 collocations from the list naturally", "target_expressions": ["collocation used 1", "collocation used 2"]}},
    {{"question": "Question 2 — different idea from text, 1-2 collocations", "target_expressions": ["collocation used"]}},
    {{"question": "Question 3 — different idea from text, 1-2 collocations", "target_expressions": ["collocation used"]}},
    {{"question": "Question 4 — different idea from text, 1-2 collocations", "target_expressions": ["collocation used"]}},
    {{"question": "Question 5 — different idea from text, 1-2 collocations", "target_expressions": ["collocation used"]}}
  ]
}}

CRITICAL REQUIREMENTS:
1. Main text MUST contain 3-5 phrasal verbs with their objects
2. ALL collocations must come from the main_text
3. The first 3-5 collocations MUST be the phrasal verbs
4. Remaining collocations should be useful expressions from the text
5. Opinion texts should naturally use some collocations but sound conversational
6. Speaking questions must each react to a DIFFERENT specific idea/claim from the main_text
7. Each speaking question uses 1-2 collocations from the list naturally in the question itself
8. Question patterns (vary across the 5): "Do you agree that [idea from text]?", 
   "To what extent is it true that [idea from text]?", "How far has [idea from text] been true in your experience?",
   "Would you say that [idea from text]?", "Is [idea from text] realistic in your view?"
9. Questions should prompt a reaction to the TEXT's ideas, not tangential topics
10. Return ONLY valid JSON, no additional text"""

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
    logger.info(f"[DeepSeek] Received response, parsing...")

    json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
    if json_match:
        content_text = json_match.group()

    content = json.loads(content_text)
    validate_deepseek_response(content)
    logger.info(f"[DeepSeek] ✅ Content validated successfully")
    return content

async def create_vocabulary_file_with_tts(collocations, topic, progress_callback=None):
    """Create Anki vocabulary file with Wavenet TTS"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic_name = safe_filename(topic)
    filename = f"{safe_topic_name}_{timestamp}_collocations.txt"

    content = ""
    audio_files = {}
    total_items = len(collocations)

    logger.info(f"[Anki TTS] Starting generation for {total_items} collocations using Wavenet-H")

    tts_tasks = []
    for item in collocations:
        tts_tasks.append(generate_tts_wavenet_async(item['english'], voice_name="en-US-Wavenet-H"))

    logger.info(f"[Anki TTS] Awaiting {len(tts_tasks)} concurrent TTS generations...")
    audio_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    logger.info(f"[Anki TTS] All TTS generations completed")

    success_count = 0
    failed_count = 0

    for idx, (item, audio_data) in enumerate(zip(collocations, audio_results)):
        english_text = item['english']

        if progress_callback:
            await progress_callback(idx + 1, total_items)

        if isinstance(audio_data, Exception):
            logger.error(f"[Anki TTS] ❌ Exception for '{english_text}': {type(audio_data).__name__}: {audio_data}")
            failed_count += 1
            content += f"{item['russian']}\t{item['english']}\n"
        elif not audio_data:
            logger.error(f"[Anki TTS] ❌ Empty data for '{english_text}'")
            failed_count += 1
            content += f"{item['russian']}\t{item['english']}\n"
        else:
            hash_object = hashlib.md5(english_text.encode())
            audio_filename = f"tts_{hash_object.hexdigest()}.mp3"
            audio_filename = safe_filename(audio_filename)
            audio_files[audio_filename] = audio_data
            anki_tag = f"[sound:{audio_filename}]"
            content += f"{item['russian']}\t{item['english']}\t{anki_tag}\n"
            success_count += 1
            logger.info(f"[Anki TTS] ✅ {idx+1}/{total_items}: '{english_text[:30]}' -> {audio_filename}")

    logger.info(f"[Anki TTS] SUMMARY: ✅ {success_count} succeeded, ❌ {failed_count} failed out of {total_items} total")

    if failed_count > 0:
        logger.warning(f"[Anki TTS] ⚠️ WARNING: {failed_count}/{total_items} TTS generations failed")

    return filename, content, audio_files

def create_zip_package(vocab_filename, vocab_content, audio_files, html_filename, html_content, topic, timestamp):
    """Create ZIP with all files"""
    safe_topic_name = safe_filename(topic)
    zip_filename = f"{safe_topic_name}_{timestamp}_complete_package.zip"
    zip_buffer = BytesIO()

    logger.info(f"[ZIP] Creating package with {len(audio_files)} audio files")

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        safe_vocab = safe_filename(vocab_filename)
        zip_file.writestr(safe_vocab, vocab_content.encode('utf-8'))
        logger.info(f"[ZIP] Added vocabulary file: {safe_vocab}")

        for audio_filename, audio_data in audio_files.items():
            safe_audio = safe_filename(audio_filename)
            zip_file.writestr(safe_audio, audio_data)
        logger.info(f"[ZIP] Added {len(audio_files)} Anki TTS audio files")

        safe_html = safe_filename(html_filename)
        zip_file.writestr(safe_html, html_content.encode('utf-8'))
        logger.info(f"[ZIP] Added HTML file: {safe_html}")

    zip_buffer.seek(0)
    file_size = zip_buffer.getbuffer().nbytes
    logger.info(f"[ZIP] Package size: {file_size / 1024 / 1024:.2f}MB")

    if file_size > config.MAX_FILE_SIZE:
        raise ValueError(f"ZIP too large: {file_size / 1024 / 1024:.1f}MB")

    return zip_filename, zip_buffer

def create_html_document(topic, content, timestamp):
    """Create HTML document"""
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

    questions_html = ""  # speaking questions delivered via bot, not HTML

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
            <!-- Speaking practice delivered via /speak command -->
        </div>
        <div class="footer">
            <p>Generated by English Learning Bot 🤖</p>
            <p>CEFR B2 / Weak C1 Level Materials</p>
        </div>
    </div>
</body>
</html>"""

    logger.info(f"[HTML] Created document: {html_filename}")
    return html_filename, html_content

# Chirp3 HD voices for narration
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
    """Main handler for topic requests"""
    user_id = update.effective_user.id
    topic_raw = update.message.text.strip()

    logger.info(f"[Bot] User {user_id} requested topic: '{topic_raw}'")

    if not rate_limiter.is_allowed(user_id):
        reset_time = rate_limiter.get_reset_time(user_id)
        logger.warning(f"[Bot] User {user_id} rate limited, reset in {reset_time}s")
        await update.message.reply_text(
            f"⏱️ Rate limit reached!\n\n"
            f"You've used your 5 requests for this hour.\n"
            f"Please try again in {reset_time // 60} minutes."
        )
        return

    try:
        topic = validate_topic(topic_raw)
        logger.info(f"[Bot] Topic validated: '{topic}'")
    except ValueError as e:
        logger.error(f"[Bot] Invalid topic from user {user_id}: {str(e)}")
        await update.message.reply_text(f"❌ Invalid topic: {str(e)}\n\nPlease try a different topic.")
        return

    user = update.effective_user
    await track_usage_google_sheets(
        user_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        topic=topic
    )

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

        logger.info(f"[Bot] Starting content generation for user {user_id}")
        content = generate_content_with_deepseek(topic)

        if not content:
            logger.error(f"[Bot] Empty content returned")
            await update.message.reply_text("❌ Failed to generate content. Please try again.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = safe_filename(topic)

        await update_progress(2, "📄 Creating HTML document...")
        html_filename, html_content = create_html_document(topic, content, timestamp)
        logger.info(f"[Bot] HTML document created: {html_filename}")

        await update_progress(3, "🎧 Generating narration audio (Chirp3 HD)...")
        await update.message.chat.send_action(action="record_voice")

        text_mapping = {
            "Main_Text.mp3": content['main_text'],
            "Positive_Reaction.mp3": content['opinion_texts']['positive'],
            "Critical_Reaction.mp3": content['opinion_texts']['negative'],
            "Balanced_Reaction.mp3": content['opinion_texts']['mixed']
        }

        selected_voices = random.sample(CHIRP_VOICES, 4)
        logger.info(f"[Bot] Selected Chirp3 voices: {selected_voices}")

        audio_tasks = []
        for i, (filename, text) in enumerate(text_mapping.items()):
            voice = selected_voices[i]
            audio_tasks.append(generate_tts_chirp3_async(text, voice))

        logger.info(f"[Bot] Generating {len(audio_tasks)} Chirp3 narration files...")
        audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)

        narration_files = []
        for i, (filename, _) in enumerate(text_mapping.items()):
            audio_data = audio_results[i]
            if not isinstance(audio_data, Exception) and audio_data:
                audio_buffer = BytesIO(audio_data)
                audio_buffer.name = filename
                narration_files.append((filename, audio_buffer))
                logger.info(f"[Bot] ✅ Chirp3 audio generated: {filename}")
            else:
                logger.error(f"[Bot] ❌ Chirp3 TTS failed for {filename}: {audio_data}")

        await update_progress(4, "🎵 Generating TTS for Anki collocations (Wavenet-H)...")
        await update.message.chat.send_action(action="record_voice")

        async def vocab_progress(current, total):
            if current % 3 == 0:
                await update_progress(4, f"🎵 Generating Anki TTS... ({current}/{total})")

        vocab_filename, vocab_content, audio_files = await create_vocabulary_file_with_tts(
            content['collocations'], safe_topic, progress_callback=vocab_progress
        )

        if not audio_files:
            logger.error(f"[Bot] No Anki audio files generated!")
            await update.message.reply_text("⚠️ Warning: Could not generate TTS for Anki cards.")
        else:
            logger.info(f"[Bot] ✅ Generated {len(audio_files)} Anki TTS files")

        await update_progress(5, "📦 Creating ZIP package...")
        zip_filename, zip_buffer = create_zip_package(
            vocab_filename, vocab_content, audio_files, html_filename, html_content, topic, timestamp
        )
        logger.info(f"[Bot] ZIP package created: {zip_filename}")

        html_file = BytesIO(html_content.encode('utf-8'))
        html_file.name = html_filename
        await update.message.reply_document(
            document=html_file,
            filename=html_filename,
            caption="📄 Open this doc to see your topic texts and vocab list"
        )
        logger.info(f"[Bot] Sent HTML document")

        await update.message.reply_text(
            "👆 You can listen to the texts from the doc by playing the audio below 👇"
        )

        if narration_files:
            for filename, audio_buffer in narration_files:
                await update.message.reply_audio(audio=audio_buffer, filename=filename)
                logger.info(f"[Bot] Sent audio: {filename}")
        else:
            await update.message.reply_text("⚠️ Could not generate narration audio.")

        await update.message.reply_text("••• 💭 •••")

        await update.message.reply_text(
            "📇 If you're an Anki user, import the text doc below into Anki, "
            "and put the audio files from the ZIP folder into your Anki `collection.media` folder."
        )

        anki_file = BytesIO(vocab_content.encode('utf-8'))
        anki_file.name = "anki_import.txt"
        await update.message.reply_document(
            document=anki_file,
            filename="anki_import.txt"
        )
        logger.info(f"[Bot] Sent Anki import file")

        zip_file_obj = BytesIO(zip_buffer.getvalue())
        zip_file_obj.name = zip_filename
        await update.message.reply_document(
            document=zip_file_obj,
            filename=zip_filename
        )
        logger.info(f"[Bot] Sent ZIP package")

        file_size = zip_buffer.getbuffer().nbytes
        logger.info(f"[Bot] ✅ Successfully completed request for user {user_id}")

        # Save session to disk so /speak works even after restart
        save_speaking_session(user_id, {
            "topic": topic,
            "speaking_questions": content["speaking_questions"],
            "collocations": content["collocations"]
        })

        await update.message.reply_text(
            f"✅ All materials generated!\n\n"
            f"📊 Summary:\n"
            f"• Collocations: {len(content['collocations'])}\n"
            f"• Anki TTS files: {len(audio_files)}\n"
            f"• Narration audio: {len(narration_files)}\n"
            f"• ZIP size: {file_size / 1024 / 1024:.2f}MB\n\n"
            f"💬 When you\'ve done your Anki cards and listened to the audio, "
            f"send /speak to practise reacting to the ideas in the text."
        )

    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)[:200]}"
        logger.error(f"[Bot] ERROR for user {user_id}: {type(e).__name__}: {str(e)}", exc_info=True)
        await update.message.reply_text(error_msg)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
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
    """Handle /help command"""
    user_id = update.effective_user.id
    reset_time = rate_limiter.get_reset_time(user_id)

    help_text = (
        "📖 **How to Use:**\n\n"
        "1. Send me a topic (max 100 chars)\n"
        "2. Receive:\n"
        "   • HTML document with all materials\n"
        "   • 4 narration audio files (Chirp3 HD voices)\n"
        "   • Anki import .txt file\n"
        "   • ZIP package with Anki TTS files (Wavenet-H voice)\n\n"
        "📦 **For Anki:**\n"
        "   • Extract MP3 files from ZIP to collection.media folder\n"
        "   • Import the .txt file into Anki\n\n"
        "⚡ **Rate Limit:** 5 requests/hour"
    )

    if reset_time > 0:
        help_text += f"\n⏱️ Resets in {reset_time // 60} min"

    await update.message.reply_text(help_text, parse_mode='Markdown')

# ─── SPEAKING MODE ────────────────────────────────────────────────────────────

SPEAKING_SESSIONS_DIR = "speaking_sessions"

# In-memory speaking sessions: user_id -> {questions, index, topic, collocations}
speaking_sessions = {}

def save_speaking_session(user_id: int, data: dict):
    os.makedirs(SPEAKING_SESSIONS_DIR, exist_ok=True)
    path = os.path.join(SPEAKING_SESSIONS_DIR, f"{user_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[Speak] Saved session for user {user_id}")

def load_speaking_session(user_id: int):
    path = os.path.join(SPEAKING_SESSIONS_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def transcribe_voice_english(audio_path: str):
    """Transcribe voice using Google STT for English."""
    try:
        from google.cloud import speech as google_speech
        creds_str = os.getenv("GOOGLE_CREDENTIALS_JSON")
        if creds_str:
            credentials_dict = json.loads(creds_str)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            client = google_speech.SpeechClient(credentials=credentials)
        else:
            credentials = service_account.Credentials.from_service_account_file(
                "google-creds.json",
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            client = google_speech.SpeechClient(credentials=credentials)

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        response = client.recognize(
            config=google_speech.RecognitionConfig(
                encoding=google_speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                sample_rate_hertz=48000,
                language_code="en-US",
            ),
            audio=google_speech.RecognitionAudio(content=audio_data)
        )
        if response.results:
            return response.results[0].alternatives[0].transcript
    except Exception as e:
        logger.error(f"[Speak STT] {e}")
    return None

def generate_speaking_feedback(question: str, target_expressions: list, user_text: str) -> str:
    """
    Brief feedback:
    - One sentence on any obvious grammar/vocab error (skip minor issues)
    - Flexibility score X/5 for how flexibly they used the target expression(s)
    - One concrete tip with a rewrite example showing how to be more flexible
    """
    targets = ", ".join(f'"{e}"' for e in target_expressions)

    prompt = f"""You are a brief, encouraging English speaking coach for B2-C1 learners.

Question asked: {question}
Target expression(s) the student should use: {targets}
Student said: {user_text}

Give feedback in this exact format (3 parts, keep it SHORT):

1. ERROR (optional): Only mention ONE obvious grammar or vocab error if present. Skip minor issues. 
   If no clear error, omit this line entirely.

2. SCORE: X/5 — one sentence on HOW FLEXIBLY they used the target expression.
   Scoring guide:
   5 = Used it creatively (tense shift, adverb added, combined with another structure, hedged it)
   4 = Used it correctly and naturally
   3 = Used it but mechanically/directly from the question
   2 = Tried to use it but awkwardly
   1 = Didn't use it at all

3. TIP: One concrete example showing ONE way to be more flexible with this expression.
   Choose the most natural tip based on the structure type:
   - verb phrase → try: adverb ("inevitably X"), tense shift ("had always X"), "tend to/used to X"
   - adjective+noun → try: second adjective ("X and Y"), intensifier ("remarkably X")  
   - prediction/possibility → try: hedging stronger/weaker ("bound to", "unlikely to", "might well")
   - noun phrase → try: specific noun replacing generic, or possessive ("my own X")
   - any structure → try: conditional ("if...then X"), negation ("far from X"), question form
   Format: TIP: try [specific example rewriting part of their answer using the target more flexibly]

Keep the whole response under 60 words. Be warm and specific."""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Brief English speaking coach. Under 60 words total."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            timeout=30.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[Speak feedback] {e}")
        return "Good effort! Keep going. 💪"

async def speak_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/speak — start speaking practice on last generated topic."""
    user_id = update.effective_user.id

    session = load_speaking_session(user_id)
    if not session:
        await update.message.reply_text(
            "No previous session found. Send me a topic first to generate materials, "
            "then use /speak when you\'re ready to practise."
        )
        return

    topic     = session["topic"]
    questions = session["speaking_questions"]

    await update.message.reply_text(
        f"💬 Speaking practice — {topic}\n\n"
        f"5 questions based on the text. Answer each one with a voice message.\n"
        f"Focus on using the target expressions flexibly. Let\'s go! 🎤"
    )

    speaking_sessions[user_id] = {
        "questions": questions,
        "index": 0,
        "topic": topic,
    }

    await send_speaking_question(update.effective_chat.id, context, user_id)

async def send_speaking_question(chat_id: int, context, user_id: int):
    s         = speaking_sessions.get(user_id)
    if not s:
        return
    idx       = s["index"]
    questions = s["questions"]

    if idx >= len(questions):
        await context.bot.send_message(
            chat_id=chat_id,
            text="🎉 Speaking practice complete! Great work.\n\nSend a new topic any time to generate more materials."
        )
        del speaking_sessions[user_id]
        return

    q       = questions[idx]
    targets = q.get("target_expressions", [])

    text = (
        f"Question {idx + 1}/5\n\n"
        f"💬 {q['question']}\n\n"
        f"🎯 Target: {', '.join(targets)}\n\n"
        f"Reply with a voice message 🎤"
    )
    await context.bot.send_message(chat_id=chat_id, text=text)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s       = speaking_sessions.get(user_id)

    if not s:
        await update.message.reply_text(
            "Use /speak to start speaking practice first."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    voice_file = await update.effective_message.voice.get_file()
    voice_path = f"/tmp/voice_{user_id}_{int(time.time())}.ogg"
    await voice_file.download_to_drive(voice_path)

    user_text = transcribe_voice_english(voice_path)
    os.remove(voice_path)

    if not user_text:
        await update.message.reply_text(
            "Could not understand the audio — please try again 🎤"
        )
        return

    idx = s["index"]
    q   = s["questions"][idx]

    feedback = generate_speaking_feedback(
        question=q["question"],
        target_expressions=q.get("target_expressions", []),
        user_text=user_text
    )

    await update.message.reply_text(
        f"You said: \"{user_text}\"\n\n{feedback}"
    )

    s["index"] += 1
    if s["index"] < len(s["questions"]):
        await send_speaking_question(update.effective_chat.id, context, user_id)
    else:
        await update.message.reply_text(
            "🎉 All done! Great practice.\n\nSend a new topic any time."
        )
        del speaking_sessions[user_id]


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🤖 Starting English Learning Telegram Bot")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - Chirp3 voices for narration: {len(CHIRP_VOICES)} available")
    logger.info(f"  - Wavenet-H voice for Anki TTS")
    logger.info(f"  - Rate limit: {config.RATE_LIMIT_REQUESTS} requests per {config.RATE_LIMIT_WINDOW}s")
    logger.info("=" * 60)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("speak", speak_command))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_topic))

    logger.info("✅ Bot is running and ready to accept messages...")
    application.run_polling()