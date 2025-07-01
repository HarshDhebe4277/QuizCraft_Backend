from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_session import Session
import google.generativeai as genai
import re
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import tempfile
import os
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

# === Setup ===
load_dotenv()
app = Flask(__name__)

# Session Config
app.secret_key = os.getenv("SECRET_KEY") or "super_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_SECURE'] = False  # Only for local development
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
Session(app)




app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

flashcard_cache = {}

# === User Model ===
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=True)  # Nullable for Google accounts


# === Auth Routes ===

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Email and password required'}), 400

    user = User.query.filter_by(email=email).first()
    if user and user.password and bcrypt.check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['email'] = user.email
        session['username'] = user.username
        return jsonify({'status': 'success'})

    return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'status': 'error', 'message': 'All fields are required'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'status': 'error', 'message': 'Email already exists'}), 409

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()

    return jsonify({'status': 'success'})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# === Page Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/register-page')
def register_page():
    return render_template('register.html')

# === Flashcard Generation ===
@app.route('/generate_flashcards', methods=['POST'])
def generate_flashcards():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required', 'flashcards': []}), 401

    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        count = data.get('count', 0)

        if not text:
            return jsonify({'status': 'error', 'message': 'Please enter study notes.', 'flashcards': []}), 400

        # Check cache
        if text in flashcard_cache:
            flashcards = flashcard_cache[text]
        else:
            prompt = (
                "Generate informative flashcards from this content.\n"
                "Format each flashcard as:\n"
                "Question: <question>\nAnswer: <answer>\n"
                "No other output.\n\n"
                f"Content:\n{text}"
            )

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            output = response.text.strip()

            matches = re.findall(
                r'Question[:>]\s*(.+?)\s*Answer[:>]\s*(.+?)(?=\nQuestion[:>]|$)',
                output,
                flags=re.IGNORECASE | re.DOTALL
            )

            flashcards = []
            for q, a in matches:
                q, a = q.strip(), a.strip()
                if len(q) > 5 and len(a) > 5:
                    flashcards.append({
                        'id': len(flashcards) + 1,
                        'question': q,
                        'answer': a
                    })

            if not flashcards:
                return jsonify({
                    'status': 'error',
                    'message': 'No flashcards found. Try different input.',
                    'flashcards': []
                }), 400

            flashcard_cache[text] = flashcards

        # Limit the flashcards based on requested count
        if isinstance(count, int) and count > 0:
            flashcards = flashcards[:count]

        return jsonify({'status': 'success', 'flashcards': flashcards})

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Something went wrong: {str(e)}',
            'flashcards': []
        }), 500


@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio uploaded'}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        whisper_model = WhisperModel("base", compute_type="int8")
        segments, _ = whisper_model.transcribe(tmp.name)
        transcript = ' '.join([seg.text for seg in segments])

    if not transcript.strip():
        return jsonify({'status': 'error', 'message': 'Could not transcribe audio'}), 400

    return jsonify({'status': 'success', 'transcript': transcript})


@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    user_answer = data.get('user_answer', '').strip()
    correct_answer = data.get('correct_answer', '').strip()

    if not user_answer or not correct_answer:
        return jsonify({'correct': False, 'reason': 'Empty input'}), 400

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a smart evaluator. Compare the user’s answer to the correct answer.\n"
            f"Correct Answer: {correct_answer}\n"
            f"User Answer: {user_answer}\n"
            "Is the user’s answer semantically correct? Reply only 'yes' or 'no'."
        )
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        return jsonify({'correct': 'yes' in result})
    except Exception as e:
        return jsonify({'correct': False, 'error': str(e)}), 500


# === Run Server ===
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run()
