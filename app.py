import os
import base64
import requests
from flask import Flask, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_mime_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    mime_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')

def build_ocr_prompt(output_format):
    """Build a robust OCR prompt that preserves text verbatim while
    capturing math equations (as LaTeX) and graphs/charts/diagrams."""
    fmt = (output_format or "plaintext").lower()

    # Core transcription contract shared by every output format.
    base = (
        "You are a precise OCR transcription engine. Transcribe ALL content from "
        "the image exactly as it appears, in natural reading order (top to bottom, "
        "left to right; handle multi-column layouts column by column).\n\n"
        "STRICT RULES:\n"
        "- Reproduce text VERBATIM. Do not translate, summarize, correct spelling, "
        "rephrase, or add commentary of any kind.\n"
        "- Preserve original line breaks, paragraphs, indentation, capitalization, "
        "punctuation, and ordering.\n"
        "- Keep numbers, units, symbols, and special characters exactly as shown.\n"
        "- If a region is unreadable or ambiguous, mark it as [illegible] rather "
        "than guessing.\n"
        "- Do NOT wrap your entire response in a code fence and do NOT include any "
        "preamble such as 'Here is the text'. Output only the transcription.\n\n"
        "MATH EQUATIONS:\n"
        "- Transcribe every mathematical expression as LaTeX.\n"
        "- Use inline math delimiters \\( ... \\) for expressions within a line of "
        "text, and display math delimiters \\[ ... \\] for standalone/centered "
        "equations.\n"
        "- Preserve fractions, exponents, subscripts, roots, integrals, sums, "
        "matrices, Greek letters, and operators precisely.\n\n"
        "GRAPHS, CHARTS & DIAGRAMS:\n"
        "- For any plot, chart, graph, or diagram, transcribe all visible labels, "
        "titles, axis names, units, legends, and data values verbatim.\n"
        "- Then add a concise factual description of the figure delimited by "
        "[FIGURE] ... [/FIGURE], covering its type (e.g. bar chart, line graph, "
        "scatter plot, flowchart), what the axes represent, the data series, and "
        "any clearly readable trends or values. Describe only what is visible; do "
        "not infer beyond the image.\n"
        "- When a curve or function is plotted with an explicit equation, also "
        "transcribe that equation as LaTeX.\n"
    )

    if fmt == "markdown":
        fmt_rules = (
            "\nOUTPUT FORMAT — MARKDOWN:\n"
            "- Preserve document structure using Markdown: headings (#), lists, "
            "bold/italic, blockquotes, and tables (GitHub-flavored) for tabular data.\n"
            "- Keep math in LaTeX delimiters (\\( \\) and \\[ \\]) so it renders "
            "with MathJax/KaTeX.\n"
        )
    else:
        fmt_rules = (
            "\nOUTPUT FORMAT — PLAIN TEXT:\n"
            "- Return plain text only. Do not add Markdown styling characters that "
            "were not present in the source.\n"
            "- Keep math in LaTeX delimiters (\\( \\) and \\[ \\]).\n"
        )

    return base + fmt_rules


def call_openrouter_api(image_path, output_format):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    base64_image = encode_image_to_base64(image_path)
    mime_type = get_image_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{base64_image}"

    prompt_text = build_ocr_prompt(output_format)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]
    
    payload = {
        "model": "google/gemini-3.5-flash",
        "messages": messages
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    output_format = request.form.get('format', 'plaintext')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = call_openrouter_api(filepath, output_format)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            if 'choices' in result and len(result['choices']) > 0:
                ocr_text = result['choices'][0]['message']['content']
                return jsonify({
                    'success': True,
                    'text': ocr_text,
                    'format': output_format
                })
            else:
                return jsonify({'error': 'No text extracted from image'}), 500
                
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)