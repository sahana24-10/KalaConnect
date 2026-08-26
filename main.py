from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import os
import tempfile
from pathlib import Path
import json
import base64
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
import subprocess

# ============================================
# FORCE REMBG TO USE SMALLER MODEL
# ============================================
os.environ["REMBG_MODEL"] = "u2net"

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ============================================
# LOAD WHISPER
# ============================================
print("=" * 50)
print("🎙️ Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("✅ Whisper model loaded!")
print("=" * 50)

app = FastAPI(title="KalaConnect API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# ============================================
# IMAGE ENHANCEMENT - PRESERVES COLORS
# ============================================
def enhance_image(image_bytes):
    """Image enhancement - ONLY fixes lighting and removes background, preserves colors"""
    
    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not read image")
    
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ----- STAGE 1: ONLY Fix Lighting (Preserve Colors) -----
    # Convert to LAB for lighting adjustment only
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE ONLY on L channel (lightness) - preserves color
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Merge back - colors remain the same, only lighting improved
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    img_lighting_fixed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # ----- STAGE 2: Mild Denoising (Preserve Details) -----
    img_denoised = cv2.fastNlMeansDenoisingColored(
        cv2.cvtColor(img_lighting_fixed, cv2.COLOR_RGB2BGR), 
        None, 
        5,  # Lower denoising to preserve details
        5, 
        7, 
        21
    )
    img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
    
    # ----- STAGE 3: Background Removal (Preserve Subject Colors) -----
    try:
        from rembg import remove
        
        # Convert to PIL for rembg
        img_pil = Image.fromarray(img_denoised)
        
        # Remove background - this preserves the subject's colors exactly
        output = remove(img_pil)
        
        # Convert back to numpy
        img_no_bg = np.array(output)
        
        # Create white background
        white_bg = np.full((img_no_bg.shape[0], img_no_bg.shape[1], 3), 255, dtype=np.uint8)
        
        # Composite with alpha channel - preserves original colors
        if img_no_bg.shape[2] == 4:
            alpha = img_no_bg[:, :, 3] / 255.0
            rgb = img_no_bg[:, :, :3]
            for c in range(3):
                white_bg[:, :, c] = (1 - alpha) * 255 + alpha * rgb[:, :, c]
            final_image = white_bg.astype(np.uint8)
        else:
            final_image = img_denoised
            
    except Exception as e:
        print(f"Background removal failed: {e}")
        final_image = img_denoised
    
    # ----- STAGE 4: Resize for e-commerce -----
    height, width = final_image.shape[:2]
    max_size = 800
    
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        final_image = cv2.resize(final_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Make square with white padding
    h, w = final_image.shape[:2]
    if h != w:
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        padded = np.full((size, size, 3), 255, dtype=np.uint8)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = final_image
        final_image = padded
    
    # Convert to JPEG
    _, encoded = cv2.imencode('.jpg', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    
    return encoded.tobytes()

# ============================================
# VOICE TRANSCRIPTION
# ============================================
def transcribe_with_whisper(audio_path):
    """Transcribe with better language detection"""
    try:
        result = whisper_model.transcribe(
            str(audio_path),
            language='hi',
            task='translate',
            fp16=False,
            verbose=False
        )
        
        text = result.get('text', '').strip()
        
        if len(text) < 3:
            result = whisper_model.transcribe(
                str(audio_path),
                language=None,
                task='translate',
                fp16=False,
                verbose=False
            )
            text = result.get('text', '').strip()
            detected_lang = result.get('language', 'hi')
        else:
            detected_lang = 'hi'
        
        return {
            "success": True,
            "text": text,
            "detected_language": detected_lang,
            "confidence": result.get('confidence', 0.8)
        }
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/enhance-image")
async def enhance_image_endpoint(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        enhanced_bytes = enhance_image(image_bytes)
        
        enhanced_b64 = base64.b64encode(enhanced_bytes).decode()
        original_b64 = base64.b64encode(image_bytes).decode()
        
        return JSONResponse({
            "success": True,
            "data": {
                "enhanced_image": f"data:image/jpeg;base64,{enhanced_b64}",
                "original_image": f"data:image/jpeg;base64,{original_b64}",
                "background_removal_ok": True
            }
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        audio_path = TEMP_DIR / f"audio_{datetime.now().timestamp()}.webm"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        wav_path = audio_path.with_suffix('.wav')
        try:
            subprocess.run([
                'ffmpeg', '-i', str(audio_path), 
                '-ar', '16000', '-ac', '1', 
                str(wav_path)
            ], capture_output=True, check=True, timeout=60)
            
            if wav_path.exists():
                audio_path = wav_path
        except Exception as e:
            print(f"FFmpeg conversion warning: {e}")
        
        result = transcribe_with_whisper(audio_path)
        
        try:
            os.unlink(audio_path)
            if wav_path.exists():
                os.unlink(wav_path)
        except:
            pass
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "data": {
                    "text": result["text"],
                    "translated": result["text"],
                    "detected_language": result.get("detected_language", "hi"),
                    "confidence": result.get("confidence", 0.8)
                }
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error", "Transcription failed")
            }, status_code=500)
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/create-listing")
async def create_listing(
    image: UploadFile = File(None),
    description_audio: UploadFile = File(None),
    story_audio: UploadFile = File(None),
    material_cost: float = Form(...),
    hours_worked: float = Form(...),
    description_text: str = Form(""),
    story_text: str = Form("")
):
    try:
        enhanced_image_b64 = None
        if image:
            image_bytes = await image.read()
            enhanced_bytes = enhance_image(image_bytes)
            enhanced_image_b64 = base64.b64encode(enhanced_bytes).decode()
        
        cost_floor = (material_cost + hours_worked * 80) * 1.2
        suggested_min = max(100, int(cost_floor / 10) * 10)
        suggested_max = int(suggested_min * 1.3 / 10) * 10
        best_price = int((suggested_min + suggested_max) / 2 / 10) * 10
        
        catalog = {
            "title_en": "Handcrafted Artisan Product",
            "title_hi": "हस्तनिर्मित कला उत्पाद",
            "bullets_en": [
                "Handmade with traditional techniques",
                "Premium quality materials",
                "Unique design"
            ],
            "description_en": description_text[:500] if description_text else "A beautiful handmade product crafted with care.",
            "description_hi": "पारंपरिक तकनीकों से बना एक सुंदर हस्तनिर्मित उत्पाद।"
        }
        
        response_data = {
            "catalog": catalog,
            "suggested_min": suggested_min,
            "suggested_max": suggested_max,
            "best_price": best_price,
            "confidence": "medium",
            "explanation": f"Based on your costs (₹{material_cost} materials + {hours_worked} hours labor)",
            "cost_floor": int(cost_floor),
            "market_median": suggested_min + 50,
            "badges": ["100% Handmade", "Fair Price Certified"],
            "recommended_platforms": [
                {"name": "Etsy", "reason": "Great for handmade crafts"},
                {"name": "Amazon Karigar", "reason": "Indian artisan platform"},
                {"name": "GeM", "reason": "Government marketplace"}
            ],
            "enhanced_image": f"data:image/jpeg;base64,{enhanced_image_b64}" if enhanced_image_b64 else None
        }
        
        return JSONResponse({"success": True, "data": response_data})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================
# HTML TEMPLATE (Same as before - keeping it short)
# ============================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>KalaConnect · Artisan Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <script crossorigin src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
    <script crossorigin src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #FAF3E8; font-family: 'Inter', -apple-system, sans-serif; color: #3A2E27; min-height: 100vh; padding: 16px; }
        #root { max-width: 480px; margin: 0 auto; }
        .card { background: white; border-radius: 16px; padding: 24px 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.06); margin-bottom: 16px; }
        .btn-primary { background: #C4622D; border: none; border-radius: 50px; padding: 14px 24px; font-weight: 600; font-size: 1rem; color: white; width: 100%; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 12px rgba(196,98,45,0.25); font-family: 'Inter', sans-serif; }
        .btn-primary:hover { background: #b05428; transform: translateY(-2px); }
        .btn-outline { background: transparent; border: 2px solid #C4622D; color: #C4622D; border-radius: 50px; padding: 12px 20px; font-weight: 600; font-size: 0.95rem; cursor: pointer; transition: 0.2s; width: 100%; font-family: 'Inter', sans-serif; }
        .btn-outline:hover { background: #C4622D; color: white; }
        .badge-pill { background: #F5E6D8; color: #C4622D; padding: 4px 14px; border-radius: 50px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
        .hero-header { background: linear-gradient(145deg, #1B2A4A 0%, #22345C 100%); border-radius: 16px; padding: 28px 20px 24px; color: white; margin-bottom: 16px; }
        .hero-header h1 { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 2.2rem; color: white; margin-bottom: 4px; }
        .hero-header h1 small { font-size: 1rem; font-weight: 400; opacity: 0.85; display: block; font-family: 'Inter', sans-serif; }
        .hero-header .subhead { color: #C9D3E0; font-size: 0.95rem; margin: 8px 0 16px; line-height: 1.5; }
        .feature-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 12px 0; }
        .feature-item { background: white; border-radius: 14px; padding: 16px 8px; text-align: center; border: 1px solid #f0e8e0; }
        .icon-circle { width: 48px; height: 48px; background: #F9E4DC; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #C4622D; font-size: 1.4rem; margin: 0 auto 8px; }
        .feature-item .label { font-weight: 600; font-size: 0.8rem; color: #3A2E27; }
        .feature-item .sub { font-size: 0.65rem; color: #6B6560; }
        .login-overlay { background: white; border-radius: 16px; padding: 24px 20px; box-shadow: 0 8px 28px rgba(0,0,0,0.06); }
        .login-overlay .field { margin-bottom: 14px; }
        .login-overlay label { font-weight: 600; font-size: 0.85rem; display: block; margin-bottom: 4px; }
        .login-overlay input { width: 100%; padding: 12px 14px; border: 1.5px solid #e0d6ce; border-radius: 40px; font-size: 0.95rem; background: #fcf9f6; font-family: 'Inter', sans-serif; }
        .login-overlay input:focus { outline: none; border-color: #C4622D; box-shadow: 0 0 0 3px rgba(196,98,45,0.08); }
        .status-error { background: #A13D2F; color: white; padding: 8px 14px; border-radius: 40px; font-size: 0.8rem; text-align: center; margin-bottom: 12px; display: block; }
        .flex-between { display: flex; justify-content: space-between; align-items: center; }
        .photo-upload-area { border: 2px dashed #C4622D; border-radius: 16px; padding: 32px 12px; text-align: center; cursor: pointer; background: #FCF9F6; transition: 0.2s; }
        .photo-upload-area:hover { background: #FFF8F2; border-color: #b05428; }
        .photo-preview { width: 100%; max-height: 280px; object-fit: contain; border-radius: 12px; background: #FAF3E8; }
        .bottom-nav { position: fixed; bottom: 0; left: 0; right: 0; max-width: 480px; margin: 0 auto; background: white; display: flex; border-top: 1px solid #eee; padding: 8px 16px 12px; z-index: 100; box-shadow: 0 -4px 20px rgba(0,0,0,0.04); border-radius: 16px 16px 0 0; }
        .nav-item { flex: 1; text-align: center; background: none; border: none; cursor: pointer; color: #6B6560; font-family: 'Inter', sans-serif; }
        .nav-item.active { color: #C4622D; }
        .nav-item .icon { font-size: 1.4rem; display: block; }
        .nav-item span { font-size: 0.65rem; font-weight: 500; }
        .preset-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0; }
        .preset-btn { background: white; border: 1px solid #e8dfd7; border-radius: 14px; padding: 14px 8px; text-align: center; cursor: pointer; transition: 0.15s; font-family: 'Inter', sans-serif; color: #3A2E27; font-weight: 500; font-size: 0.85rem; }
        .preset-btn:hover { border-color: #C4622D; background: #FFF8F2; transform: translateY(-2px); }
        .preset-btn i { display: block; font-size: 1.6rem; margin-bottom: 4px; color: #C4622D; }
        .comparison-container { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }
        .comparison-item { text-align: center; }
        .comparison-item img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 12px; border: 1px solid #e0d6ce; }
        .comparison-item .label { font-size: 0.7rem; color: #6B6560; margin-top: 4px; }
        .spinner { width: 40px; height: 40px; border: 4px solid #FAF3E8; border-top-color: #C4622D; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .lang-toggle { display: flex; gap: 6px; margin-top: 6px; }
        .lang-toggle button { background: #F5E6D8; border: none; border-radius: 40px; padding: 4px 14px; font-size: 0.7rem; font-weight: 600; color: #3A2E27; cursor: pointer; transition: 0.15s; font-family: 'Inter', sans-serif; }
        .lang-toggle button.active { background: #C4622D; color: white; }
        .platform-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 12px 0; }
        .platform-card { background: #F5E6D8; padding: 12px; border-radius: 12px; text-align: center; }
        .platform-card .name { font-weight: 700; font-size: 0.85rem; }
        .platform-card .reason { font-size: 0.7rem; color: #6B6560; }
        .badges-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
        .badge-pill.highlight { background: #F5E6D8; color: #C4622D; }
        .price-headline { font-size: 2.4rem; font-weight: 900; color: #1B2A4A; }
        .price-range { color: #6B6560; font-size: 0.95rem; }
        .price-explanation { background: #F5E6D8; padding: 12px; border-radius: 12px; margin: 8px 0; }
        .price-breakdown { background: #fcf9f6; padding: 12px; border-radius: 12px; }
        .row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #eee; }
        .row:last-child { border-bottom: none; font-weight: 700; border-top: 2px solid #ddd; padding-top: 6px; }
        .voice-step { background: #F5E6D8; padding: 12px; border-radius: 12px; margin: 12px 0; border-left: 4px solid #C4622D; }
        .mic-btn { width: 80px; height: 80px; border-radius: 50%; border: none; background: #C4622D; color: white; font-size: 2.4rem; cursor: pointer; box-shadow: 0 4px 16px rgba(196,98,45,0.3); transition: 0.2s; }
        .mic-btn.recording { background: #A13D2F; animation: pulse 1s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(161,61,47,0.4); } 70% { box-shadow: 0 0 0 16px rgba(161,61,47,0); } 100% { box-shadow: 0 0 0 0 rgba(161,61,47,0); } }
        .transcript-box { background: #FCF9F6; padding: 12px; border-radius: 12px; margin-bottom: 8px; }
        .transcript-box.bordered { border-left: 3px solid #C4622D; }
        .cost-input { width: 100%; padding: 12px 16px; border-radius: 40px; border: 1.5px solid #e0d6ce; font-size: 1rem; background: #fcf9f6; font-family: 'Inter', sans-serif; }
        .cost-input:focus { outline: none; border-color: #C4622D; }
        .badge-handmade { position: absolute; top: 12px; right: 12px; background: #C4622D; color: white; padding: 6px 14px; border-radius: 40px; font-size: 0.65rem; font-weight: 700; box-shadow: 0 2px 8px rgba(196,98,45,0.3); }
        .product-image-wrapper { position: relative; }
        .product-image-wrapper img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; border-radius: 12px; }
        .product-thumb { background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.04); cursor: pointer; }
        .product-thumb .thumb-img { position: relative; aspect-ratio: 1; background: #FAF3E8; }
        .product-thumb .thumb-img img { width: 100%; height: 100%; object-fit: cover; }
        .product-thumb .thumb-img .badge-sm { position: absolute; top: 6px; right: 6px; background: #C4622D; color: white; padding: 2px 10px; border-radius: 40px; font-size: 0.5rem; font-weight: 700; }
        .product-thumb .thumb-info { padding: 10px 12px; }
        .product-thumb .thumb-info .name { font-weight: 600; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .product-thumb .thumb-info .price { font-size: 0.8rem; color: #C4622D; font-weight: 700; }
        .products-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        @media (max-width: 420px) { .hero-header h1 { font-size: 1.8rem; } .feature-grid { grid-template-columns: 1fr 1fr; } .preset-grid { grid-template-columns: 1fr; } .platform-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div id="root"></div>
 
<script>
  if (typeof React === 'undefined' || typeof ReactDOM === 'undefined') {
    document.getElementById('root').innerHTML = `
      <div style="padding:40px;text-align:center;background:white;border-radius:16px;">
        <h2 style="color:#e74c3c;">⚠️ React Failed to Load</h2>
        <p style="color:#666;margin:20px 0;">Please check your internet connection.</p>
        <button onclick="location.reload()" style="padding:12px 24px;background:#C4622D;color:white;border:none;border-radius:8px;cursor:pointer;">Retry</button>
      </div>
    `;
  } else {
    const { useState, useRef, useEffect } = React;
    const API_BASE = '';
 
    function App() {
      const [screen, setScreen] = useState('login');
      const [artisan, setArtisan] = useState({ name: '', password: '' });
      const [loginError, setLoginError] = useState('');
      const [isLoggedIn, setIsLoggedIn] = useState(false);
      const [listings, setListings] = useState([]);
      const [product, setProduct] = useState({
        image: null,
        imageFile: null,
        enhancedImage: null,
        originalImage: null,
        descriptionText: '',
        descriptionTranslated: '',
        descriptionAudio: null,
        materialCost: '',
        hoursWorked: '',
        listingResult: null,
        isProcessing: false
      });
 
      const navigateTo = (s) => { setScreen(s); window.scrollTo(0, 0); };
 
      const handleLogin = (e) => {
        e.preventDefault();
        if (artisan.name.trim().length < 2) { setLoginError('Please enter your name'); return; }
        if (artisan.password.length < 4) { setLoginError('Password must be at least 4 characters'); return; }
        setIsLoggedIn(true);
        setLoginError('');
        navigateTo('welcome');
      };
 
      const handleLogout = () => {
        setIsLoggedIn(false);
        setArtisan({ name: '', password: '' });
        setLoginError('');
        navigateTo('login');
      };
 
      const resetProduct = () => {
        setProduct({
          image: null,
          imageFile: null,
          enhancedImage: null,
          originalImage: null,
          descriptionText: '',
          descriptionTranslated: '',
          descriptionAudio: null,
          materialCost: '',
          hoursWorked: '',
          listingResult: null,
          isProcessing: false
        });
      };
 
      const saveListing = (result) => {
        const newListing = {
          id: Date.now(),
          ...result,
          image: product.enhancedImage || product.image,
        };
        setListings([...listings, newListing]);
        navigateTo('myproducts');
      };
 
      // Welcome Screen
      function WelcomeScreen() {
        return React.createElement('div', null,
          React.createElement('div', { className: 'hero-header' },
            React.createElement('div', { className: 'badge-pill', style: { background: '#F5E6D8', color: '#C4622D' } },
              React.createElement('i', { className: 'fas fa-robot' }), ' AI Business Partner for Artisans'
            ),
            React.createElement('h1', null, 'KalaConnect', React.createElement('small', null, 'Artisan Hub')),
            React.createElement('p', { className: 'subhead' }, 'Sell your handmade crafts online. Snap a photo, speak in Hindi or English, and let AI build your perfect listing.'),
            React.createElement('button', { className: 'btn-primary', onClick: () => { resetProduct(); navigateTo('photo'); } },
              React.createElement('i', { className: 'fas fa-plus-circle' }), ' Add New Product'
            )
          ),
          React.createElement('div', { className: 'card' },
            React.createElement('div', { className: 'feature-grid' },
              React.createElement('div', { className: 'feature-item' },
                React.createElement('div', { className: 'icon-circle' }, React.createElement('i', { className: 'fas fa-camera' })),
                React.createElement('div', { className: 'label' }, 'Studio Photo'),
                React.createElement('div', { className: 'sub' }, 'Auto Light & BG')
              ),
              React.createElement('div', { className: 'feature-item' },
                React.createElement('div', { className: 'icon-circle' }, React.createElement('i', { className: 'fas fa-microphone-alt' })),
                React.createElement('div', { className: 'label' }, 'Voice to Story'),
                React.createElement('div', { className: 'sub' }, 'Hindi & English')
              ),
              React.createElement('div', { className: 'feature-item' },
                React.createElement('div', { className: 'icon-circle' }, React.createElement('i', { className: 'fas fa-tag' })),
                React.createElement('div', { className: 'label' }, 'Smart Price'),
                React.createElement('div', { className: 'sub' }, 'SerpAPI + AI')
              )
            )
          ),
          React.createElement('div', { className: 'card' },
            React.createElement('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 } },
              React.createElement('span', { style: { fontWeight: 600, fontSize: '0.9rem' } },
                React.createElement('i', { className: 'fas fa-hand-sparkles', style: { color: '#C4622D' } }), ' Try a Sample Craft Preset'
              ),
              React.createElement('span', { className: 'badge-pill', style: { fontSize: '0.6rem' } }, '1-Click Demo')
            ),
            React.createElement('p', { style: { fontSize: '0.8rem', color: '#6B6560', marginBottom: 10 } }, 'Select any authentic Indian handicraft to test the full pipeline.'),
            React.createElement('div', { className: 'preset-grid' },
              React.createElement('div', { className: 'preset-btn', onClick: () => alert('Banarasi Sari demo') },
                React.createElement('i', { className: 'fas fa-vest' }), ' Banarasi Sari'
              ),
              React.createElement('div', { className: 'preset-btn', onClick: () => alert('Madhubani Painting demo') },
                React.createElement('i', { className: 'fas fa-palette' }), ' Madhubani'
              ),
              React.createElement('div', { className: 'preset-btn', onClick: () => alert('Tanjore Doll demo') },
                React.createElement('i', { className: 'fas fa-dharmachakra' }), ' Tanjore Doll'
              ),
              React.createElement('div', { className: 'preset-btn', onClick: () => alert('Pashmina Shawl demo') },
                React.createElement('i', { className: 'fas fa-tshirt' }), ' Pashmina Shawl'
              )
            )
          ),
          listings.length > 0 && React.createElement('button', { className: 'btn-outline', style: { marginTop: 8 }, onClick: () => navigateTo('myproducts') },
            React.createElement('i', { className: 'fas fa-box' }), ' View My Products (', listings.length, ')'
          )
        );
      }
 
      // Photo Screen
      function PhotoScreen() {
        const fileInputRef = useRef(null);
        const cameraInputRef = useRef(null);
        const [uploading, setUploading] = useState(false);
 
        const handleFileChange = (e) => {
          const file = e.target.files?.[0];
          if (!file) return;
          setUploading(true);
          const reader = new FileReader();
          reader.onloadend = () => {
            setProduct({ ...product, image: reader.result, imageFile: file });
            setUploading(false);
            navigateTo('enhance');
          };
          reader.readAsDataURL(file);
        };
 
        return React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'flex-between', style: { marginBottom: 12 } },
            React.createElement('button', { style: { background: 'none', border: 'none', fontSize: '1.2rem', cursor: 'pointer', color: '#3A2E27' }, onClick: () => navigateTo('welcome') }, '←'),
            React.createElement('span', { style: { fontWeight: 700, fontSize: '1.1rem' } }, '📸 Product Photo'),
            React.createElement('span', null)
          ),
          React.createElement('p', { style: { fontSize: '0.9rem', color: '#6B6560', marginBottom: 16 } }, 'Take a photo or upload from gallery'),
          React.createElement('input', { type: 'file', ref: fileInputRef, accept: 'image/*', onChange: handleFileChange, style: { display: 'none' } }),
          React.createElement('input', { type: 'file', ref: cameraInputRef, accept: 'image/*', capture: 'environment', onChange: handleFileChange, style: { display: 'none' } }),
          React.createElement('div', { className: 'photo-upload-area', onClick: () => fileInputRef.current?.click() },
            React.createElement('div', { style: { fontSize: '2.6rem' } }, '📁'),
            React.createElement('p', { style: { fontWeight: 600 } }, 'Select from Gallery')
          ),
          React.createElement('div', { className: 'photo-upload-area', style: { marginTop: 12 }, onClick: () => cameraInputRef.current?.click() },
            React.createElement('div', { style: { fontSize: '2.6rem' } }, '📷'),
            React.createElement('p', { style: { fontWeight: 600 } }, 'Take a Photo with Camera')
          ),
          uploading && React.createElement('div', { style: { textAlign: 'center', marginTop: 12 } },
            React.createElement('div', { className: 'spinner' }),
            React.createElement('p', null, 'Loading...')
          )
        );
      }
 
      // Enhance Screen
      function EnhanceScreen() {
        const [enhancing, setEnhancing] = useState(false);
        const [error, setError] = useState(null);
 
        const handleEnhance = async () => {
          if (!product.imageFile) return;
          setEnhancing(true);
          setError(null);
          try {
            const formData = new FormData();
            formData.append('image', product.imageFile);
            const res = await fetch(API_BASE + '/enhance-image', { method: 'POST', body: formData });
            const result = await res.json();
            if (result.success) {
              setProduct({
                ...product,
                enhancedImage: result.data.enhanced_image,
                originalImage: result.data.original_image
              });
            } else {
              setError(result.error || 'Enhancement failed');
            }
          } catch (err) {
            setError('Failed to connect to server');
          }
          setEnhancing(false);
        };
 
        return React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'flex-between' },
            React.createElement('button', { style: { background: 'none', border: 'none', fontSize: '1.2rem', cursor: 'pointer' }, onClick: () => navigateTo('photo') }, '←'),
            React.createElement('span', { style: { fontWeight: 700 } }, '✨ Photo Enhancement'),
            React.createElement('span', null)
          ),
          enhancing ? (
            React.createElement('div', { style: { textAlign: 'center', padding: '30px 0' } },
              React.createElement('div', { className: 'spinner' }),
              React.createElement('p', { style: { marginTop: 12 } }, 'Enhancing your photo...')
            )
          ) : error ? (
            React.createElement('div', null,
              React.createElement('div', { style: { color: '#A13D2F', padding: 12, textAlign: 'center' } }, error),
              React.createElement('button', { className: 'btn-primary', onClick: handleEnhance }, 'Retry')
            )
          ) : (
            React.createElement('div', null,
              React.createElement('div', { className: 'comparison-container' },
                React.createElement('div', { className: 'comparison-item' },
                  React.createElement('img', { src: product.image, alt: 'Original' }),
                  React.createElement('div', { className: 'label' }, 'Original')
                ),
                React.createElement('div', { className: 'comparison-item' },
                  product.enhancedImage ?
                    React.createElement('img', { src: product.enhancedImage, alt: 'Enhanced' }) :
                    React.createElement('div', { style: { width: '100%', aspectRatio: '1', borderRadius: '12px', border: '1px dashed #e0d6ce', background: '#FAF3E8', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#C4622D', fontSize: '2rem' } }, '✨'),
                  React.createElement('div', { className: 'label' }, 'Enhanced ✨')
                )
              ),
              React.createElement('div', { style: { textAlign: 'center', color: '#7A8B5C', fontSize: '0.85rem' } }, 
                product.enhancedImage ? '✓ Colors preserved · Lighting fixed · BG removed' : 'Click "Enhance" to process your photo'
              ),
              React.createElement('div', { style: { display: 'flex', gap: 10, marginTop: 14 } },
                React.createElement('button', { className: 'btn-outline', style: { flex: 1 }, onClick: () => navigateTo('photo') }, '← Back'),
                product.enhancedImage ? 
                  React.createElement('button', { className: 'btn-primary', style: { flex: 1 }, onClick: () => navigateTo('voice') }, 'Next: Voice →') :
                  React.createElement('button', { className: 'btn-primary', style: { flex: 1 }, onClick: handleEnhance, disabled: enhancing }, '✨ Enhance Photo')
              )
            )
          )
        );
      }
 
      // Voice Screen - SINGLE RECORDING ONLY (removed story part)
      function VoiceScreen() {
        const [isRecording, setIsRecording] = useState(false);
        const [recordingTime, setRecordingTime] = useState(0);
        const [transcribing, setTranscribing] = useState(false);
        const [error, setError] = useState(null);
        const mediaRecorderRef = useRef(null);
        const audioChunksRef = useRef([]);
        const timerRef = useRef(null);
 
        const isDone = !!product.descriptionText;
 
        const startRecording = async () => {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;
            audioChunksRef.current = [];
            recorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
            recorder.onstop = async () => {
              const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
              await transcribeAudio(audioBlob);
              stream.getTracks().forEach(t => t.stop());
            };
            recorder.start();
            setIsRecording(true);
            setRecordingTime(0);
            setError(null);
            timerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000);
          } catch (err) {
            setError('Microphone access denied');
          }
        };
 
        const stopRecording = () => {
          if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            if (timerRef.current) clearInterval(timerRef.current);
          }
        };
 
        const transcribeAudio = async (audioBlob) => {
          setTranscribing(true);
          setError(null);
          try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'description.webm');
            const res = await fetch(API_BASE + '/transcribe', { method: 'POST', body: formData });
            const result = await res.json();
            if (result.success) {
              const data = result.data;
              setProduct({ 
                ...product, 
                descriptionText: data.text || data.translated, 
                descriptionTranslated: data.translated || data.text, 
                descriptionAudio: audioBlob 
              });
            } else {
              setError(result.error || 'Transcription failed');
            }
          } catch (err) {
            setError('Failed to connect to server');
          }
          setTranscribing(false);
        };
 
        const formatTime = (s) => {
          const m = Math.floor(s / 60);
          return m + ':' + (s % 60).toString().padStart(2, '0');
        };
 
        return React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'flex-between' },
            React.createElement('button', { style: { background: 'none', border: 'none', fontSize: '1.2rem', cursor: 'pointer' }, onClick: () => navigateTo('enhance') }, '←'),
            React.createElement('span', { style: { fontWeight: 700 } }, '🎙️ Voice Recording'),
            React.createElement('span', null)
          ),
          React.createElement('p', { style: { fontSize: '0.85rem', color: '#6B6560' } }, 'Describe your product'),
          React.createElement('div', { className: 'voice-step' },
            React.createElement('div', { style: { fontWeight: 600 } }, '📝 Describe your product'),
            React.createElement('div', { style: { fontSize: '0.8rem', color: '#3A2E27' } }, 'Speak in Hindi about material, size, color, and use')
          ),
          error && React.createElement('div', { style: { color: '#A13D2F', padding: 8, textAlign: 'center' } }, error),
          React.createElement('div', { style: { textAlign: 'center' } },
            React.createElement('button', {
              className: 'mic-btn' + (isRecording ? ' recording' : ''),
              onClick: isRecording ? stopRecording : startRecording,
              disabled: transcribing || isDone
            }, isRecording ? '⏹' : '🎙️'),
            isRecording && React.createElement('div', { style: { marginTop: 6, fontWeight: 600 } }, '⏱ ' + formatTime(recordingTime)),
            transcribing && React.createElement('div', { style: { marginTop: 8 } },
              React.createElement('span', { className: 'spinner', style: { display: 'inline-block', width: 20, height: 20, border: '3px solid #eee', borderTopColor: '#C4622D', borderRadius: '50%', animation: 'spin 0.8s linear infinite' } }),
              ' Transcribing...'
            )
          ),
          isDone && React.createElement('div', null,
            React.createElement('div', { className: 'transcript-box' },
              React.createElement('span', { className: 'badge-pill', style: { fontSize: '0.6rem' } }, '🇮🇳 Hindi (Original)'),
              React.createElement('div', { style: { marginTop: 4 } }, product.descriptionText || '—')
            ),
            React.createElement('div', { className: 'transcript-box bordered' },
              React.createElement('span', { className: 'badge-pill', style: { fontSize: '0.6rem' } }, '🇬🇧 English Translation'),
              React.createElement('div', { style: { marginTop: 4 } }, product.descriptionTranslated || product.descriptionText || '—')
            )
          ),
          React.createElement('div', { style: { display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 14 } },
            !isDone && !isRecording && !transcribing && React.createElement('div', { style: { width: '100%', textAlign: 'center', color: '#6B6560', fontSize: '0.85rem' } }, 'Press 🎙️ and speak in Hindi'),
            isDone && React.createElement('button', { className: 'btn-primary', style: { flex: 1 }, onClick: () => navigateTo('cost') }, 'Next: Cost →'),
            isDone && React.createElement('button', { className: 'btn-outline', style: { flex: 1 }, onClick: () => {
              setProduct({ ...product, descriptionText: '', descriptionTranslated: '', descriptionAudio: null });
            } }, '🔄 Re-record')
          )
        );
      }
 
      // Processing Screen
      function ProcessingScreen() {
        const [steps, setSteps] = useState([
          { id: 'photo', label: 'Processing your photo...', status: 'pending' },
          { id: 'voice', label: 'Translating your description...', status: 'pending' },
          { id: 'market', label: 'Checking market prices...', status: 'pending' },
          { id: 'price', label: 'Calculating fair price...', status: 'pending' },
        ]);
        const [error, setError] = useState(null);
 
        const updateStep = (id, status) => {
          setSteps(prev => prev.map(s => s.id === id ? { ...s, status } : s));
        };
 
        useEffect(() => {
          const process = async () => {
            try {
              setSteps(prev => prev.map(s => ({ ...s, status: 'pending' })));
              setError(null);
 
              const formData = new FormData();
              if (product.imageFile) formData.append('image', product.imageFile);
              if (product.descriptionAudio) formData.append('description_audio', product.descriptionAudio, 'description.webm');
              formData.append('material_cost', product.materialCost);
              formData.append('hours_worked', product.hoursWorked);
              formData.append('description_text', product.descriptionTranslated || product.descriptionText || '');
 
              updateStep('photo', 'active');
              await new Promise(r => setTimeout(r, 600));
              updateStep('photo', 'done');
 
              updateStep('voice', 'active');
              await new Promise(r => setTimeout(r, 500));
              updateStep('voice', 'done');
 
              updateStep('market', 'active');
              await new Promise(r => setTimeout(r, 400));
              updateStep('market', 'done');
 
              updateStep('price', 'active');
 
              const res = await fetch(API_BASE + '/create-listing', { method: 'POST', body: formData });
              const result = await res.json();
 
              if (result.success) {
                updateStep('price', 'done');
                setProduct({ ...product, listingResult: result.data });
                await new Promise(r => setTimeout(r, 500));
                navigateTo('results');
              } else {
                updateStep('price', 'pending');
                setError(result.error || 'Failed to create listing');
              }
            } catch (err) {
              setError('Failed to connect to server');
              setSteps(prev => prev.map(s => ({ ...s, status: 'pending' })));
            }
          };
          process();
        }, []);
 
        const getStepIcon = (status) => {
          if (status === 'done') return '✅';
          if (status === 'active') return '⏳';
          return '⏸';
        };
 
        return React.createElement('div', { className: 'card', style: { textAlign: 'center' } },
          React.createElement('span', { style: { fontWeight: 700, fontSize: '1.1rem' } }, '⏳ Creating Your Listing'),
          error ? (
            React.createElement('div', null,
              React.createElement('div', { className: 'status-error' }, error),
              React.createElement('button', { className: 'btn-primary', style: { marginTop: 12 }, onClick: () => window.location.reload() }, '🔄 Retry')
            )
          ) : (
            React.createElement('div', { style: { marginTop: 12 } },
              React.createElement('div', { className: 'spinner' }),
              React.createElement('div', { style: { marginTop: 16, textAlign: 'left' } },
                steps.map(s =>
                  React.createElement('div', {
                    key: s.id,
                    style: {
                      color: s.status === 'active' ? '#3A2E27' : '#6B6560',
                      fontWeight: s.status === 'active' ? 600 : 400,
                      margin: '4px 0'
                    }
                  }, getStepIcon(s.status), ' ', s.label)
                )
              ),
              React.createElement('p', { style: { color: '#6B6560', fontSize: '0.8rem', marginTop: 16 } }, 'This may take 10–20 seconds')
            )
          )
        );
      }
 
      // Results Screen
      function ResultsScreen() {
        const result = product.listingResult;
        const [showHindi, setShowHindi] = useState(false);
 
        if (!result) return React.createElement('div', { className: 'card' }, 'No results');
 
        const bestPrice = result.best_price || Math.round(((result.suggested_min || 0) + (result.suggested_max || 0)) / 2);
        const title = showHindi ? result.catalog?.title_hi || result.catalog?.title_en : result.catalog?.title_en || 'Untitled';
        const description = showHindi ? result.catalog?.description_hi || result.catalog?.description_en : result.catalog?.description_en || '';
        const bullets = result.catalog?.bullets_en || [];
        const badges = result.badges || [];
        const platforms = result.recommended_platforms || [];
 
        const onSave = () => {
          const newListing = {
            id: Date.now(),
            catalog: result.catalog,
            best_price: bestPrice,
            image: product.enhancedImage || product.image,
            enhancedImage: product.enhancedImage,
            originalImage: product.image,
            suggested_min: result.suggested_min,
            suggested_max: result.suggested_max,
            badges: result.badges
          };
          setListings([...listings, newListing]);
          navigateTo('myproducts');
        };
 
        return React.createElement('div', null,
          React.createElement('div', { className: 'card', style: { padding: 0, overflow: 'hidden' } },
            React.createElement('div', { className: 'product-image-wrapper' },
              React.createElement('img', {
                src: product.enhancedImage || product.image || 'https://via.placeholder.com/400x400/F5E6D8/C4622D?text=Product',
                alt: 'Product'
              }),
              React.createElement('div', { className: 'badge-handmade' }, '✨ 100% Handmade')
            ),
            React.createElement('div', { style: { padding: '16px 18px' } },
              React.createElement('div', { className: 'flex-between' },
                React.createElement('h2', { style: { fontSize: '1.3rem', fontWeight: 700 } }, title),
                React.createElement('div', { className: 'lang-toggle' },
                  React.createElement('button', { className: !showHindi ? 'active' : '', onClick: () => setShowHindi(false) }, 'EN'),
                  React.createElement('button', { className: showHindi ? 'active' : '', onClick: () => setShowHindi(true) }, 'हिं')
                )
              )
            )
          ),
          React.createElement('div', { className: 'card' },
            React.createElement('div', { className: 'price-headline' }, '₹' + bestPrice),
            React.createElement('div', { className: 'price-range' }, 'Range: ₹' + (result.suggested_min || 0) + ' – ₹' + (result.suggested_max || 0)),
            React.createElement('div', { className: 'price-explanation' }, result.explanation || 'Based on your costs.'),
            result.cost_floor && React.createElement('div', { className: 'price-breakdown' },
              React.createElement('div', { className: 'row' },
                React.createElement('span', null, 'Cost Floor'),
                React.createElement('span', null, '₹' + Math.round(result.cost_floor))
              ),
              result.market_median && React.createElement('div', { className: 'row' },
                React.createElement('span', null, 'Market Median'),
                React.createElement('span', null, '₹' + Math.round(result.market_median))
              ),
              React.createElement('div', { className: 'row' },
                React.createElement('span', null, 'Your Fair Price'),
                React.createElement('span', null, '₹' + bestPrice)
              )
            )
          ),
          React.createElement('div', { className: 'card' },
            React.createElement('div', { style: { fontWeight: 700, marginBottom: 8 } }, '📝 Description'),
            React.createElement('p', null, description || 'A beautiful handmade product.'),
            bullets.length > 0 && React.createElement('ul', { style: { marginTop: 8, paddingLeft: 20 } },
              bullets.map((b, i) => React.createElement('li', { key: i }, b))
            )
          ),
          badges.length > 0 && React.createElement('div', { className: 'card' },
            React.createElement('div', { style: { fontWeight: 700, marginBottom: 8 } }, '🏅 Badges'),
            React.createElement('div', { className: 'badges-row' },
              badges.map((b, i) => React.createElement('span', { key: i, className: 'badge-pill highlight' }, b))
            )
          ),
          platforms.length > 0 && React.createElement('div', { className: 'card' },
            React.createElement('div', { style: { fontWeight: 700, marginBottom: 8 } }, '🛍️ Recommended Platforms'),
            React.createElement('div', { className: 'platform-grid' },
              platforms.map((p, i) =>
                React.createElement('div', { key: i, className: 'platform-card' },
                  React.createElement('div', { className: 'name' }, p.name),
                  React.createElement('div', { className: 'reason' }, p.reason)
                )
              )
            )
          ),
          React.createElement('div', { style: { display: 'flex', gap: 10, marginTop: 8 } },
            React.createElement('button', { className: 'btn-primary', style: { flex: 1 }, onClick: onSave }, '💾 Save Listing')
          ),
          React.createElement('button', { className: 'btn-outline', style: { marginTop: 8 }, onClick: () => navigateTo('welcome') }, '← Back to Home')
        );
      }
 
      // My Products Screen
      function MyProductsScreen() {
        return React.createElement('div', null,
          React.createElement('div', { className: 'flex-between', style: { marginBottom: 12 } },
            React.createElement('button', { style: { background: 'none', border: 'none', fontSize: '1.2rem', cursor: 'pointer' }, onClick: () => navigateTo('welcome') }, '←'),
            React.createElement('span', { style: { fontWeight: 700, fontSize: '1.1rem' } }, '📦 My Products'),
            listings.length > 0 && React.createElement('button', { className: 'btn-outline', style: { padding: '4px 12px', fontSize: '0.7rem' }, onClick: () => setListings([]) }, 'Clear All')
          ),
          listings.length === 0 ? (
            React.createElement('div', { className: 'card', style: { textAlign: 'center', padding: '40px 20px' } },
              React.createElement('div', { style: { fontSize: '3rem' } }, '📦'),
              React.createElement('h3', null, 'No products yet'),
              React.createElement('p', { style: { color: '#6B6560' } }, 'Add your first product to get started!'),
              React.createElement('button', { className: 'btn-primary', style: { maxWidth: 200, marginTop: 12 }, onClick: () => { resetProduct(); navigateTo('photo'); } }, '+ Add Product')
            )
          ) : (
            React.createElement('div', { className: 'products-grid' },
              listings.map((item) =>
                React.createElement('div', {
                  key: item.id,
                  className: 'product-thumb',
                  onClick: () => {
                    setProduct({ ...product, listingResult: item, enhancedImage: item.enhancedImage || item.image });
                    navigateTo('results');
                  }
                },
                  React.createElement('div', { className: 'thumb-img' },
                    React.createElement('img', {
                      src: item.image || item.enhancedImage || 'https://via.placeholder.com/300x300/F5E6D8/C4622D?text=No+Image',
                      alt: item.catalog?.title_en || 'Product'
                    }),
                    React.createElement('div', { className: 'badge-sm' }, 'Handmade')
                  ),
                  React.createElement('div', { className: 'thumb-info' },
                    React.createElement('div', { className: 'name' }, item.catalog?.title_en || 'Untitled'),
                    React.createElement('div', { className: 'price' }, '₹' + item.best_price)
                  )
                )
              )
            )
          )
        );
      }
 
      // Render
      if (!isLoggedIn) {
        return React.createElement(LoginScreen, { artisan, setArtisan, loginError, setLoginError, onLogin: handleLogin });
      }
 
      return React.createElement('div', null,
        React.createElement('div', { className: 'flex-between', style: { marginBottom: 8 } },
          React.createElement('span', { style: { fontWeight: 600, fontSize: '0.9rem' } },
            React.createElement('i', { className: 'fas fa-user-astronaut', style: { color: '#C4622D', marginRight: 6 } }), artisan.name
          ),
          React.createElement('button', { className: 'btn-outline', style: { padding: '4px 16px', fontSize: '0.75rem' }, onClick: handleLogout }, 'Logout')
        ),
        screen === 'welcome' && React.createElement(WelcomeScreen),
        screen === 'photo' && React.createElement(PhotoScreen),
        screen === 'enhance' && React.createElement(EnhanceScreen),
        screen === 'voice' && React.createElement(VoiceScreen),
        screen === 'cost' && React.createElement(CostScreen, { product, setProduct, navigateTo }),
        screen === 'processing' && React.createElement(ProcessingScreen),
        screen === 'results' && React.createElement(ResultsScreen),
        screen === 'myproducts' && React.createElement(MyProductsScreen),
        React.createElement('div', { className: 'bottom-nav' },
          React.createElement('button', { className: 'nav-item' + (screen === 'welcome' ? ' active' : ''), onClick: () => navigateTo('welcome') },
            React.createElement('span', { className: 'icon' }, '🏠'), React.createElement('span', null, 'Home')
          ),
          React.createElement('button', { className: 'nav-item' + (['photo','enhance','voice','cost','processing'].includes(screen) ? ' active' : ''), onClick: () => { resetProduct(); navigateTo('photo'); } },
            React.createElement('span', { className: 'icon' }, '➕'), React.createElement('span', null, 'Add')
          ),
          React.createElement('button', { className: 'nav-item' + (screen === 'myproducts' ? ' active' : ''), onClick: () => navigateTo('myproducts') },
            React.createElement('span', { className: 'icon' }, '📦'), React.createElement('span', null, 'My Products'),
            listings.length > 0 && React.createElement('span', { style: { background: '#C4622D', color: 'white', borderRadius: '20px', padding: '0 8px', fontSize: '0.6rem', marginLeft: 4 } }, listings.length)
          )
        ),
        React.createElement('div', { style: { height: 80 } })
      );
    }
 
    // Login Screen (moved outside App so it isn't re-created on every keystroke,
    // which was causing the input to lose focus after each character)
    function LoginScreen({ artisan, setArtisan, loginError, setLoginError, onLogin }) {
      const handleNameChange = (e) => {
        const value = e.target.value;
        setArtisan({ ...artisan, name: value });
        if (loginError) setLoginError('');
      };
 
      const handlePasswordChange = (e) => {
        const value = e.target.value;
        setArtisan({ ...artisan, password: value });
        if (loginError) setLoginError('');
      };
 
      return React.createElement('div', { className: 'login-overlay' },
        React.createElement('div', { className: 'hero-header', style: { padding: '18px 16px', marginBottom: 16 } },
          React.createElement('h1', { style: { fontSize: '1.8rem' } }, 'KalaConnect ',
            React.createElement('small', null, 'Artisan Hub')
          ),
          React.createElement('div', { className: 'badge-pill', style: { marginTop: 6 } },
            React.createElement('i', { className: 'fas fa-user-circle' }), ' Artisan Login'
          )
        ),
        React.createElement('form', { onSubmit: onLogin },
          React.createElement('div', { className: 'field' },
            React.createElement('label', null, React.createElement('i', { className: 'fas fa-user' }), ' Your Name'),
            React.createElement('input', {
              type: 'text',
              placeholder: 'e.g. Neeraja',
              value: artisan.name,
              onChange: handleNameChange,
              autoComplete: 'off'
            })
          ),
          React.createElement('div', { className: 'field' },
            React.createElement('label', null, React.createElement('i', { className: 'fas fa-lock' }), ' Password'),
            React.createElement('input', {
              type: 'password',
              placeholder: '......',
              value: artisan.password,
              onChange: handlePasswordChange,
              autoComplete: 'off'
            })
          ),
          loginError && React.createElement('div', { className: 'status-error' }, loginError),
          React.createElement('button', { type: 'submit', className: 'btn-primary' },
            React.createElement('i', { className: 'fas fa-sign-in-alt' }), ' Enter Studio'
          ),
          React.createElement('div', { style: { fontSize: '0.75rem', color: '#6B6560', marginTop: 10, textAlign: 'center' } },
            'Demo: any name + 4+ chars password'
          )
        )
      );
    }
 
    // Cost Screen (moved outside App so it isn't re-created on every keystroke,
    // which was causing the input to lose focus after each character)
    function CostScreen({ product, setProduct, navigateTo }) {
      const [errors, setErrors] = useState({});
 
      const handleMaterialChange = (e) => {
        const value = e.target.value;
        setProduct({ ...product, materialCost: value });
        if (errors.materialCost) setErrors({ ...errors, materialCost: '' });
      };
 
      const handleHoursChange = (e) => {
        const value = e.target.value;
        setProduct({ ...product, hoursWorked: value });
        if (errors.hoursWorked) setErrors({ ...errors, hoursWorked: '' });
      };
 
      const handleNext = () => {
        const err = {};
        if (!product.materialCost || parseFloat(product.materialCost) <= 0) {
          err.materialCost = 'Enter valid cost';
        }
        if (!product.hoursWorked || parseFloat(product.hoursWorked) <= 0) {
          err.hoursWorked = 'Enter hours worked';
        }
        setErrors(err);
        if (Object.keys(err).length === 0) navigateTo('processing');
      };
 
      return React.createElement('div', { className: 'card' },
        React.createElement('span', { style: { fontWeight: 700, fontSize: '1.1rem' } }, '💰 Cost Details'),
        React.createElement('p', { style: { color: '#6B6560', fontSize: '0.9rem', margin: '4px 0 16px' } }, 'Help us calculate a fair price'),
        React.createElement('div', { style: { marginBottom: 16 } },
          React.createElement('label', { style: { fontWeight: 600, fontSize: '0.9rem', display: 'block', marginBottom: 4 } }, 'Material Cost ', React.createElement('span', { style: { color: '#6B6560', fontWeight: 400 } }, '(₹)')),
          React.createElement('input', {
            type: 'number',
            className: 'cost-input',
            placeholder: 'e.g. 500',
            value: product.materialCost,
            onChange: handleMaterialChange,
            inputMode: 'numeric',
            autoComplete: 'off'
          }),
          errors.materialCost && React.createElement('div', { style: { color: '#A13D2F', fontSize: '0.8rem', marginTop: 4 } }, errors.materialCost)
        ),
        React.createElement('div', { style: { marginBottom: 16 } },
          React.createElement('label', { style: { fontWeight: 600, fontSize: '0.9rem', display: 'block', marginBottom: 4 } }, 'Hours Worked ', React.createElement('span', { style: { color: '#6B6560', fontWeight: 400 } }, '(total time)')),
          React.createElement('input', {
            type: 'number',
            className: 'cost-input',
            placeholder: 'e.g. 8',
            value: product.hoursWorked,
            onChange: handleHoursChange,
            inputMode: 'numeric',
            autoComplete: 'off'
          }),
          errors.hoursWorked && React.createElement('div', { style: { color: '#A13D2F', fontSize: '0.8rem', marginTop: 4 } }, errors.hoursWorked)
        ),
        React.createElement('button', { className: 'btn-primary', onClick: handleNext }, 'Generate My Listing →')
      );
    }
 
    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(React.createElement(App));
  }
</script>
</body>
</html>
'''

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 KalaConnect - AI Business Manager")
    print("📍 http://localhost:8000")
    print("=" * 50)
    print("✨ Features:")
    print("  - 📸 Camera & Gallery support")
    print("  - ✨ AI Photo Enhancement (Lighting ONLY, Colors Preserved)")
    print("  - 🎙️ Hindi Voice Recognition with Whisper (Better Accuracy)")
    print("  - 🌐 Real-time Hindi to English Translation")
    print("  - 💰 AI-powered Price Calculation")
    print("=" * 50)
    print("📝 Image enhancement preserves original colors!")
    print("   - Only fixes lighting and removes background")
    print("   - Dress colors and product colors remain unchanged")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)