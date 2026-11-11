# KalaConnect: AI-Driven Market Linkage & Smart Cataloging Platform

An AI-powered cross-platform mobile application developed for Smart India Hackathon (SIH 2026), built to empower marginalized artisans by digitizing handmade products, bypassing middlemen, and linking directly with global B2B buyers[cite: 4, 6].

## 🚀 Features

- **AI Image Studio:** Utilizes OpenCV for automatic background removal, lighting enhancement, and professional e-commerce formatting[cite: 4, 6].
- **Multilingual Voice Cataloger:** Powered by Open Whisper AI and LLMs, allowing artisans to describe products in their regional language to automatically generate English titles and descriptions[cite: 4, 6].
- **Dynamic Pricing Assistant:** Fetches real-time market data via SerpAPI, factoring in material costs and labor to suggest fair selling prices[cite: 4, 6].
- **B2B Market Linkage:** Instantly alerts registered B2B buyers via SMS/WhatsApp when new artisan catalogs are published[cite: 4, 6].

## 🛠️ Tech Stack

- **Frontend:** React Native (Cross-platform mobile application)[cite: 4, 6]
- **Backend:** Python, FastAPI, Asynchronous REST APIs[cite: 4, 6]
- **AI & Data Services:** OpenCV (Image Processing), Open Whisper AI (Speech-to-Text), LLMs, SerpAPI (Market Trends)[cite: 4, 6]
- **Database:** PostgreSQL / Firebase Firestore[cite: 4, 6]

## 💡 System Workflow

1. **Onboarding & Capture:** Artisans sign in and capture raw product photos via mobile camera[cite: 5, 7].
2. **AI Processing:** OpenCV cleans the photo[cite: 4, 6], while Whisper AI transcribes the artisan's voice note[cite: 4, 6].
3. **Smart Pricing & Review:** SerpAPI suggests optimal pricing[cite: 4, 6], and the artisan reviews the final auto-generated listing[cite: 5, 7].
4. **Publishing:** Approved products trigger instant B2B buyer alerts[cite: 4, 5, 6, 7].
