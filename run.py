# ============================================
# run.py - Single command launcher
# ============================================
import subprocess
import sys
import os

def main():
    print("🚀 Starting KalaConnect...")
    print("📦 Installing dependencies...")
    
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("✅ Dependencies installed!")
    print("🔥 Starting server...")
    
    # Run the app
    subprocess.check_call([sys.executable, "main.py"])

if __name__ == "__main__":
    main()