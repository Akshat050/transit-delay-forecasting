#!/usr/bin/env python3
"""
DelaySenseAI Deployment Script
Helps deploy the application to various platforms
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_prerequisites():
    """Check if all required files exist"""
    required_files = [
        'dashboard.py',
        'data_processor.py', 
        'delay_predictor.py',
        'visualizer.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def check_git_status():
    """Check git repository status"""
    try:
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository found")
            return True
        else:
            print("❌ Not a git repository")
            return False
    except FileNotFoundError:
        print("❌ Git not installed")
        return False

def create_streamlit_config():
    """Create Streamlit configuration file"""
    config_dir = Path('.streamlit')
    config_dir.mkdir(exist_ok=True)
    
    config_content = """[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false
"""
    
    config_file = config_dir / 'config.toml'
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ Created .streamlit/config.toml")

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import xgboost
        import folium
        import plotly
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def deploy_streamlit_cloud():
    """Deploy to Streamlit Cloud"""
    print("\n🚀 Deploying to Streamlit Cloud...")
    print("\nFollow these steps:")
    print("1. Go to https://streamlit.io/cloud")
    print("2. Sign in with GitHub")
    print("3. Click 'New app'")
    print("4. Select your repository")
    print("5. Set main file path to: dashboard.py")
    print("6. Click 'Deploy'")
    
    # Check if repository is on GitHub
    try:
        result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
        if 'github.com' in result.stdout:
            print("\n✅ GitHub remote found")
            print("Your app will be available at: https://yourusername-delaysenseai.streamlit.app")
        else:
            print("\n⚠️  No GitHub remote found")
            print("Make sure to push your code to GitHub first")
    except:
        pass

def deploy_heroku():
    """Deploy to Heroku"""
    print("\n🚀 Deploying to Heroku...")
    
    # Check if Heroku CLI is installed
    try:
        result = subprocess.run(['heroku', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Heroku CLI found")
            
            # Check if app exists
            app_name = input("Enter Heroku app name (or press Enter to create new): ").strip()
            
            if not app_name:
                app_name = f"delaysenseai-{os.getenv('USERNAME', 'app')}"
                print(f"Creating app: {app_name}")
                subprocess.run(['heroku', 'create', app_name])
            else:
                print(f"Using existing app: {app_name}")
            
            # Deploy
            print("Deploying to Heroku...")
            subprocess.run(['git', 'push', 'heroku', 'main'])
            
            print(f"✅ App deployed to: https://{app_name}.herokuapp.com")
            
        else:
            print("❌ Heroku CLI not working properly")
    except FileNotFoundError:
        print("❌ Heroku CLI not installed")
        print("Install from: https://devcenter.heroku.com/articles/heroku-cli")

def deploy_docker():
    """Deploy using Docker"""
    print("\n🐳 Deploying with Docker...")
    
    # Check if Docker is installed
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker found")
            
            # Build image
            print("Building Docker image...")
            subprocess.run(['docker', 'build', '-t', 'delaysenseai', '.'])
            
            # Run container
            print("Starting container...")
            subprocess.run(['docker', 'run', '-d', '-p', '8501:8501', '--name', 'delaysenseai-app', 'delaysenseai'])
            
            print("✅ App running at: http://localhost:8501")
            print("Stop with: docker stop delaysenseai-app")
            
        else:
            print("❌ Docker not working properly")
    except FileNotFoundError:
        print("❌ Docker not installed")
        print("Install from: https://docs.docker.com/get-docker/")

def main():
    """Main deployment function"""
    print("🚌 DelaySenseAI Deployment Script")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Create Streamlit config
    create_streamlit_config()
    
    # Check git status
    if not check_git_status():
        print("\n⚠️  Please initialize git repository first:")
        print("git init")
        print("git add .")
        print("git commit -m 'Initial commit'")
        print("git remote add origin https://github.com/yourusername/delaysenseai.git")
        print("git push -u origin main")
    
    # Deployment options
    print("\n🌐 Choose deployment option:")
    print("1. Streamlit Cloud (Recommended - Free)")
    print("2. Heroku")
    print("3. Docker (Local)")
    print("4. All options")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        deploy_streamlit_cloud()
    elif choice == '2':
        deploy_heroku()
    elif choice == '3':
        deploy_docker()
    elif choice == '4':
        deploy_streamlit_cloud()
        deploy_heroku()
        deploy_docker()
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    print("\n🎉 Deployment complete!")
    print("\n📚 For more details, see DEPLOYMENT.md")
    print("🌐 Share your app URL with the world!")

if __name__ == "__main__":
    main() 