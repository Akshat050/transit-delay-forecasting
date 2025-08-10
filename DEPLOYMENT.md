# 🚀 Deployment Guide for DelaySenseAI

This guide covers deploying DelaySenseAI to various platforms so it can be used by everyone.

## 🌟 Quick Deploy Options

### 1. Streamlit Cloud (Recommended - Free)
**Best for**: Quick deployment, automatic updates, no server management

1. **Fork this repository** on GitHub
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
3. **Sign in with GitHub**
4. **Click "New app"**
5. **Select your forked repository**
6. **Set the path to dashboard.py**
7. **Click "Deploy"**

Your app will be live at: `https://yourusername-delaysenseai.streamlit.app`

### 2. Heroku (Free tier discontinued)
**Best for**: Custom domains, more control

1. **Install Heroku CLI**
2. **Login to Heroku**:
   ```bash
   heroku login
   ```
3. **Create Heroku app**:
   ```bash
   heroku create your-delaysenseai-app
   ```
4. **Deploy**:
   ```bash
   git push heroku main
   ```

### 3. Docker (Self-hosted)
**Best for**: Full control, custom infrastructure

1. **Build and run**:
   ```bash
   docker build -t delaysenseai .
   docker run -p 8501:8501 delaysenseai
   ```

2. **Using Docker Compose**:
   ```bash
   docker-compose up -d
   ```

## 📋 Prerequisites

### For All Deployments
- ✅ Python 3.11+
- ✅ All dependencies in `requirements.txt`
- ✅ GTFS data files (or sample data)
- ✅ Trained model files

### For Streamlit Cloud
- ✅ GitHub account
- ✅ Repository with all files
- ✅ `dashboard.py` as main entry point

### For Heroku
- ✅ Heroku account
- ✅ Heroku CLI installed
- ✅ Git repository

### For Docker
- ✅ Docker installed
- ✅ Docker Compose (optional)

## 🔧 Configuration

### Environment Variables
Set these in your deployment platform:

```bash
# Required
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Optional
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
```

### Streamlit Configuration
Create `.streamlit/config.toml`:

```toml
[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false
```

## 📁 File Structure for Deployment

Ensure your repository has this structure:

```
delaysenseai/
├── dashboard.py           # Main Streamlit app
├── data_processor.py     # Data processing
├── delay_predictor.py    # ML model
├── visualizer.py         # Visualizations
├── transfer_processor.py # Transfer analysis
├── calibration.py        # Model calibration
├── scenario_utils.py     # Scenario simulation
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── LICENSE              # License
├── .gitignore          # Git ignore
├── Procfile            # Heroku deployment
├── setup.sh            # Heroku setup
├── runtime.txt          # Python version
├── Dockerfile           # Docker deployment
├── docker-compose.yml   # Docker compose
└── .github/             # GitHub Actions
    └── workflows/
        └── deploy.yml
```

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: DelaySenseAI deployment ready"
   ```

2. **Create GitHub repository**:
   - Go to GitHub.com
   - Click "New repository"
   - Name it `delaysenseai`
   - Make it public
   - Don't initialize with README (we already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/yourusername/delaysenseai.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Fork the repository** (if you want to contribute back)
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
3. **Sign in with GitHub**
4. **Click "New app"**
5. **Configure**:
   - **Repository**: Select your repository
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
   - **App URL**: Customize if desired
6. **Click "Deploy"**

### Step 3: Test Your Deployment

1. **Wait for deployment** (usually 2-5 minutes)
2. **Test the app**:
   - Load a model
   - Navigate through tabs
   - Test visualizations
3. **Check for errors** in the Streamlit Cloud logs

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: Module not found errors
**Solution**: Ensure all Python files are in the repository root

#### 2. Missing Dependencies
**Problem**: Package installation failures
**Solution**: Check `requirements.txt` and update versions if needed

#### 3. Memory Issues
**Problem**: App crashes due to memory
**Solution**: Use smaller datasets or optimize data processing

#### 4. Model Loading Errors
**Problem**: Can't load trained models
**Solution**: Ensure model files are in the repository or use sample data

### Debug Steps

1. **Check Streamlit Cloud logs** for error messages
2. **Test locally** first: `streamlit run dashboard.py`
3. **Verify file paths** and imports
4. **Check Python version** compatibility

## 🌐 Custom Domain & HTTPS

### Streamlit Cloud
- **Custom domain**: Not supported in free tier
- **HTTPS**: Automatically provided
- **SSL**: Managed by Streamlit

### Heroku
- **Custom domain**: Supported
- **HTTPS**: Automatically provided
- **SSL**: Managed by Heroku

### Docker
- **Custom domain**: Configure in reverse proxy
- **HTTPS**: Use Let's Encrypt or similar
- **SSL**: Manual configuration required

## 📊 Monitoring & Analytics

### Streamlit Cloud
- **Usage statistics**: Built-in analytics
- **Performance**: Automatic monitoring
- **Logs**: Real-time log access

### Heroku
- **Metrics**: Heroku dashboard
- **Logs**: `heroku logs --tail`
- **Performance**: New Relic integration

### Docker
- **Monitoring**: Docker stats, Prometheus
- **Logs**: Docker logs
- **Health checks**: Built-in health check

## 🔄 Continuous Deployment

### GitHub Actions
The included workflow automatically:
- Tests code on push/PR
- Validates dependencies
- Prepares for deployment

### Streamlit Cloud
- **Auto-deploy**: On every push to main branch
- **Manual deploy**: From dashboard
- **Rollback**: Previous versions available

## 💰 Cost Considerations

### Free Options
- **Streamlit Cloud**: Free tier available
- **GitHub Pages**: Free hosting
- **Netlify**: Free tier available

### Paid Options
- **Heroku**: $7/month minimum
- **AWS/GCP**: Pay-per-use
- **DigitalOcean**: $5/month minimum

## 🎯 Production Considerations

### Security
- **Environment variables**: Don't commit secrets
- **Data validation**: Validate all inputs
- **Rate limiting**: Implement if needed

### Performance
- **Caching**: Cache model predictions
- **Data size**: Optimize for large datasets
- **Concurrent users**: Test with multiple users

### Maintenance
- **Updates**: Regular dependency updates
- **Monitoring**: Set up alerts for errors
- **Backups**: Backup models and data

## 📞 Getting Help

- **GitHub Issues**: Report deployment problems
- **Streamlit Community**: [community.streamlit.io](https://community.streamlit.io)
- **Documentation**: Check platform-specific docs
- **Community**: Ask in GitHub Discussions

## 🎉 Success!

Once deployed, your DelaySenseAI will be:
- 🌐 **Accessible worldwide** via web browser
- 📱 **Mobile-friendly** responsive design
- 🔄 **Auto-updating** with code changes
- 📊 **Production-ready** for real users

Share your deployment URL and help make transit systems better everywhere! 🚌 