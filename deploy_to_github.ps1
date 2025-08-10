# GitHub Deployment Script for Transit Delay Forecasting System
# Run this script after creating your GitHub repository

Write-Host "🚀 GitHub Deployment Script for Transit Delay Forecasting System" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green
Write-Host ""

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Error: Not in a Git repository. Please run 'git init' first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Git repository found!" -ForegroundColor Green
Write-Host ""

# Check current branch
$currentBranch = git branch --show-current
Write-Host "📍 Current branch: $currentBranch" -ForegroundColor Yellow

# Check remote status
$remotes = git remote -v
if ($remotes) {
    Write-Host "✅ Remote repositories configured:" -ForegroundColor Green
    Write-Host $remotes -ForegroundColor Cyan
} else {
    Write-Host "❌ No remote repositories configured" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 To connect to GitHub, you need to:" -ForegroundColor Yellow
    Write-Host "   1. Create a repository on GitHub.com" -ForegroundColor White
    Write-Host "   2. Run: git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git" -ForegroundColor Cyan
    Write-Host "   3. Run: git push -u origin $currentBranch" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Yellow
Write-Host "==============" -ForegroundColor Yellow
Write-Host "1. Go to GitHub.com and create a new repository" -ForegroundColor White
Write-Host "2. Copy the repository URL" -ForegroundColor White
Write-Host "3. Run the commands shown above with your actual repository URL" -ForegroundColor White
Write-Host "4. Your project will be uploaded to GitHub!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Pro Tip: After uploading, GitHub will automatically deploy your Streamlit app" -ForegroundColor Cyan
Write-Host "   if you have the repository connected to Streamlit Cloud!" -ForegroundColor Cyan 