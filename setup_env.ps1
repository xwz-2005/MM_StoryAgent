# ========================================
# MM-StoryAgent Environment Variables Setup
# Usage: .\setup_env.ps1
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MM-StoryAgent Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ====================================
# 1. DashScope API Configuration
# ====================================
Write-Host "Setting up DashScope API..." -ForegroundColor Yellow

$env:DASHSCOPE_API_KEY = "sk-c5574beb924a4d9fbedd573f681090b0"

Write-Host "  [OK] DASHSCOPE_API_KEY set" -ForegroundColor Green
Write-Host ""

# ====================================
# 2. Aliyun Speech API Configuration
# ====================================
Write-Host "Setting up Aliyun Speech API..." -ForegroundColor Yellow

$env:ALIYUN_APP_KEY = *
Write-Host "  [OK] ALIYUN_APP_KEY set" -ForegroundColor Green

$env:ALIYUN_ACCESS_KEY_ID = *
Write-Host "  [OK] ALIYUN_ACCESS_KEY_ID set" -ForegroundColor Green

$env:ALIYUN_ACCESS_KEY_SECRET = *
Write-Host "  [OK] ALIYUN_ACCESS_KEY_SECRET set" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All environment variables set!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Configuration Summary:" -ForegroundColor Cyan
Write-Host "  * DASHSCOPE_API_KEY: sk-c5574beb924a..." -ForegroundColor Gray
Write-Host "  * ALIYUN_APP_KEY: AzX8QcMX3u1809OI" -ForegroundColor Gray
Write-Host "  * ALIYUN_ACCESS_KEY_ID: LTAI5tFsk7..." -ForegroundColor Gray
Write-Host "  * ALIYUN_ACCESS_KEY_SECRET: [hidden]" -ForegroundColor Gray
Write-Host ""

Write-Host "Next step: Run the program" -ForegroundColor Cyan
Write-Host "  python run.py -c configs/mm_story_agent_api.yaml" -ForegroundColor Yellow
Write-Host ""
