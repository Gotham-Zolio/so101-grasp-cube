# 安装真机部署所需的所有依赖 (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Green
Write-Host "安装 DiffusionPolicy 真机部署依赖" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# 1. 安装 env_client 库
Write-Host ""
Write-Host "[1/3] 安装 env_client 库..." -ForegroundColor Cyan

if (Test-Path "packages/env-client") {
    Write-Host "运行: uv pip install -e packages/env-client"
    uv pip install -e packages/env-client
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ env_client 已安装" -ForegroundColor Green
    } else {
        Write-Host "❌ env_client 安装失败" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ 找不到 packages/env-client 目录" -ForegroundColor Red
    exit 1
}

# 2. 安装 LeRobot 库
Write-Host ""
Write-Host "[2/3] 安装 LeRobot 库..." -ForegroundColor Cyan

Write-Host "运行: pip install lerobot[compute_metrics]"
pip install "lerobot[compute_metrics]"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ LeRobot 已安装" -ForegroundColor Green
} else {
    Write-Host "❌ LeRobot 安装失败" -ForegroundColor Red
    exit 1
}

# 3. 验证安装
Write-Host ""
Write-Host "[3/3] 验证安装..." -ForegroundColor Cyan

try {
    python -c "import env_client; print('✅ env_client 导入成功')"
} catch {
    Write-Host "❌ env_client 导入失败" -ForegroundColor Red
}

try {
    python -c "import lerobot; print('✅ lerobot 导入成功')"
} catch {
    Write-Host "❌ lerobot 导入失败" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ 所有依赖安装完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "现在可以运行服务器："
Write-Host "  uv run python grasp_cube/real/serve_diffusion_policy.py \" -ForegroundColor Yellow
Write-Host "      --policy.path checkpoints/lift_real/checkpoint-best \" -ForegroundColor Yellow
Write-Host "      --policy.task lift" -ForegroundColor Yellow
