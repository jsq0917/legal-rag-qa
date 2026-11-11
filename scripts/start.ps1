# LawAnswer 一键启动：容器(MySQL/Redis/Milvus/MinIO) -> 建库/模型/入库/后端(8101) -> 前端 Vite(5180)
# 用法：powershell -ExecutionPolicy Bypass -File scripts/start.ps1
# 说明：本脚本只做「编排」，真正的启动逻辑仍是仓库自带的 scripts/bootstrap.ps1（幂等，已完成的步骤会跳过）。
#       透传参数给 bootstrap，例如：scripts/start.ps1 -SkipModels -SkipIngest

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BootstrapArgs
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # 关掉 Invoke-WebRequest 的进度条噪音
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# conda 环境常把 SSL_CERT_DIR / SSL_CERT_FILE 指到空目录或不存在的文件，uv 会往
# stderr 打告警；在 $ErrorActionPreference='Stop' 下这类 native stderr 可能被当成
# 终止错误，导致启动在半路失败。这里先丢掉无效值。
foreach ($name in @('SSL_CERT_DIR', 'SSL_CERT_FILE')) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if (-not $value) { continue }
    $item = Get-Item -LiteralPath $value -ErrorAction SilentlyContinue
    $valid = $false
    if ($item) {
        if ($item.PSIsContainer) {
            $valid = [bool](Get-ChildItem -LiteralPath $value -File -ErrorAction SilentlyContinue | Select-Object -First 1)
        } else {
            $valid = $true
        }
    }
    if (-not $valid) {
        Write-Host "      忽略无效的 $name=$value" -ForegroundColor DarkGray
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

$Bootstrap = Join-Path $Root 'scripts\bootstrap.ps1'
if (-not (Test-Path $Bootstrap)) {
    Write-Host "缺少 scripts\bootstrap.ps1，无法继续。" -ForegroundColor Red
    exit 1
}

# 前置：Docker Desktop 必须在运行（bootstrap 第一步就要起容器）
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker 未就绪：请先启动 Docker Desktop，等托盘图标变绿后重跑。" -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] 执行官方 bootstrap（依赖 -> 容器 -> 建库 -> 模型 -> 入库 -> 后端 8101）..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File $Bootstrap @BootstrapArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "bootstrap 未成功结束（退出码 $LASTEXITCODE），前端不再启动。请看上方步骤输出。" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

$Front = Join-Path $Root 'frontend'
if (-not (Test-Path (Join-Path $Front 'package.json'))) {
    Write-Host "未找到 frontend\package.json，后端已就绪但前端未启动。" -ForegroundColor Yellow
    exit 1
}

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $npm) {
    Write-Host "未找到 npm：请安装 Node.js 并确认在 PATH 中，然后重跑。" -ForegroundColor Red
    exit 1
}

$FrontUrl = 'http://localhost:5180'
$portBusy = $false
$probe = New-Object System.Net.Sockets.TcpClient
try {
    $iar = $probe.BeginConnect('127.0.0.1', 5180, $null, $null)
    $portBusy = $iar.AsyncWaitHandle.WaitOne(1000, $false) -and $probe.Connected
} catch { $portBusy = $false } finally { $probe.Close() }

if ($portBusy) {
    Write-Host "[2/2] 前端端口 5180 已在监听，假定前端已在运行" -ForegroundColor Yellow
} else {
    Write-Host "[2/2] 启动前端: $FrontUrl" -ForegroundColor Cyan
    if (-not (Test-Path (Join-Path $Front 'node_modules'))) {
        Write-Host "      首次运行，先 npm install（约 1~2 分钟）..." -ForegroundColor Cyan
        Push-Location $Front
        & $npm.Source install
        $installCode = $LASTEXITCODE
        Pop-Location
        if ($installCode -ne 0) {
            Write-Host "npm install 失败（退出码 $installCode），请查看上方输出。" -ForegroundColor Red
            exit $installCode
        }
    }
    $logDir = Join-Path $Root 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $outLog = Join-Path $logDir 'frontend.log'
    $errLog = Join-Path $logDir 'frontend.err.log'
    Start-Process -FilePath $npm.Source -ArgumentList 'run', 'dev' -WorkingDirectory $Front `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    $deadline = (Get-Date).AddSeconds(60)
    $frontReady = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $FrontUrl -TimeoutSec 3 -UseBasicParsing
            if ($resp.StatusCode -ge 200) { $frontReady = $true; break }
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $frontReady) {
        Write-Host "前端未在 60s 内就绪，日志：$errLog" -ForegroundColor Red
        Write-Host "常见原因：端口 5180 被占用、node_modules 不完整、npm 不在 PATH。" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "启动完成：" -ForegroundColor Green
Write-Host "  前端      $FrontUrl" -ForegroundColor Green
Write-Host "  后端 API  http://127.0.0.1:8101/docs" -ForegroundColor Green
Write-Host "  健康检查  http://127.0.0.1:8101/api/health/live" -ForegroundColor Green
Write-Host "  前端日志  logs\frontend.log（错误另见 logs\frontend.err.log）" -ForegroundColor DarkGray
Write-Host "停止：关闭各窗口；容器用 docker compose down。" -ForegroundColor Green
