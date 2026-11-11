<#
.SYNOPSIS
    LawAnswer 一键启动：容器 -> 建库 -> 下载模型 -> 建知识库 -> 入库 -> 起服务。

.DESCRIPTION
    面向「删掉容器与库后从零跑通」的验收场景，全程幂等：重复执行会跳过已完成步骤。
    步骤编号与 README「5 分钟跑通」一一对应，失败时会停在当前步骤并给出下一步建议。

    前置要求：Docker Desktop 已启动、uv 已安装（irm https://astral.sh/uv/install.ps1 | iex）。
    密钥请提前写入系统环境变量（DeepSeek 示例）：
        setx DEEPSEEK_API_KEY sk-xxxx
    密钥不允许写进 .env，启动自检会拒绝（config.assert_secrets_not_in_dotenv）。

.PARAMETER SkipModels
    跳过模型下载（已手工放置模型到 models/ 时使用）。

.PARAMETER SkipIngest
    跳过建库与入库（只想起服务时用）。

.PARAMETER WithEval
    额外安装 eval 依赖组（ragas 等，约 200MB，仅跑评估时需要）。

.PARAMETER SkipSync
    跳过 uv sync（已装好依赖、只想快速起服务时用）。

.PARAMETER NoStart
    做完数据准备后不启动后端（例如要在 IDE 里启动）。

.PARAMETER Reset
    重建 Milvus collection（--drop），会清空已有向量，慎用。

.EXAMPLE
    .\scripts\bootstrap.ps1
    .\scripts\bootstrap.ps1 -SkipModels -SkipIngest
#>
[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$SkipIngest,
    [switch]$WithEval,
    [switch]$SkipSync,
    [switch]$NoStart,
    [switch]$Reset
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# 中文 Windows 控制台默认 GBK，子进程（uv run python）输出的 UTF-8 中文会被解成乱码。
# 这里统一成 UTF-8，保证步骤日志里的中文可读。
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------
$script:StepNo = 0
$script:TotalStart = Get-Date

function Write-Step {
    param([string]$Message)
    $script:StepNo++
    Write-Host ""
    Write-Host ("[step {0}] {1}" -f $script:StepNo, $Message) -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host ("  ok   {0}" -f $Message) -ForegroundColor Green
}

function Write-Skip {
    param([string]$Message)
    Write-Host ("  skip {0}" -f $Message) -ForegroundColor DarkGray
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host ("  warn {0}" -f $Message) -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host ("  FAIL {0}" -f $Message) -ForegroundColor Red
}

function Step-Timer {
    param([scriptblock]$Block)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Block
    $sw.Stop()
    Write-Host ("  ({0:N1}s)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
}

function Test-TcpPort {
    param([string]$HostName = '127.0.0.1', [int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(1500, $false)
        if ($ok) { return $client.Connected } else { return $false }
    } catch {
        return $false
    } finally {
        $client.Close()
        $client.Dispose()
    }
}

function Wait-ForPort {
    param([int]$Port, [string]$Label, [int]$TimeoutSec = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-TcpPort -Port $Port) { Write-Ok "$Label 端口 $Port 已就绪"; return $true }
        Start-Sleep -Seconds 3
    }
    Write-Fail "$Label 端口 $Port 在 ${TimeoutSec}s 内未就绪"
    return $false
}

function Invoke-JsonPost {
    param([string]$Uri, [object]$Body, [int]$TimeoutSec = 60, [string]$Token)
    $json = ($Body | ConvertTo-Json -Depth 8 -Compress)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $headers = @{}
    if ($Token) { $headers['Authorization'] = "Bearer $Token" }
    return Invoke-RestMethod -Method Post -Uri $Uri -Body $bytes `
        -ContentType 'application/json; charset=utf-8' -TimeoutSec $TimeoutSec -Headers $headers
}

function Invoke-JsonGet {
    param([string]$Uri, [int]$TimeoutSec = 60, [string]$Token)
    $headers = @{}
    if ($Token) { $headers['Authorization'] = "Bearer $Token" }
    return Invoke-RestMethod -Method Get -Uri $Uri -TimeoutSec $TimeoutSec -Headers $headers
}

# ---------------------------------------------------------------------------
# 环境与路径
# ---------------------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
Write-Host "LawAnswer 一键启动" -ForegroundColor Magenta
Write-Host "仓库目录：$RepoRoot"

# 从 .env 读取端口，避免与 README/设计文档里的偏移端口脱节
function Get-EnvValue {
    param([string]$Key, [string]$Default)
    $path = Join-Path $RepoRoot '.env'
    if (Test-Path $path) {
        $hit = Select-String -Path $path -Pattern "^\s*$Key\s*=\s*(.+)$" | Select-Object -First 1
        if ($hit) { return $hit.Matches[0].Groups[1].Value.Trim().Trim('"') }
    }
    return $Default
}

$ApiPort     = [int](Get-EnvValue 'APP_PORT' '8101')
$MilvusPort  = [int](Get-EnvValue 'MILVUS_PORT' '19531')
$MilvusHttp  = [int](Get-EnvValue 'MILVUS_HTTP_PORT' '9092')
$MysqlPort   = [int](Get-EnvValue 'MYSQL_PORT' '3307')
$RedisPort   = [int](Get-EnvValue 'REDIS_PORT' '6380')
$BaseUrl     = "http://127.0.0.1:$ApiPort"
Write-Host "端口：API=$ApiPort MySQL=$MysqlPort Redis=$RedisPort Milvus=$MilvusPort/$MilvusHttp"

# ---------------------------------------------------------------------------
# step 1 前置检查
# ---------------------------------------------------------------------------
Write-Step '前置检查（docker / uv / 密钥）'
$missing = @()
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { $missing += 'docker' }
if (-not (Get-Command uv    -ErrorAction SilentlyContinue)) { $missing += 'uv' }
if ($missing.Count -gt 0) {
    Write-Fail "缺少依赖：$($missing -join ', ')"
    Write-Host "  docker: https://www.docker.com/products/docker-desktop/"
    Write-Host "  uv    : irm https://astral.sh/uv/install.ps1 | iex"
    Pop-Location; exit 1
}
Write-Ok "docker / uv 已就绪"

$keyFound = $false
foreach ($name in @('DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY', 'LLM_API_KEY')) {
    if ([System.Environment]::GetEnvironmentVariable($name, 'User')) { $keyFound = $true; Write-Ok "检测到用户级环境变量 $name"; break }
    if ([System.Environment]::GetEnvironmentVariable($name, 'Process')) { $keyFound = $true; Write-Ok "检测到进程级环境变量 $name"; break }
}
if (-not $keyFound) {
    Write-WarnMsg '未检测到 LLM 密钥：入库/检索可用，但问答会失败'
    Write-Host "  执行：setx DEEPSEEK_API_KEY sk-xxxx（新开终端生效）"
}

if (-not (Test-Path (Join-Path $RepoRoot '.env'))) {
    Copy-Item (Join-Path $RepoRoot '.env.example') (Join-Path $RepoRoot '.env')
    Write-Ok '已从 .env.example 生成 .env'
} else {
    Write-Skip '.env 已存在'
}

# ---------------------------------------------------------------------------
# step 2 起容器
# ---------------------------------------------------------------------------
Write-Step '启动基础设施容器（MySQL / Redis / etcd / MinIO / Milvus）'
Step-Timer {
    docker compose up -d | Out-Host
}
if (-not (Wait-ForPort -Port $MysqlPort  -Label 'MySQL'  -TimeoutSec 180)) { Pop-Location; exit 1 }
if (-not (Wait-ForPort -Port $RedisPort  -Label 'Redis'  -TimeoutSec 120)) { Pop-Location; exit 1 }
if (-not (Wait-ForPort -Port $MilvusPort -Label 'Milvus' -TimeoutSec 240)) { Pop-Location; exit 1 }

$deadline = (Get-Date).AddSeconds(240)
$milvusOk = $false
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$MilvusHttp/healthz" -TimeoutSec 5 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $milvusOk = $true; break }
    } catch { Start-Sleep -Seconds 3 }
}
if ($milvusOk) { Write-Ok 'Milvus healthz = OK' }
else { Write-WarnMsg 'Milvus healthz 未返回 200，继续尝试（首次启动较慢）' }

# ---------------------------------------------------------------------------
# step 3 安装 Python 依赖
# ---------------------------------------------------------------------------
Write-Step '安装 Python 依赖（uv sync）'
$syncArgs = @()
if ($WithEval) { $syncArgs += @('--group', 'eval') }

# 坑：pyproject 里 torch 声明为 CPU 版，而本机手工装的是 GPU 版（2.14.1+cu130）。
# 直接 uv sync 会尝试把 GPU 版卸掉换回 CPU 版（还会因文件被占用而失败）。
# 这里检测到 GPU 版已装就加 --no-install-package torch 保住它。
$venvPy = Join-Path $RepoRoot '.venv/Scripts/python.exe'
$torchGuard = @()
$runArgs = @()   # 传给后续 `uv run` 的参数
if ((Test-Path $venvPy) -and ((Get-EnvValue 'EMBEDDING_DEVICE' 'cpu') -match 'cuda')) {
    # 探测 site-packages 里是否装着带 +cu 后缀的 torch。
    # 不依赖 `python -c` 输出（本机曾出现 -c 调用静默返回空导致守卫失效、
    # uv run 自动同步把 GPU torch 换成 CPU 版），直接看 dist-info 目录名最稳。
    $sitePkgs = Join-Path (Split-Path $venvPy) '..\Lib\site-packages'
    $gpuTorch = Get-ChildItem -LiteralPath $sitePkgs -Directory -Filter 'torch-*+cu*.dist-info' `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($gpuTorch) {
        Write-Ok "检测到 GPU 版 torch（$($gpuTorch.Name)），本次跳过 torch 相关同步（避免被 CPU 版覆盖）"
        $torchGuard = @('--no-install-package', 'torch')
        # uv run 默认会先自动 sync，同样会把 GPU 版换掉，必须一并关掉
        $runArgs = @('--no-sync')
    } else {
        Write-WarnMsg '未检测到 GPU 版 torch（site-packages 无 torch-*+cu*.dist-info），按 CPU 环境继续'
    }
}

if ($SkipSync) {
    Write-Skip '按 -SkipSync 跳过'
} else {
    Step-Timer { uv sync @syncArgs @torchGuard | Out-Host }
}

# ---------------------------------------------------------------------------
# step 4 建表 + Milvus collection
# ---------------------------------------------------------------------------
Write-Step '初始化 MySQL 表结构'
Step-Timer { uv run @runArgs python -m app.db.init_db | Out-Host }

Write-Step '初始化 Milvus collection（law_chunks）'
if ($Reset) {
    Step-Timer { uv run @runArgs python -m scripts.init_milvus --drop | Out-Host }
} else {
    Step-Timer { uv run @runArgs python -m scripts.init_milvus | Out-Host }
}

# ---------------------------------------------------------------------------
# step 5 模型
# ---------------------------------------------------------------------------
Write-Step '本地模型（bge-m3 / bge-reranker-v2-m3）'
if ($SkipModels) {
    Write-Skip '按 -SkipModels 跳过'
} else {
    $prevEap = $ErrorActionPreference
    # uv 会往 stderr 打环境告警（如 SSL_CERT_DIR 无效）；Stop 下 native stderr 会被
    # 当成终止错误，这里只取退出码，不让告警打断启动。
    $ErrorActionPreference = 'Continue'
    $check = uv run @runArgs python -m scripts.download_models --check 2>&1 | Out-String
    $ErrorActionPreference = $prevEap
    if ($LASTEXITCODE -eq 0) {
        Write-Ok '模型已完整，无需下载'
    } else {
        Write-WarnMsg '模型缺失，开始下载（约 4.4GB，hf-mirror 优先，失败自动切 ModelScope）'
        Step-Timer { $env:HF_HUB_DISABLE_XET = '1'; uv run @runArgs python -m scripts.download_models | Out-Host }
        $ErrorActionPreference = 'Continue'
        $check = uv run @runArgs python -m scripts.download_models --check 2>&1 | Out-String
        $ErrorActionPreference = $prevEap
        if ($LASTEXITCODE -eq 0) { Write-Ok '模型下载完成' }
        else { Write-Fail '模型仍不完整，可手工下载后放到 models/ 目录'; Pop-Location; exit 1 }
    }
}

# ---------------------------------------------------------------------------
# step 6 启动后端
# ---------------------------------------------------------------------------
Write-Step "启动后端（$BaseUrl）"
$serverProc = $null
if (Test-TcpPort -Port $ApiPort) {
    Write-Skip "端口 $ApiPort 已被占用，假定后端已在运行"
} else {
    $serverArgs = @('run') + $runArgs + @('python', '-m', 'app.server')
    $serverProc = Start-Process -FilePath 'uv' -ArgumentList $serverArgs `
        -WorkingDirectory $RepoRoot -PassThru -WindowStyle Minimized
    Write-Host "  后端进程 PID=$($serverProc.Id)，等待健康检查…"
}

$deadline = (Get-Date).AddSeconds(180)
$apiReady = $false
while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-JsonGet -Uri "$BaseUrl/api/health/live" -TimeoutSec 5
        if ($h.status) { $apiReady = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $apiReady) {
    Write-Fail "后端未就绪，请查看 logs/app.log 或直接运行 uv run python -m app.server"
    Pop-Location; exit 1
}
Write-Ok "后端就绪：$BaseUrl/docs"
try {
    $health = Invoke-JsonGet -Uri "$BaseUrl/api/health" -TimeoutSec 10
    Write-Host ("  依赖：mysql={0} redis={1} milvus={2} llm_key={3}" -f `
        $health.services.mysql.status, $health.services.redis.status,
        $health.services.milvus.status, $health.llm_key_configured)
} catch { Write-WarnMsg '健康检查详情获取失败' }

if ($NoStart) {
    Write-Host ''
    Write-Host '按 -NoStart 停在后端已启动状态（未做知识库与入库）。' -ForegroundColor Yellow
    Pop-Location; exit 0
}

# 强制 JWT 之后 /api/kbs、/api/ingest/local 都需要 Bearer token；脚本化建库没有
# 交互式登录（登录接口还带验证码），用本地账号签一个 bootstrap token。
Write-Host '  签发本地 bootstrap token…'
$ErrorActionPreference = 'Continue'
$tokenOut = uv run @runArgs python -m scripts.bootstrap_token 2>$null
$ErrorActionPreference = 'Stop'
$Token = ($tokenOut | Select-Object -Last 1).Trim()
if (-not $Token -or $Token.Split('.').Count -ne 3) {
    Write-Fail '获取 bootstrap token 失败：建库/入库接口需要登录'
    Write-Host '  可手动执行：uv run python -m scripts.bootstrap_token'
    Pop-Location; exit 1
}
Write-Ok '已获取 bootstrap token'

# ---------------------------------------------------------------------------
# step 7 建知识库 + 入库
# ---------------------------------------------------------------------------
if ($SkipIngest) {
    Write-Step '建知识库与入库'
    Write-Skip '按 -SkipIngest 跳过'
} else {
    Write-Step '创建知识库（幂等）'
    $kbs = @(
        @{ id = 'kb_labor'; name = '劳动法知识库'; desc = '劳动合同法（98 条）与劳动法（107 条）' },
        @{ id = 'kb_civil'; name = '民法典知识库'; desc = '中华人民共和国民法典（7 编 1260 条）' }
    )
    try {
        $existing = Invoke-JsonGet -Uri "$BaseUrl/api/kbs" -TimeoutSec 20 -Token $Token
        $existingIds = @($existing | ForEach-Object { $_.id })
    } catch {
        Write-Fail "获取知识库列表失败：$($_.Exception.Message)"
        Write-Host "  接口需要登录：token 由 scripts/bootstrap_token.py 签发，可用 uv run python -m scripts.bootstrap_token 排查"
        Pop-Location; exit 1
    }
    foreach ($kb in $kbs) {
        if ($existingIds -contains $kb.id) { Write-Skip "$($kb.id) 已存在"; continue }
        try {
            Invoke-JsonPost -Uri "$BaseUrl/api/kbs" -Body @{
                id = $kb.id; name = $kb.name; description = $kb.desc
            } -TimeoutSec 30 -Token $Token | Out-Null
            Write-Ok "已创建 $($kb.id)"
        } catch {
            Write-WarnMsg "创建 $($kb.id) 失败：$($_.Exception.Message)"
        }
    }

    Write-Step '入库 data/raw 下的法律文件（幂等，内容未变则跳过）'
    $rawDir = Join-Path $RepoRoot 'data/raw'
    if (-not (Test-Path $rawDir)) {
        Write-Fail "目录不存在：$rawDir"; Pop-Location; exit 1
    }
    $files = Get-ChildItem -Path $rawDir -File | Where-Object { $_.Extension -in @('.md', '.txt') }
    if ($files.Count -eq 0) {
        Write-WarnMsg 'data/raw 下没有可入库的文件，跳过'
    }
    foreach ($f in $files) {
        $kbId = if ($f.Name -match '民法典') { 'kb_civil' } else { 'kb_labor' }
        $uri = "$BaseUrl/api/ingest/local?kb_id=$kbId&filename=" + [uri]::EscapeDataString($f.Name)
        Write-Host "  入库 $($f.Name) -> $kbId"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $r = Invoke-RestMethod -Method Post -Uri $uri -TimeoutSec 3600 `
                -ContentType 'application/json; charset=utf-8' `
                -Headers @{ Authorization = "Bearer $Token" }
            $sw.Stop()
            Write-Ok ("status={0} 条文={1} 子块={2} ({3:N1}s)" -f $r.status, $r.article_count, $r.child_count, $sw.Elapsed.TotalSeconds)
        } catch {
            $sw.Stop()
            Write-WarnMsg "入库失败：$(($_.ErrorDetails.Message | Out-String).Trim())"
        }
    }
}

# ---------------------------------------------------------------------------
# step 8 冒烟演示
# ---------------------------------------------------------------------------
Write-Step '冒烟演示（10 问覆盖 6 条路由）'
try {
    uv run @runArgs python -m scripts.demo --base-url $BaseUrl | Out-Host
    Write-Ok '演示脚本执行完毕'
} catch {
    Write-WarnMsg "演示脚本异常：$($_.Exception.Message)"
}

$total = (Get-Date) - $script:TotalStart
Write-Host ""
Write-Host ("全部完成，总耗时 {0:N1}s" -f $total.TotalSeconds) -ForegroundColor Magenta
Write-Host "  后端文档：$BaseUrl/docs"
Write-Host "  前端启动：cd frontend && npm install && npm run dev  (http://localhost:5180)"
Write-Host "  停止服务：docker compose down"
Pop-Location
