<#
.SYNOPSIS
    Запуск pytest із жорстким лімітом памʼяті, який тримає ядро.

.DESCRIPTION
    Навіщо ліміт. Незакритий цикл у тестах уже одного разу зʼїв 27 ГБ і поклав
    VS Code ("The window terminated unexpectedly (reason: 'oom')"). Тому тести
    в цих проектах не запускаються без стелі памʼяті.

    Чому саме Docker, а не Job Object. Попередній варіант ставив ліміт через
    Windows Job Object, викликаючи kernel32 з PowerShell (Add-Type + P/Invoke:
    CreateJobObject / AssignProcessToJobObject / TerminateJobObject). Ліміт
    працював, але Avast розпізнає такий скрипт як IDP.HELU.PSD11 — евристика
    "маніпуляція процесами" — і відправляє його в карантин.

    Docker дає рівноцінний захист без жодного P/Invoke: `--memory` — це cgroup,
    його теж тримає ядро, і перевищення вбиває саме контейнер, а не машину.
    Побічний бонус: тести їдуть у тому самому оточенні, що й CI, тож "у мене
    працює" перестає бути аргументом.

    Вивід іде у файл, у термінал — один рядок. Renderer VS Code має ліміт heap
    ~4 ГБ незалежно від RAM машини, і потік тексту з тестів валить саме його.

.PARAMETER MemoryMB
    Ліміт памʼяті на контейнер. Дефолт 1024 МБ.

.PARAMETER TimeoutSec
    Стеля часу на весь прогін. Дефолт 300 с.

.PARAMETER Path
    Конкретний файл або тека замість повного набору.

.PARAMETER Rebuild
    Перезібрати образ, навіть якщо він свіжий.

.EXAMPLE
    .\scripts\run_tests_capped.ps1
    .\scripts\run_tests_capped.ps1 -Path tests\test_agent_guards.py
    .\scripts\run_tests_capped.ps1 -MemoryMB 512
#>
[CmdletBinding()]
param(
    [int]$MemoryMB   = 1024,
    [int]$TimeoutSec = 300,
    [string]$Path    = "",
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent $PSScriptRoot
$Image  = 'inbox-agent-tests'
$LogDir = Join-Path $Root '.test-logs'
$Log    = Join-Path $LogDir 'pytest.log'

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# --- чи є Docker -----------------------------------------------------------
$dockerOk = $false
try {
    docker version --format '{{.Server.Version}}' | Out-Null
    $dockerOk = ($LASTEXITCODE -eq 0)
} catch {
    $dockerOk = $false
}

if (-not $dockerOk) {
    Write-Host "Docker недоступний — запусти Docker Desktop." -ForegroundColor Red
    Write-Host "Без нього немає стелі памʼяті, а без стелі тести тут не ганяємо." -ForegroundColor DarkGray
    exit 3
}

# --- образ -----------------------------------------------------------------
$needBuild = $Rebuild.IsPresent
if (-not $needBuild) {
    $existing = docker images -q $Image 2>$null
    if (-not $existing) {
        $needBuild = $true
    } else {
        # Перезбираємо, якщо залежності новіші за образ.
        $imageDate = [datetime]::Parse((docker inspect -f '{{.Created}}' $Image))
        $reqDate   = (Get-Item (Join-Path $Root 'requirements.txt')).LastWriteTime
        if ($reqDate -gt $imageDate) { $needBuild = $true }
    }
}

if ($needBuild) {
    Write-Host "збираю образ $Image ..." -ForegroundColor DarkGray
    docker build -q -t $Image $Root | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "збірка образу впала" -ForegroundColor Red
        exit 4
    }
}

# --- прогін ----------------------------------------------------------------
$target = if ($Path) { $Path.Replace('\', '/') } else { 'tests' }
$mount  = "$($Root):/work"

$dockerArgs = @(
    'run', '--rm',
    '--memory', "${MemoryMB}m",
    # Без цього ядро дозволить вихід у swap, і ліміт перестане бути лімітом.
    '--memory-swap', "${MemoryMB}m",
    '--cpus', '2',
    '--network', 'none',           # тести не ходять у мережу — заборонимо явно
    '-v', $mount,
    '-w', '/work',
    '-e', 'PYTHONDONTWRITEBYTECODE=1',
    '--entrypoint', 'python',
    $Image,
    '-m', 'pytest', $target, '-q', '--no-header', '--tb=short', '-p', 'no:cacheprovider'
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath 'docker' -ArgumentList $dockerArgs -PassThru -NoNewWindow `
                      -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"

# Звертання до .Handle кешує дескриптор процесу. Без цього PowerShell 5.1
# втрачає доступ до ExitCode після завершення, і будь-який прогін виглядає
# як FAIL з порожнім кодом — навіть коли всі тести зелені.
$null = $proc.Handle

while (-not $proc.HasExited) {
    Start-Sleep -Milliseconds 300
    if ($sw.Elapsed.TotalSeconds -gt $TimeoutSec) {
        Write-Host "ТАЙМАУТ (> $TimeoutSec с) — контейнер зупинено" -ForegroundColor Red
        $proc.Kill()
        exit 1
    }
}
# Без цього виклику $proc.ExitCode лишається порожнім навіть після HasExited:
# Start-Process -PassThru заповнює його лише коли процес дочекано явно.
$proc.WaitForExit()
$sw.Stop()

$summary = ''
if (Test-Path $Log) {
    $hits = @(Get-Content $Log -Tail 40 -ErrorAction SilentlyContinue |
              Where-Object { $_ -match '(passed|failed|error|no tests ran)' })
    if ($hits.Count -gt 0) { $summary = $hits[-1].Trim() }
}

$code = $proc.ExitCode
# 137 = SIGKILL від OOM-killer: контейнер уперся в ліміт памʼяті.
if ($code -eq 137) {
    Write-Host ("MEMLIMIT: тести перевищили {0} МБ і були вбиті ядром" -f $MemoryMB) -ForegroundColor Red
    Write-Host "деталі — у $Log" -ForegroundColor Yellow
    exit 1
}

$color = if ($code -eq 0) { 'Green' } else { 'Red' }
Write-Host ("{0,-8} ліміт {1} МБ | {2,5:N1} с | {3}" -f `
    $(if ($code -eq 0) { 'PASS' } else { "FAIL ($code)" }), $MemoryMB, `
    $sw.Elapsed.TotalSeconds, $summary) -ForegroundColor $color

if ($code -ne 0) { Write-Host "деталі — у $Log" -ForegroundColor Yellow }
exit $code
