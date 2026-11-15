# 稳定启动API，优先使用原端口；Windows保留或其他程序占用时使用项目备用端口。
param([int]$ImportPort=8000, [int]$QueryPort=8001)
$ErrorActionPreference = 'Stop'
$apiProjectRoot = Split-Path -Parent $PSScriptRoot
$apiPythonPath = Join-Path $apiProjectRoot '.venv\Scripts\python.exe'
$apiOutputDir = Join-Path $apiProjectRoot 'output\runtime'
New-Item -ItemType Directory -Path $apiOutputDir -Force | Out-Null
$apiRuntimeRecords = @()
foreach ($apiDefinition in @(@{Name='import-api';Port=$ImportPort;Fallback=18000;Module='app.api.http.import_server:app'}, @{Name='query-api';Port=$QueryPort;Fallback=18001;Module='app.api.http.query_server:app'})) {
    $apiChosenPort = $null
    foreach ($apiCandidatePort in @($apiDefinition.Port,$apiDefinition.Fallback) | Select-Object -Unique) {
        $apiExistingListener = Get-NetTCPConnection -State Listen -LocalPort $apiCandidatePort -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($apiExistingListener) {
            $apiExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($apiExistingListener.OwningProcess)"
            if ($apiExistingProcess.CommandLine -like "*$($apiDefinition.Module)*") {
                $apiChosenPort = $apiCandidatePort
                break
            }
            continue
        }
        $apiTestListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $apiCandidatePort)
        try {
            $apiTestListener.Start()
            $apiChosenPort = $apiCandidatePort
        } catch [System.Net.Sockets.SocketException] {
            Write-Output "端口${apiCandidatePort}不可绑定，尝试本项目备用端口。"
        } finally {
            $apiTestListener.Stop()
        }
        if ($null -ne $apiChosenPort) {
            Start-Process -FilePath $apiPythonPath -ArgumentList @('-X','utf8','-m','uvicorn',$apiDefinition.Module,'--host','127.0.0.1','--port',"$apiChosenPort") -WorkingDirectory $apiProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $apiOutputDir "$($apiDefinition.Name).stdout.log") -RedirectStandardError (Join-Path $apiOutputDir "$($apiDefinition.Name).stderr.log") | Out-Null
            break
        }
    }
    if ($null -eq $apiChosenPort) { throw "$($apiDefinition.Name)没有可用端口，未启动。" }
    $apiHealthy = $false
    # 冷启动会加载torch/transformers，首次启动允许90秒，不把导入时间误判为断网。
    for ($apiAttempt=0; $apiAttempt -lt 180; $apiAttempt++) {
        try {
            $apiHealth = Invoke-WebRequest -Uri "http://127.0.0.1:$apiChosenPort/health" -TimeoutSec 1 -UseBasicParsing
            if ($apiHealth.StatusCode -eq 200) { $apiHealthy=$true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $apiHealthy) { throw "$($apiDefinition.Name)启动后健康检查未通过，请检查output/runtime日志。" }
    $apiListener = Get-NetTCPConnection -State Listen -LocalPort $apiChosenPort | Select-Object -First 1
    $apiRuntimeRecords += @{Name=$apiDefinition.Name;Port=$apiChosenPort;ProcessId=$apiListener.OwningProcess;Mode='stable-no-reload';BaseUrl="http://127.0.0.1:$apiChosenPort"}
    Write-Output "$($apiDefinition.Name)已通过健康检查：http://127.0.0.1:$apiChosenPort/html"
}
$apiRuntimeRecords | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $apiOutputDir 'api-processes.json') -Encoding UTF8
