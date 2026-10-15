param(
    [switch]$NoKey
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$priorPythonPath = $env:PYTHONPATH
$pythonExe = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path $pythonExe)) {
    $pythonExe = 'python'
}
$torchDeps = Join-Path (Get-Location) '.torch-deps'
if (Test-Path $torchDeps) {
    $env:PYTHONPATH = if ($priorPythonPath) { "$torchDeps$([IO.Path]::PathSeparator)$priorPythonPath" } else { $torchDeps }
}

& $pythonExe -c 'import torch; print(torch.__version__)'
if ($LASTEXITCODE -ne 0) {
    throw 'PyTorch is not available to this Python installation. The Nebius key prompt was not shown.'
}

$portCheck = [Net.Sockets.TcpClient]::new()
try {
    $portCheck.Connect('127.0.0.1', 8000)
    throw 'Another Headroom runner is already using port 8000. Close that runner before entering a key.'
}
catch [Net.Sockets.SocketException] {
    # Nothing is listening, so this runner can start.
}
finally {
    $portCheck.Dispose()
}

if (-not $NoKey) {
    $secureKey = Read-Host 'Paste your NEW Nebius API key here (input is hidden)' -AsSecureString
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:NEBIUS_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
        $secureKey.Dispose()
    }
    if (-not $env:NEBIUS_API_KEY.StartsWith('v1.') -or $env:NEBIUS_API_KEY.Length -lt 100) {
        Remove-Item Env:NEBIUS_API_KEY -ErrorAction SilentlyContinue
        throw 'That does not look like a full Nebius Token Factory API key. Enter the complete key shown only when a new key is created, not a service-account ID.'
    }
}

try {
    Write-Host 'Starting Headroom runner on http://127.0.0.1:8000 ...'
    & $pythonExe -m headroom.server
}
finally {
    Remove-Item Env:NEBIUS_API_KEY -ErrorAction SilentlyContinue
    if ($null -eq $priorPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $priorPythonPath
    }
}
