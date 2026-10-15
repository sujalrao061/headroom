$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path $pythonExe)) { $pythonExe = 'python' }

$secureKey = Read-Host 'Paste your private Nebius API key here (input is hidden)' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try { $env:NEBIUS_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer) }
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $secureKey.Dispose()
}

try {
    if (-not $env:NEBIUS_API_KEY.StartsWith('v1.') -or $env:NEBIUS_API_KEY.Length -lt 100) {
        throw 'That does not look like a complete Nebius Token Factory API key. No model call was made.'
    }
    Write-Host 'Comparing one Nemotron attention decision with the independent measured choice.'
    & $pythonExe -m scripts.evaluate_attention_decision --live
    if ($LASTEXITCODE -ne 0) { throw 'The attention decision check did not complete.' }
    Write-Host 'Result saved to evidence\attention-decision-live.json'
}
finally { Remove-Item Env:NEBIUS_API_KEY -ErrorAction SilentlyContinue }
