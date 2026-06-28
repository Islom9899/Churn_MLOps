param(
    [string]$Bucket = "churn-mlops-dvc-mansurov-ap-northeast-2",
    [string]$Region = "ap-northeast-2"
)

$ErrorActionPreference = "Stop"

function Convert-SecureStringToPlainText {
    param([securestring]$Secret)

    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secret)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtualenv python not found at $python"
}

$accessKey = Read-Host "AWS Access key ID"
$secretSecure = Read-Host "AWS Secret access key" -AsSecureString
$secretKey = Convert-SecureStringToPlainText $secretSecure

try {
    $env:AWS_ACCESS_KEY_ID = $accessKey
    $env:AWS_SECRET_ACCESS_KEY = $secretKey
    $env:AWS_DEFAULT_REGION = $Region

    & $python -m pip show dvc-s3 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & $python -m pip install -r requirements.txt
    }

    & $python -m dvc remote modify local_remote url "s3://$Bucket/dvc"
    & $python -m dvc remote modify local_remote region $Region
    & $python -m dvc push
}
finally {
    Remove-Item Env:\AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
    Remove-Item Env:\AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\AWS_DEFAULT_REGION -ErrorAction SilentlyContinue
}
