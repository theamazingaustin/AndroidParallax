# scripts/commit.ps1 - Fast git commit helper
param(
    [Parameter(Mandatory=$true)]
    [string]$Message
)

git add -A
git commit -m $Message
Write-Output "COMMITTED: $Message"
