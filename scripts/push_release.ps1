# scripts/push_release.ps1 - Git push and create release tag
param(
    [string]$Tag = ""
)

if ($Tag -ne "") {
    git tag -a $Tag -m "Release $Tag"
    git push origin main --tags
    Write-Output "PUSHED with tag $Tag"
} else {
    git push origin main
    Write-Output "PUSHED to main"
}
