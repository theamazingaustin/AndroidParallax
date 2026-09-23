# scripts/git_ops.ps1 - Token-saving unified Git commit, push, and release waiter
param(
    [string]$Message = "",
    [switch]$Push,
    [string]$Tag = "",
    [switch]$Wait
)

$ErrorActionPreference = "Stop"

# 1. Commit if Message is provided
if ($Message -ne "") {
    git add -A
    git commit -m $Message
    Write-Output "COMMITTED: $Message"
}

# 2. Push if requested
if ($Push) {
    if ($Tag -ne "") {
        git tag -a $Tag -m "Release $Tag"
        git push origin main --tags
        Write-Output "PUSHED: main with tag $Tag"
    } else {
        git push origin main
        Write-Output "PUSHED: main"
    }

    # Automatically enable wait when pushing
    $Wait = $true
}

# 3. Wait for release APK if requested
if ($Wait) {
    Write-Output "Waiting for GitHub Actions to build and release APK..."

    # Retrieve GitHub token from Git Credential Manager
    $cred = @("protocol=https", "host=github.com", "") | & git.exe credential fill
    $tokenLine = $cred | Where-Object { $_ -like "password=*" }
    if (-not $tokenLine) {
        Write-Output "No GitHub token found in Credential Manager. Cannot poll API."
        exit 0
    }
    $token = $tokenLine.Substring(9)
    $headers = @{
        "Authorization" = "Bearer $token"
        "Accept" = "application/vnd.github+json"
        "User-Agent" = "PowerShell"
    }

    $repo = "theamazingaustin/AndroidParallax"
    $maxWaitSec = 600
    $elapsed = 0
    $interval = 25

    while ($elapsed -lt $maxWaitSec) {
        Start-Sleep -Seconds $interval
        $elapsed += $interval

        try {
            # Check latest releases
            $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases" -Headers $headers
            $targetRelease = if ($Tag -ne "") {
                $releases | Where-Object { $_.tag_name -eq $Tag }
            } else {
                $releases | Select-Object -First 1
            }

            if ($targetRelease) {
                $apk = $targetRelease.assets | Where-Object { $_.name -like "*.apk" } | Select-Object -First 1
                if ($apk) {
                    Write-Output "=================================================="
                    Write-Output "APK_DOWNLOAD_READY: $($apk.browser_download_url)"
                    Write-Output "RELEASE_PAGE: $($targetRelease.html_url)"
                    Write-Output "RELEASE_TAG: $($targetRelease.tag_name)"
                    Write-Output "APK_SIZE: $([math]::Round($apk.size / 1MB, 2)) MB"
                    Write-Output "=================================================="
                    exit 0
                }
            }
            Write-Output "Still compiling on GitHub Actions ($elapsed s)..."
        } catch {
            Write-Output "Polling API error: $_"
        }
    }

    Write-Output "Timed out waiting for GitHub Release after $maxWaitSec seconds."
}
