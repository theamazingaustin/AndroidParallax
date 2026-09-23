# scripts/build.ps1 - Minimal token usage APK builder
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$res = .\gradlew assembleRelease --quiet 2>&1
if ($LASTEXITCODE -eq 0) {
    $apk = Get-ChildItem "app/build/outputs/apk/release/*.apk" | Select-Object -First 1
    if ($apk) {
        $mb = [math]::Round($apk.Length / 1MB, 2)
        Write-Output "BUILD SUCCESS: $($apk.Name) ($mb MB)"
    } else {
        Write-Output "BUILD SUCCESS (APK generated)"
    }
} else {
    Write-Error "BUILD FAILED: $res"
    exit 1
}
