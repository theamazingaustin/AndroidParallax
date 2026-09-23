# scripts/test.ps1 - Minimal token usage unit test runner
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$res = .\gradlew testDebugUnitTest --quiet 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Output "TESTS PASSED"
} else {
    Write-Error "TESTS FAILED: $res"
    exit 1
}
