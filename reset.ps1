# Reset script for speak-logger
# Deletes unknown_speakers directory, voice_logs.db and config.json files

Write-Host "Resetting speak-logger data..." -ForegroundColor Yellow

# Remove unknown_speakers directory
if (Test-Path -Path "./unknown_speakers") {
    Write-Host "Removing unknown_speakers directory..."
    Remove-Item -Path "./unknown_speakers" -Recurse -Force
}

# Remove voice_logs.db file
if (Test-Path -Path "./voice_logs.db") {
    Write-Host "Removing voice_logs.db file..."
    Remove-Item -Path "./voice_logs.db" -Force
}

# Remove config.json file
if (Test-Path -Path "./config.json") {
    Write-Host "Removing config.json file..."
    Remove-Item -Path "./config.json" -Force
}

Write-Host "Reset complete." -ForegroundColor Green