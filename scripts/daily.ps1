$ErrorActionPreference = "Stop"
Set-Location "C:\Users\hp\job-apply-agent"
& .\.venv\Scripts\python.exe -m job_agent fetch
& .\.venv\Scripts\python.exe -m job_agent export-pending
Write-Host "Review pending packets in data\outbox and run: python -m job_agent list"
