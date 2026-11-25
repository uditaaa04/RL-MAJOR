# Shared helpers for deploy_aws.ps1 / update_server.ps1 / teardown_aws.ps1 (Windows PowerShell 5.1+).
# Dot-sourced, not run directly.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Secrets = Join-Path $PSScriptRoot ".secrets"

function Assert-AwsCli {
    if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
        throw "AWS CLI v2 not found. Install it from https://awscli.amazonaws.com/AWSCLIV2.msi, then run 'aws configure' (region ap-south-1)."
    }
}

function Invoke-Aws {
    # runs the AWS CLI, returns its stdout, throws on a non-zero exit code
    $out = & aws @args
    if ($LASTEXITCODE -ne 0) { throw "aws $($args -join ' ') failed (exit code $LASTEXITCODE)" }
    return $out
}

function Test-Aws {
    # runs the AWS CLI quietly, returns $true if it succeeded.
    # (local 'Continue': in Windows PowerShell 5.1 a redirected stderr line would otherwise throw)
    $ErrorActionPreference = "Continue"
    & aws @args *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-ProjectPython {
    $venv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    return "python"
}

function Get-BucketName([string]$Account, [string]$Region) { return "smart-conveyor-$Account-$Region" }
function Get-KeyParamName([string]$StackName) { return "/smart-conveyor/$StackName-api-key" }

function New-ApiKey {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    return ([System.BitConverter]::ToString($bytes) -replace "-", "").ToLower()
}

function Publish-Bundle([string]$Bucket, [string]$Region) {
    $py = Get-ProjectPython
    Write-Host "[deploy] packaging the server bundle ..."
    & $py (Join-Path $Root "cloud\package_bundle.py")
    if ($LASTEXITCODE -ne 0) { throw "packaging failed" }
    $zip = Join-Path $Root "dist\server_bundle.zip"
    Write-Host "[deploy] uploading to s3://$Bucket/smart-conveyor/bundle/ ..."
    Invoke-Aws s3 cp $zip "s3://$Bucket/smart-conveyor/bundle/server_bundle.zip" --region $Region --only-show-errors | Out-Null
}

function Get-StackOutputs([string]$StackName, [string]$Region) {
    $json = Invoke-Aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs" --output json
    $h = @{}
    foreach ($o in (($json -join "`n") | ConvertFrom-Json)) { $h[$o.OutputKey] = $o.OutputValue }
    return $h
}

function Invoke-RemoteUpdate([string]$InstanceId, [string]$Region) {
    Write-Host "[update] telling the instance to pull the new bundle and restart (SSM Run Command) ..."
    $cmdId = Invoke-Aws ssm send-command --region $Region --instance-ids $InstanceId --document-name AWS-RunShellScript `
        --comment "smart-conveyor update" --parameters "commands=/opt/smart-conveyor/update.sh" --query Command.CommandId --output text
    $cmdId = ($cmdId | Out-String).Trim()
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 4
        $ErrorActionPreference = "Continue"     # the invocation may not exist for a few seconds
        $st = (& aws ssm get-command-invocation --region $Region --command-id $cmdId --instance-id $InstanceId --query Status --output text 2>$null | Out-String).Trim()
        $ErrorActionPreference = "Stop"
        if ($st -in @("Success", "Failed", "Cancelled", "TimedOut")) {
            $outText = Invoke-Aws ssm get-command-invocation --region $Region --command-id $cmdId --instance-id $InstanceId --query StandardOutputContent --output text
            Write-Host ($outText | Out-String)
            if ($st -ne "Success") { throw "remote update finished with status $st" }
            Write-Host "[update] server updated."
            return
        }
    }
    throw "remote update did not finish in 6 minutes (command $cmdId)"
}
