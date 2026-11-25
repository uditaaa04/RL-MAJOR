<#
.SYNOPSIS
  Push the current code + models to the running EC2 server and restart it (no SSH needed: SSM Run Command).
  The instance keeps its data (box log, Q-table backups); it rolls back if the new version does not start.
.EXAMPLE
  .\cloud\update_server.ps1
#>
param(
    [string]$Region = "ap-south-1",
    [string]$StackName = "smart-conveyor"
)
. (Join-Path $PSScriptRoot "aws_common.ps1")
Assert-AwsCli
$account = (Invoke-Aws sts get-caller-identity --query Account --output text | Out-String).Trim()
$bucket = Get-BucketName $account $Region
$out = Get-StackOutputs $StackName $Region
Publish-Bundle $bucket $Region
Invoke-RemoteUpdate $out["InstanceId"] $Region
Write-Host "Server: $($out['ServerUrl'])"
