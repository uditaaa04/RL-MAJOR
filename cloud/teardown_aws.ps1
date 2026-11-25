<#
.SYNOPSIS
  Delete the cloud deployment so it stops costing money.
  Removes the CloudFormation stack (EC2 instance, Elastic IP, security group, IAM role, CloudWatch dashboard,
  alarms, log group). The S3 bucket (code bundle + backups of box logs and Q-tables) and the API key are KEPT
  unless you add -DeleteBucket / -DeleteApiKey.
.EXAMPLE
  .\cloud\teardown_aws.ps1
.EXAMPLE
  .\cloud\teardown_aws.ps1 -DeleteBucket -DeleteApiKey     # remove everything, including the backups
#>
param(
    [string]$Region = "ap-south-1",
    [string]$StackName = "smart-conveyor",
    [switch]$DeleteBucket,
    [switch]$DeleteApiKey,
    [switch]$Yes
)
. (Join-Path $PSScriptRoot "aws_common.ps1")
Assert-AwsCli
$account = (Invoke-Aws sts get-caller-identity --query Account --output text | Out-String).Trim()
$bucket = Get-BucketName $account $Region

$what = "stack '$StackName' in $Region (EC2 server, public IP, monitoring)"
if ($DeleteBucket) { $what += ", bucket s3://$bucket WITH ALL BACKUPS (cannot be undone)" }
if ($DeleteApiKey) { $what += ", API key $(Get-KeyParamName $StackName)" }
if (-not $Yes) {
    $ans = Read-Host "Delete $what ? Type 'yes' to continue"
    if ($ans -ne "yes") { Write-Host "cancelled"; exit 1 }
}

if (Test-Aws cloudformation describe-stacks --stack-name $StackName --region $Region) {
    Write-Host "[teardown] deleting stack $StackName ..."
    Invoke-Aws cloudformation delete-stack --stack-name $StackName --region $Region | Out-Null
    Invoke-Aws cloudformation wait stack-delete-complete --stack-name $StackName --region $Region | Out-Null
    Write-Host "[teardown] stack deleted"
} else {
    Write-Host "[teardown] no stack named $StackName"
}
if ($DeleteBucket -and (Test-Aws s3api head-bucket --bucket $bucket --region $Region)) {
    Invoke-Aws s3 rb "s3://$bucket" --force --region $Region | Out-Null
    Write-Host "[teardown] bucket $bucket deleted"
}
if ($DeleteApiKey) {
    if (Test-Aws ssm delete-parameter --name (Get-KeyParamName $StackName) --region $Region) { Write-Host "[teardown] API key deleted" }
}
Write-Host "done."
