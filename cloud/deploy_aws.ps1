<#
.SYNOPSIS
  Deploy the Smart Conveyor cloud server to AWS in one command:
  S3 bucket + API key (SSM SecureString) + code bundle + CloudFormation stack (EC2, Elastic IP,
  security group, IAM role, CloudWatch dashboard/logs/alarms).

.EXAMPLE
  .\cloud\deploy_aws.ps1
      Mumbai (ap-south-1), t3.micro (as in the paper), port 8000 open only to YOUR current public IP.

.EXAMPLE
  .\cloud\deploy_aws.ps1 -AllowedCidr 0.0.0.0/0 -AlarmEmail you@example.com -InstanceType c6i.large
      Reachable from any network (the API key still protects it), e-mail alarms, faster CPU for experiments.

.NOTES
  Needs: AWS CLI v2 configured with your own account ('aws configure'), Python with the project installed.
  Cost: you pay AWS for what runs (t3.micro in Mumbai is about USD 0.01 per hour, free tier for eligible
  accounts, plus the public IP). Check the AWS pricing page for current numbers.
  Remove everything afterwards with .\cloud\teardown_aws.ps1
  Running it again updates the stack and pushes the newest code to the instance.
#>
param(
    [string]$Region = "ap-south-1",
    [string]$StackName = "smart-conveyor",
    [ValidateSet("t3.micro", "t3.small", "t3.medium", "c6i.large", "c7i.large")]
    [string]$InstanceType = "t3.micro",
    [string]$AllowedCidr = "",
    [string]$AlarmEmail = "",
    [string]$KeyPairName = "",
    [string]$LineName = "line-1",
    [switch]$RotateApiKey
)
. (Join-Path $PSScriptRoot "aws_common.ps1")
Assert-AwsCli

$account = (Invoke-Aws sts get-caller-identity --query Account --output text | Out-String).Trim()
$bucket = Get-BucketName $account $Region
$keyParam = Get-KeyParamName $StackName
Write-Host "[deploy] account $account  region $Region  stack $StackName  bucket $bucket"

# ---------------------------------------------------------------- 1. S3 bucket (private, encrypted)
if (-not (Test-Aws s3api head-bucket --bucket $bucket --region $Region)) {
    Write-Host "[deploy] creating bucket $bucket"
    if ($Region -eq "us-east-1") {
        Invoke-Aws s3api create-bucket --bucket $bucket --region $Region | Out-Null
    } else {
        Invoke-Aws s3api create-bucket --bucket $bucket --region $Region --create-bucket-configuration "LocationConstraint=$Region" | Out-Null
    }
    Invoke-Aws s3api put-public-access-block --bucket $bucket --region $Region `
        --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" | Out-Null
}

# ---------------------------------------------------------------- 2. API key in SSM Parameter Store
New-Item -ItemType Directory -Force $Secrets | Out-Null
$keyFile = Join-Path $Secrets "api_key_$StackName.txt"
$haveKey = Test-Aws ssm get-parameter --name $keyParam --region $Region
if ($RotateApiKey -or -not $haveKey) {
    $apiKey = New-ApiKey
    Invoke-Aws ssm put-parameter --name $keyParam --type SecureString --overwrite --region $Region --value $apiKey `
        --description "Smart Conveyor API key ($StackName)" | Out-Null
    Write-Host "[deploy] new API key stored in SSM $keyParam"
} else {
    $apiKey = (Invoke-Aws ssm get-parameter --name $keyParam --with-decryption --region $Region --query Parameter.Value --output text | Out-String).Trim()
}
Set-Content -Path $keyFile -Value $apiKey -Encoding ascii -NoNewline
Write-Host "[deploy] API key saved locally in $keyFile  (keep it private; give it to the Pi)"

# ---------------------------------------------------------------- 3. code bundle
Publish-Bundle $bucket $Region

# ---------------------------------------------------------------- 4. who may connect
if (-not $AllowedCidr) {
    $myIp = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com" -TimeoutSec 10).Trim()
    $AllowedCidr = "$myIp/32"
    Write-Host "[deploy] port open only to your current public IP $AllowedCidr"
    Write-Host "         (if the Pi is on another network, re-run with -AllowedCidr <its-public-ip>/32 or 0.0.0.0/0)"
}

# ---------------------------------------------------------------- 5. CloudFormation
$existed = Test-Aws cloudformation describe-stacks --stack-name $StackName --region $Region
$params = @("InstanceType=$InstanceType", "ArtifactBucket=$bucket", "ApiKeyParameterName=$keyParam",
            "AllowedCidr=$AllowedCidr", "LineName=$LineName")
if ($AlarmEmail) { $params += "AlarmEmail=$AlarmEmail" }
if ($KeyPairName) { $params += "KeyPairName=$KeyPairName" }
Write-Host "[deploy] creating/updating the stack (first time: ~5-8 minutes, it waits until the server answers) ..."
Invoke-Aws cloudformation deploy --region $Region --stack-name $StackName `
    --template-file (Join-Path $Root "cloud\aws\cloudformation.yaml") `
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --tags Project=smart-conveyor `
    --parameter-overrides @params | Out-Host

$out = Get-StackOutputs $StackName $Region
if ($existed) {
    # stack was already there: make sure the instance runs the code we just uploaded
    Invoke-RemoteUpdate $out["InstanceId"] $Region
}

$info = [ordered]@{ stack = $StackName; region = $Region; server_url = $out["ServerUrl"]; dashboard = $out["DashboardUrl"];
                    cloudwatch = $out["CloudWatchDashboard"]; instance = $out["InstanceId"]; bucket = $bucket;
                    api_key_file = $keyFile; deployed = (Get-Date -Format "yyyy-MM-dd HH:mm") }
$info | ConvertTo-Json | Set-Content -Path (Join-Path $Secrets "stack_$StackName.json") -Encoding utf8

Write-Host ""
Write-Host "=================================================================="
Write-Host " Cloud server:  $($out['ServerUrl'])"
Write-Host " Dashboard:     $($out['DashboardUrl'])   (press 'Key' and paste the API key)"
Write-Host " CloudWatch:    $($out['CloudWatchDashboard'])"
Write-Host " API key file:  $keyFile"
Write-Host "=================================================================="
Write-Host " Check it from this laptop:"
Write-Host "   `$env:CONVEYOR_API_KEY = Get-Content '$keyFile'"
Write-Host "   python -m tools.check_server --server $($out['ServerUrl'])"
Write-Host " Mock Pi against the real cloud (no hardware):"
Write-Host "   python -m gateway_node.main --mock --server $($out['ServerUrl'])"
Write-Host " On the Raspberry Pi:"
Write-Host "   sudo bash deploy/pi/install_pi.sh $($out['ServerUrl']) <api-key>"
