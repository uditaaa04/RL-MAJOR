#!/bin/bash
# =============================================================================
# Deploy the Smart Conveyor cloud server from AWS CloudShell (a terminal inside
# the AWS website - no AWS CLI or keys needed on your own PC).
#
#   1. AWS console -> region "Asia Pacific (Mumbai)" -> CloudShell icon (>_)
#   2. Actions -> Upload file -> dist/server_bundle.zip
#   3. unzip -o -q server_bundle.zip 'cloud/*' -d sc && bash sc/cloud/deploy_cloudshell.sh server_bundle.zip
#
# Does the same as cloud/deploy_aws.ps1: private S3 bucket, API key in SSM
# (generated here, never typed by anyone), code bundle upload, CloudFormation
# stack (EC2 + Elastic IP + security group + IAM role + CloudWatch).
# Run it again later to push a new bundle (new models) to the running server.
#
# Options (environment variables):
#   REGION=ap-south-1  STACK=smart-conveyor  ITYPE=t3.micro  CIDR=0.0.0.0/0
# Remove everything afterwards:  bash sc/cloud/deploy_cloudshell.sh --teardown
# =============================================================================
set -euo pipefail

REGION="${REGION:-ap-south-1}"
STACK="${STACK:-smart-conveyor}"
ITYPE="${ITYPE:-t3.micro}"
CIDR="${CIDR:-0.0.0.0/0}"
ARG="${1:-server_bundle.zip}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="smart-conveyor-${ACCOUNT}-${REGION}"
KEYPARAM="/smart-conveyor/${STACK}-api-key"
BUNDLE_KEY="smart-conveyor/bundle/server_bundle.zip"

show_key_hint() {
  echo " API key (show it):  aws ssm get-parameter --region $REGION --name $KEYPARAM --with-decryption --query Parameter.Value --output text"
}

if [ "$ARG" = "--teardown" ]; then
  echo "[teardown] deleting stack $STACK (server, IP, role, monitoring) ..."
  aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
  echo "[teardown] emptying and deleting bucket $BUCKET ..."
  aws s3 rm "s3://$BUCKET" --recursive --region "$REGION" --only-show-errors || true
  aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION" || true
  aws ssm delete-parameter --region "$REGION" --name "$KEYPARAM" || true
  echo "[teardown] done - nothing of this project is left running."
  exit 0
fi

if [ "$ARG" = "--status" ]; then
  aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
      --query "Stacks[0].[StackStatus,Outputs]" --output json
  show_key_hint
  exit 0
fi

BUNDLE="$ARG"
[ -f "$BUNDLE" ] || { echo "bundle not found: $BUNDLE  (upload dist/server_bundle.zip first)"; exit 1; }
echo "[deploy] account $ACCOUNT  region $REGION  stack $STACK  type $ITYPE  open-to $CIDR"

# ---------------------------------------------------------------- 1. S3 bucket (private)
if ! aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null; then
  echo "[deploy] creating bucket $BUCKET"
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
        --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --region "$REGION" \
      --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
fi

# ---------------------------------------------------------------- 2. API key in SSM (random, made here)
if ! aws ssm get-parameter --name "$KEYPARAM" --region "$REGION" >/dev/null 2>&1; then
  aws ssm put-parameter --name "$KEYPARAM" --type SecureString --overwrite --region "$REGION" \
      --value "$(openssl rand -hex 32)" --description "Smart Conveyor API key ($STACK)" >/dev/null
  echo "[deploy] new API key stored in SSM $KEYPARAM"
else
  echo "[deploy] API key already in SSM $KEYPARAM (kept)"
fi

# ---------------------------------------------------------------- 3. code bundle
echo "[deploy] uploading bundle to s3://$BUCKET/$BUNDLE_KEY"
aws s3 cp "$BUNDLE" "s3://$BUCKET/$BUNDLE_KEY" --region "$REGION" --only-show-errors

# ---------------------------------------------------------------- 4. CloudFormation
TMP="$(mktemp -d)"
unzip -o -q "$BUNDLE" 'cloud/aws/cloudformation.yaml' -d "$TMP"
EXISTED=0
if aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" >/dev/null 2>&1; then EXISTED=1; fi
echo "[deploy] creating/updating the stack (first time: 5-8 minutes, it waits until the server answers) ..."
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
    --template-file "$TMP/cloud/aws/cloudformation.yaml" \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --tags Project=smart-conveyor \
    --parameter-overrides "InstanceType=$ITYPE" "ArtifactBucket=$BUCKET" "ApiKeyParameterName=$KEYPARAM" \
                          "AllowedCidr=$CIDR" "LineName=line-1"

out() { aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
URL="$(out ServerUrl)"
IID="$(out InstanceId)"

# ---------------------------------------------------------------- 5. stack was already there: push the new bundle
if [ "$EXISTED" = "1" ]; then
  echo "[update] telling the server to pull the new bundle and restart ..."
  CMD="$(aws ssm send-command --region "$REGION" --instance-ids "$IID" --document-name AWS-RunShellScript \
        --comment "smart-conveyor update" --parameters "commands=/opt/smart-conveyor/update.sh" \
        --query Command.CommandId --output text)"
  ST="Pending"
  for _ in $(seq 1 90); do
    sleep 4
    ST="$(aws ssm get-command-invocation --region "$REGION" --command-id "$CMD" --instance-id "$IID" \
          --query Status --output text 2>/dev/null || echo Pending)"
    case "$ST" in Success|Failed|Cancelled|TimedOut) break;; esac
  done
  echo "[update] status: $ST"
  [ "$ST" = "Success" ] || { echo "[update] FAILED - the old version keeps running"; exit 1; }
fi

# ---------------------------------------------------------------- 6. check + summary
echo
echo "[check] $URL/health"
for _ in $(seq 1 20); do
  if curl -fsS --max-time 5 "$URL/health"; then echo; break; fi
  sleep 3
done
echo
echo "=================================================================="
echo " Cloud server : $URL"
echo " Dashboard    : $URL   (press 'Key' and paste the API key)"
echo " Instance     : $IID   ($ITYPE, $REGION)"
show_key_hint
echo " Stop paying  : bash sc/cloud/deploy_cloudshell.sh --teardown"
echo "=================================================================="
