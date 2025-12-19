#!/bin/bash
set -e

# ReasonForge Infrastructure Deployment Script
# Deploys S3 buckets and IAM roles for the training infrastructure
# Note: Training instance should be created manually in AWS Console for learning purposes

# Configuration
PROJECT_NAME="reasonforge"
ENVIRONMENT="dev"
REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to wait for stack completion
wait_for_stack() {
    local stack_name=$1
    print_info "Waiting for stack ${stack_name} to complete..."

    aws cloudformation wait stack-create-complete \
        --stack-name ${stack_name} \
        --region ${REGION} 2>/dev/null || \
    aws cloudformation wait stack-update-complete \
        --stack-name ${stack_name} \
        --region ${REGION} 2>/dev/null || true
}

# Function to deploy a stack
deploy_stack() {
    local stack_name=$1
    local template_file=$2
    local parameters=$3

    print_info "Deploying stack: ${stack_name}"

    if aws cloudformation describe-stacks --stack-name ${stack_name} --region ${REGION} >/dev/null 2>&1; then
        print_warn "Stack ${stack_name} already exists. Updating..."
        aws cloudformation update-stack \
            --stack-name ${stack_name} \
            --template-body file://${template_file} \
            --parameters ${parameters} \
            --capabilities CAPABILITY_NAMED_IAM \
            --region ${REGION} 2>/dev/null || {
                if [ $? -eq 254 ]; then
                    print_info "No updates to be performed for ${stack_name}"
                else
                    print_error "Failed to update stack ${stack_name}"
                    return 1
                fi
            }
    else
        print_info "Creating new stack: ${stack_name}"
        aws cloudformation create-stack \
            --stack-name ${stack_name} \
            --template-body file://${template_file} \
            --parameters ${parameters} \
            --capabilities CAPABILITY_NAMED_IAM \
            --region ${REGION}
    fi

    wait_for_stack ${stack_name}
    print_info "Stack ${stack_name} deployment complete!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            REGION="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "This script deploys S3 buckets and IAM roles for ReasonForge."
            echo "Training instance should be created manually in AWS Console."
            echo ""
            echo "Options:"
            echo "  --region REGION         AWS region (default: us-east-1)"
            echo "  --environment ENV       Environment name (default: dev)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Deploy with default settings"
            echo "  $0"
            echo ""
            echo "  # Deploy to a different region"
            echo "  $0 --region us-west-2"
            echo ""
            echo "  # Deploy to staging environment"
            echo "  $0 --environment staging"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Common parameters
COMMON_PARAMS="ParameterKey=ProjectName,ParameterValue=${PROJECT_NAME} ParameterKey=Environment,ParameterValue=${ENVIRONMENT}"

print_info "=========================================="
print_info "ReasonForge Infrastructure Deployment"
print_info "=========================================="
print_info "Project: ${PROJECT_NAME}"
print_info "Environment: ${ENVIRONMENT}"
print_info "Region: ${REGION}"
echo ""

# Deploy S3 Buckets
deploy_stack \
    "${PROJECT_NAME}-s3-buckets-${ENVIRONMENT}" \
    "cloudformation/01-s3-buckets.yaml" \
    "${COMMON_PARAMS}"
echo ""

# Deploy IAM Roles
deploy_stack \
    "${PROJECT_NAME}-iam-roles-${ENVIRONMENT}" \
    "cloudformation/02-iam-roles.yaml" \
    "${COMMON_PARAMS}"
echo ""

# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Print summary
print_info "=========================================="
print_info "Deployment Complete!"
print_info "=========================================="
echo ""
print_info "S3 Buckets Created:"
echo "  • Models:   ${PROJECT_NAME}-models-${ENVIRONMENT}-${ACCOUNT_ID}"
echo "  • Adapters: ${PROJECT_NAME}-adapters-${ENVIRONMENT}-${ACCOUNT_ID}"
echo "  • Datasets: ${PROJECT_NAME}-datasets-${ENVIRONMENT}-${ACCOUNT_ID}"
echo ""
print_info "IAM Role Created:"
echo "  • ${PROJECT_NAME}-training-instance-role-${ENVIRONMENT}"
echo "  • ${PROJECT_NAME}-training-instance-profile-${ENVIRONMENT}"
echo ""
print_warn "Next Steps:"
echo "  1. Create EC2 training instance manually in AWS Console"
echo "  2. Use instance type: g5.xlarge (A10G GPU)"
echo "  3. Select Deep Learning AMI GPU PyTorch"
echo "  4. Attach IAM instance profile: ${PROJECT_NAME}-training-instance-profile-${ENVIRONMENT}"
echo "  5. Configure security group for SSH (22), Jupyter (8888), TensorBoard (6006)"
echo ""
print_info "For detailed instance setup instructions, see README.md"
