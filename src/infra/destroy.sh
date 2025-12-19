#!/bin/bash
set -e

# ReasonForge Infrastructure Destruction Script
# This script destroys the CloudFormation stacks in reverse order

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

# Function to wait for stack deletion
wait_for_deletion() {
    local stack_name=$1
    print_info "Waiting for stack ${stack_name} to be deleted..."

    aws cloudformation wait stack-delete-complete \
        --stack-name ${stack_name} \
        --region ${REGION} 2>/dev/null || true
}

# Function to delete a stack
delete_stack() {
    local stack_name=$1

    if aws cloudformation describe-stacks --stack-name ${stack_name} --region ${REGION} >/dev/null 2>&1; then
        print_warn "Deleting stack: ${stack_name}"

        aws cloudformation delete-stack \
            --stack-name ${stack_name} \
            --region ${REGION}

        wait_for_deletion ${stack_name}
        print_info "Stack ${stack_name} deleted!"
    else
        print_info "Stack ${stack_name} does not exist. Skipping..."
    fi
}

# Function to empty S3 bucket
empty_bucket() {
    local bucket_name=$1

    if aws s3 ls "s3://${bucket_name}" --region ${REGION} 2>/dev/null; then
        print_warn "Emptying bucket: ${bucket_name}"
        aws s3 rm "s3://${bucket_name}" --recursive --region ${REGION}
        print_info "Bucket ${bucket_name} emptied!"
    else
        print_info "Bucket ${bucket_name} does not exist. Skipping..."
    fi
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
            echo "This script destroys S3 buckets and IAM roles created for ReasonForge."
            echo "Training instance should be terminated manually in AWS Console."
            echo ""
            echo "Options:"
            echo "  --region REGION         AWS region (default: us-east-1)"
            echo "  --environment ENV       Environment name (default: dev)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Destroy infrastructure"
            echo "  $0"
            echo ""
            echo "  # Destroy in different region"
            echo "  $0 --region us-west-2"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_warn "=========================================="
print_warn "WARNING: This will destroy infrastructure!"
print_warn "=========================================="
print_info "Project: ${PROJECT_NAME}"
print_info "Environment: ${ENVIRONMENT}"
print_info "Region: ${REGION}"
echo ""
print_warn "This will destroy:"
echo "  • S3 Buckets (all data will be PERMANENTLY deleted)"
echo "  • IAM Roles and Policies"
echo ""
print_info "Note: Training instance should be terminated manually in AWS Console"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    print_info "Destruction cancelled."
    exit 0
fi

echo ""
print_info "Starting infrastructure destruction..."
echo ""

# Delete IAM Roles first
delete_stack "${PROJECT_NAME}-iam-roles-${ENVIRONMENT}"
echo ""

# Delete S3 Buckets
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Empty and delete buckets
print_warn "Emptying S3 buckets before deletion..."
empty_bucket "${PROJECT_NAME}-models-${ENVIRONMENT}-${ACCOUNT_ID}"
empty_bucket "${PROJECT_NAME}-adapters-${ENVIRONMENT}-${ACCOUNT_ID}"
empty_bucket "${PROJECT_NAME}-datasets-${ENVIRONMENT}-${ACCOUNT_ID}"
echo ""

delete_stack "${PROJECT_NAME}-s3-buckets-${ENVIRONMENT}"
echo ""

print_info "=========================================="
print_info "Destruction Complete!"
print_info "=========================================="
echo ""
print_warn "All CloudFormation-managed infrastructure has been destroyed."
print_warn "All data in S3 buckets has been permanently deleted."
echo ""
print_info "If you created a training instance manually, terminate it in AWS Console."
