output "aws_region" {
  description = "AWS region for deployed resources"
  value       = var.aws_region
}

output "bucket_name" {
  description = "S3 bucket for raw .zst and prepared .csv files"
  value       = aws_s3_bucket.data.bucket
}

output "ecr_url" {
  description = "ECR repository URL for Lambda container images"
  value       = aws_ecr_repository.lambda.repository_url
}

output "download_function_name" {
  description = "Lambda function name for download"
  value       = aws_lambda_function.download.function_name
}

output "prepare_function_name" {
  description = "Lambda function name for prepare"
  value       = aws_lambda_function.prepare.function_name
}
