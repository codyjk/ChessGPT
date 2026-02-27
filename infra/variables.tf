variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used as prefix for resource naming"
  type        = string
  default     = "chessgpt"
}

variable "bucket_suffix" {
  description = "Suffix appended to S3 bucket name for global uniqueness"
  type        = string
}
