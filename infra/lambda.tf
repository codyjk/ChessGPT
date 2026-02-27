resource "aws_lambda_function" "download" {
  function_name = "${var.project_name}-download"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.lambda.repository_url}:latest"
  timeout       = 900 # 15 minutes
  memory_size   = 512

  ephemeral_storage {
    size = 10240 # 10 GB
  }

  image_config {
    command = ["chessgpt.lambdas.download_handler.handler"]
  }
}

resource "aws_lambda_function" "prepare" {
  function_name = "${var.project_name}-prepare"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.lambda.repository_url}:latest"
  timeout       = 900 # 15 minutes
  memory_size   = 2048

  ephemeral_storage {
    size = 10240 # 10 GB
  }

  image_config {
    command = ["chessgpt.lambdas.prepare_handler.handler"]
  }
}

resource "aws_cloudwatch_log_group" "download" {
  name              = "/aws/lambda/${aws_lambda_function.download.function_name}"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "prepare" {
  name              = "/aws/lambda/${aws_lambda_function.prepare.function_name}"
  retention_in_days = 14
}
