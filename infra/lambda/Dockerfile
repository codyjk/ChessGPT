FROM public.ecr.aws/lambda/python:3.11

COPY infra/lambda/requirements-lambda.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-lambda.txt

# Copy only the chessgpt package (no torch, no python-chess needed)
COPY src/chessgpt/ ${LAMBDA_TASK_ROOT}/chessgpt/

# Default handler -- overridden per function in Terraform
CMD ["chessgpt.lambdas.download_handler.handler"]
