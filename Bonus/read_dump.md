



## 6. Containerization & Resource Management

### 6.1 Dockerfile Structure

```dockerfile
# Base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build arguments
ARG BUILD_VERSION=dev
ARG VECTOR_STRATEGY=s3

# Environment variables
ENV BUILD_VERSION=${BUILD_VERSION} \
    VECTOR_STRATEGY=${VECTOR_STRATEGY} \
    PORT=8080

# Configure supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port
EXPOSE 8080

# Command to run
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

### 6.2 Supervisord Configuration

```ini
[supervisord]
nodaemon=true
user=root
logfile=/dev/null
logfile_maxbytes=0

[program:gunicorn]
command=gunicorn app:app --bind 0.0.0.0:8080 --workers 4 --threads 2
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true

[program:vector-sync]
command=python -m utils.vector_sync_service
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
```

## 7. Monitoring & Observability

### 7.1 Key Metrics

- **Application Performance**:
  - Request latency (p50, p90, p99)
  - Query throughput
  - Error rates

- **RAG-Specific Metrics**:
  - Vector retrieval latency
  - Relevance scores
  - Token usage (for LLM)
  - Cache hit/miss rates

- **Resource Utilization**:
  - CPU/Memory usage
  - S3 GET requests
  - Network I/O

### 7.2 Logging Strategy

- **Structured JSON Logs**:
  - Request ID for traceability
  - User query (anonymized)
  - Retrieved documents
  - LLM prompt tokens
  - Response generation time

- **Log Destinations**:
  - CloudWatch Logs
  - Optional ELK/DataDog integration

### 7.3 Alerting

- **CloudWatch Alarms**:
  - 5xx error rate > 1%
  - p99 latency > 2s
  - ECS service < 100% healthy tasks

- **Custom RAG Alerts**:
  - Zero vector results > 10% of queries
  - LLM token usage spike
  - Vectorstore access errors


## 9. Deployment Configuration

### 9.1 ECS Task Definition

The task definition includes environment variables that control how the container accesses vectorstore data:

```json
{
  "containerDefinitions": [
    {
      "name": "rag-chatbot",
      "image": "{{ ECR_REGISTRY }}/rag-chatbot:{{ GITHUB_SHA }}",
      "essential": true,
      "environment": [
        {
          "name": "VECTORSTORE_TYPE",
          "value": "{{ VECTOR_STRATEGY }}"
        },
        {
          "name": "VECTORSTORE_S3_PATH",
          "value": "s3://{{ S3_BUCKET }}/vectorstores/{{ GITHUB_SHA }}/"
        },
        {
          "name": "VECTORSTORE_LOCAL_PATH",
          "value": "/app/vectorstore"
        },
        {
          "name": "CACHE_STRATEGY",
          "value": "hybrid"
        },
        {
          "name": "CACHE_TTL_SECONDS",
          "value": "3600"
        }
      ],
      "portMappings": [
        {
          "containerPort": 8080,
          "hostPort": 8080,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-chatbot",
          "awslogs-region": "{{ AWS_REGION }}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "executionRoleArn": "arn:aws:iam::{{ AWS_ACCOUNT_ID }}:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::{{ AWS_ACCOUNT_ID }}:role/rag-chatbot-task-role",
  "family": "rag-chatbot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048"
}
```










