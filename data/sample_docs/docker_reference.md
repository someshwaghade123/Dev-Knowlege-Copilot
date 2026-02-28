# Docker — Complete Reference Guide for Developers

Docker is a platform for developing, shipping, and running applications in containers. Containers allow a developer to package up an application with all parts it needs, such as libraries and other dependencies, and ship it all out as one package.

## Core Concepts

### Images vs Containers

- **Image**: A read-only template with instructions for creating a Docker container. Like a class in OOP.
- **Container**: A runnable instance of an image. Like an object (instance of a class).

### Dockerfile

A text document that contains all the commands to assemble an image.

```dockerfile
# Use official Python runtime as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker CLI Commands

### Building Images

```bash
# Build an image from a Dockerfile
docker build -t my-app:latest .

# Build with specific Dockerfile
docker build -f Dockerfile.prod -t my-app:prod .

# Build without cache
docker build --no-cache -t my-app:latest .
```

### Running Containers

```bash
# Run a container
docker run my-app:latest

# Run in detached mode (background)
docker run -d my-app:latest

# Map ports (host:container)
docker run -p 8000:8000 my-app:latest

# Run with environment variables
docker run -e DATABASE_URL=sqlite:///./db.sqlite my-app:latest

# Run with volume mount
docker run -v /host/path:/container/path my-app:latest

# Run interactively
docker run -it my-app:latest /bin/bash

# Auto-remove container when stopped
docker run --rm my-app:latest
```

### Container Management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop <container_id>

# Remove a container
docker rm <container_id>

# View container logs
docker logs <container_id>

# Follow logs in real-time
docker logs -f <container_id>

# Execute command in running container
docker exec -it <container_id> /bin/bash
```

### Image Management

```bash
# List images
docker images

# Remove an image
docker rmi my-app:latest

# Pull from Docker Hub
docker pull postgres:15

# Push to Docker Hub
docker push username/my-app:latest
```

## Docker Compose

Docker Compose is for defining and running multi-container applications.

### docker-compose.yml Example

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Docker Compose Commands

```bash
# Start all services
docker compose up

# Start in detached mode
docker compose up -d

# Build and start
docker compose up --build

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# View logs
docker compose logs -f backend

# Scale a service
docker compose up --scale backend=3
```

## Volumes and Persistence

```bash
# Create a named volume
docker volume create my_data

# List volumes
docker volume ls

# Inspect a volume
docker volume inspect my_data

# Remove a volume
docker volume rm my_data
```

## Networking

```bash
# Create a network
docker network create my_network

# Run container on specific network
docker run --network my_network my-app:latest

# List networks
docker network ls
```

## Best Practices

### Layer Caching

Order your Dockerfile instructions from least to most frequently changed:

```dockerfile
# 1. Base image (rarely changes)
FROM python:3.11-slim

# 2. System dependencies (rarely changes)
RUN apt-get update && apt-get install -y curl

# 3. Python dependencies (changes when requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. Application code (changes frequently)
COPY . .
```

### Multi-stage Builds

Reduce final image size by using build stages:

```dockerfile
# Build stage
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["python", "main.py"]
```

### .dockerignore

Exclude unnecessary files from the build context:

```
__pycache__/
*.pyc
*.pyo
.env
.git
.gitignore
node_modules/
*.log
data/faiss_index.bin
```

## Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

## Environment Variables

```bash
# Pass env file
docker run --env-file .env my-app:latest

# Override specific variable
docker run -e LOG_LEVEL=DEBUG my-app:latest
```
