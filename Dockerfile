FROM python:3.10-slim

# Install system utilities (curl is critical for Health Checks)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Initialize the work directory
WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock* ./

# Install ALL dependencies (FastAPI + Streamlit + Redis)
RUN uv sync --frozen

# Copy the rest of the application code
COPY . .

# NOTE: No CMD or EXPOSE here. We define start commands in ECS.