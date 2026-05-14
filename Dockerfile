# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /workspace

# Install system dependencies (if needed for any packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . /workspace

# Install Python dependencies
# Create requirements.txt if it doesn't exist, or install from existing one
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scipy \
    wrds

# Set environment variables for WRDS (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1

# Default command - open Python REPL, or can be overridden
CMD ["python"]
