# Use the slim Python image as the base image
FROM python:3.12-slim

# Update the package list and install required system dependencies
RUN apt update -y && apt install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libmariadb-dev-compat \
    libmariadb-dev \
    pkg-config \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run your app
CMD ["python3", "app.py"]