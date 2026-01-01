# Packaging Your Policy Server with Docker

## Overview

This guide explains how to package your **Python policy server** into a Docker container, so that it can be run and tested consistently by instructors or TAs.

---

## 1. Project Structure

Your project may look like:

```
student_repo/
├── serve_policy.py
└── ...other files
```

---

## 2. Dockerfile Template

Create a file named `Dockerfile` in the root of your repo:

```dockerfile
# Use a proper image
FROM ...

# Set working directory
WORKDIR /app

# Copy the needed files
COPY ...

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN ...

# Default command to run your policy server
CMD ["python", "serve_policy.py"]
```

> Make sure your server listens on `0.0.0.0`, not `127.0.0.1`.

---

## 3. Build Docker Image

Run the following in your project directory:

```bash
docker build -t <your-policy-server> .
```

---

## 4. Run the Server

The instructor or TA can run the container and map any host port to the container port:

```bash
docker run --rm --gpus all -p 8000:8000 -v $(pwd)/models:/models <your-policy-server> [args for serve_policy.py]
```

* Replace `8000` with any desired port.
* The server inside the container must listen on `0.0.0.0`.
* run with GPU support (requires NVIDIA Container Toolkit)

---

## 5. Submission Guidelines

1. **Provide a Dockerfile** in your repository, or a pre-built Docker image file (`.tar`).
2. Make sure your server (`serve_policy.py`) listens on **0.0.0.0** so it can accept connections from outside the container.
3. The image should be runnable by the TA without additional setup.

* Export image as `.tar`:

```bash
docker save -o your-policy-server.tar your-policy-server
```

* TA can import and run:

```bash
docker load -i your-policy-server.tar
docker run ...
```