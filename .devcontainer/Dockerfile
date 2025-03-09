# Use a base image with Python installed
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Update package lists and install COLMAP
RUN apt-get update && \
    apt-get install -y colmap

# (Optional) Install other system dependencies you need for COLMAP or your project
# RUN apt-get install -y <other-dependencies>

# Set the working directory (optional)
WORKDIR /workspace