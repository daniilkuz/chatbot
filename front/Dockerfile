# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the backend code into the container
COPY . .

# Install dependencies (adjust this as needed for your requirements)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the backend port (modify as needed)
EXPOSE 5000

# Define environment variable for backend (if needed)
ENV FLASK_APP=app.py

# Command to run the backend server
CMD ["python", "app.py"]
