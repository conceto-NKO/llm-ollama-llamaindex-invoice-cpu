# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run both scripts when the container launches
# Note: This is a placeholder. The actual command will be detailed below.
CMD ["python", "app.py"]
