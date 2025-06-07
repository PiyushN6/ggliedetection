# Use a Python 3.10 slim image as the base
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies. Using pip for more direct control within Docker.
# We'll rely on pip to find the correct CPU version of torch for Python 3.10.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Streamlit runs on port 8501 by default, but Cloud Run expects port 8080
# Set the PORT environment variable to 8080, which Streamlit will use
ENV PORT 8080

# Expose the port Streamlit will listen on
EXPOSE 8080

# Define the command to run your Streamlit application
# --server.port listens on the specified PORT
# --server.address 0.0.0.0 is crucial for listening on all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
