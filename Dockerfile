# Use a Python 3.10 slim image as the base. Python 3.10 has good compatibility with PyTorch.
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# --- ADD THIS LINE HERE ---
RUN pip install --upgrade pip
# --- END ADDITION ---

# Install Python dependencies.
# --no-cache-dir saves space by not caching package data.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Google Cloud Run provides a PORT environment variable. Streamlit needs to listen on it.
# We set a default of 8080, which Cloud Run often uses.
ENV PORT 8080

# Expose the port that Streamlit will listen on inside the container
EXPOSE 8080

# Define the command to run your Streamlit application
# --server.port listens on the specified PORT environment variable
# --server.address 0.0.0.0 is crucial for listening on all network interfaces inside the container
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
