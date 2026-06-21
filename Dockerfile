# Use the stable Python 3.10 image to bypass the Rust compilation errors
FROM python:3.10

# Set the working directory inside the cloud server
WORKDIR /app

# Copy your requirements file first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# CRITICAL: Hugging Face Spaces strictly requires web servers to run on port 7860
EXPOSE 7860

# Start the FastAPI server using the mandatory Hugging Face port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]