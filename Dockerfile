# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /app
COPY . /

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Streamlit
EXPOSE 8080

# Command to run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
