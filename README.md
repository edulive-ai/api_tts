# Text-to-Speech API

A Flask-based API service that converts text to speech.

## System Requirements

### Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
sudo apt-get install -y libsndfile1-dev
```

## Setup Environment

### Option 1: Using Anaconda

```bash
# Create a new conda environment
conda create -n tts_env python=3.8
conda activate tts_env
```

### Option 2: Using Python Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt

# Run the setup script to install additional TTS components
python setup_tts.py
```

## Running the API

Start the Flask API server:

```bash
python main.py
```

## API Usage

### Convert Text to Speech

Use curl to call the API:

```bash
curl -X POST https://intern.nnq962.pro/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Xin chào, đây là mẫu thử nghiệm, các bạn hãy thử nhé hihi"}' \
     --output output.wav
```

You can modify the "text" field to customize the speech content.

## Notes

- The API endpoint is hosted at `https://intern.nnq962.pro/tts`
- The output is saved as a WAV file
- Make sure to include Vietnamese tone marks for proper pronunciation