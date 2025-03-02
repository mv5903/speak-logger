## Installation
Only Python 3.12 is officially supported. As of 3/1/25, 3.13 or greater throws errors during requirements installation.

### Windows
It is strongly encouraged NOT to use WSL as your audio devices and gpu(s) may not get detected correctly. Your best bet is Powershell.
```sh
# Create virtual environment - make sure that you confirm that venv\pyvenv.cfg is using the correct version!
path/to/python3.12 -m venv venv\ 

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements (this may take several minutes)
pip install -r requirements-win.txt
```

### Mac/Linux
coming soon


## Usage

Add person
```sh
python app.py add-person ./Matthew.mp3 "Matthew"
```

Live listen
```sh
python app.py log-words
```

Review unknown recordings
```sh
python app.py review
```

Access db
```sh
sqlite3 voice_logs.db
```

Clear all data (dangerous!)
```sh
chmod +x reset.sh
./reset.sh
```


### Mac Extras
```sh
# for audio playback in review mode
brew install sox 
```


### Windows Neccessities
```sh
# pytorch with cuda support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 

# ffmpeg for windows (make sure to restart your terminal)
winget install "FFmpeg (Essentials Build)" 
```
