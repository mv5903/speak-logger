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

# pytorch with cuda support - required to utilize your gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 

# ffmpeg for windows - make sure to restart your terminal and reactivate the virtual environment
winget install "FFmpeg (Essentials Build)" 
```

### Mac/Linux
coming soon
```sh
# for audio playback in review mode
brew install sox 
```


## Usage

### Add person
```sh
python app.py add-person ./Matthew.mp3 "Matthew"
```
File types `.wav`, `.mp3`, and `.m4a` are supported. <br>
When adding a person for the first time, use the longest clip you have possible. You can use shorter ones, but you may need more manual reviewing for the first couple runs until you notice a confidence increase.

### Live listen
```sh
python app.py log-words
```

### Review unknown recordings
```sh
python app.py review
```

### Update a person's speech model manually with additional recordings
```sh
python app.py update-profile "Name"
```

### Reset a profile
```sh
python app.py reset-profile "Name"
```

### Reset the entire database (probably *not* a good idea unless you're testing)
```sh
# Windows
./reset.ps1

# Linux/Mac
./reset.sh
```

### Access db
```sh
sqlite3 voice_logs.db
```
You can also install a vscode extension to view the database file. 