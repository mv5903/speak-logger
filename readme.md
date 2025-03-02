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
brew install sox (for audio playback in review mode)


### Windows Extras
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (pytorch with cuda support)
