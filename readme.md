## Usage
Add person
```sh
python app.py add-person /Users/matt/GitHub/speak-logger/Matthew.mp3 "Matthew"
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

brew install sox