## Usage
Add person
```sh
python app.py add-person /Users/matt/GitHub/speak-logger/RobertSpeaking.m4a "Robert Vandenberg"
```

Live listen
```sh
python app.py log-words
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