import argparse
import os
import time
import sqlite3
import numpy as np
import torch
import librosa
import sounddevice as sd
import whisper
import wave
from datetime import datetime, timedelta
import re
from speechbrain.inference import SpeakerRecognition
from pydub import AudioSegment
import tempfile

class VoiceLogger:
    def __init__(self, db_path="voice_logs.db", unknown_dir="unknown_speakers"):
        # Initialize database
        self.db_path = db_path
        self.setup_database()
        
        # Create directory for unknown speaker fragments
        self.unknown_dir = unknown_dir
        os.makedirs(self.unknown_dir, exist_ok=True)
        
        # Initialize models
        self.speech_recognizer = whisper.load_model("base")
        self.speaker_recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        
        # Speaker embeddings storage
        self.speakers = {}
        self.load_speakers()
        
        # Audio settings
        self.sample_rate = 16000
        self.buffer_duration = 5  # seconds
        self.confidence_threshold = 0.75  # Minimum confidence to identify a speaker
        self.speech_confidence_threshold = 0.6  # Minimum confidence for speech recognition
        
    def setup_database(self):
        """Set up SQLite database for storing word frequencies and conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            embedding BLOB
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS word_logs (
            id INTEGER PRIMARY KEY,
            speaker_id INTEGER,
            word TEXT,
            frequency INTEGER DEFAULT 1,
            timestamp TEXT,
            confidence REAL DEFAULT 0.0,
            FOREIGN KEY (speaker_id) REFERENCES speakers(id)
        )
        ''')
        
        # Add confidence column if it doesn't exist
        try:
            cursor.execute("SELECT confidence FROM word_logs LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE word_logs ADD COLUMN confidence REAL DEFAULT 0.0")
            print("Added confidence column to word_logs table")
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            speaker_id INTEGER,
            listener_id INTEGER,
            timestamp TEXT,
            content TEXT,
            FOREIGN KEY (speaker_id) REFERENCES speakers(id),
            FOREIGN KEY (listener_id) REFERENCES speakers(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS unknown_fragments (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            transcription TEXT,
            timestamp TEXT,
            is_labeled BOOLEAN DEFAULT 0,
            assigned_speaker_id INTEGER,
            FOREIGN KEY (assigned_speaker_id) REFERENCES speakers(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_speakers(self):
        """Load speaker embeddings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, embedding FROM speakers")
        for name, embedding_blob in cursor.fetchall():
            self.speakers[name] = np.frombuffer(embedding_blob, dtype=np.float32)
            
        conn.close()
        print(f"Loaded {len(self.speakers)} speakers: {', '.join(self.speakers.keys())}")
    
    def add_person(self, audio_file, name):
        """Add a new person based on audio recording"""
        if not os.path.exists(audio_file):
            print(f"Error: File {audio_file} not found")
            return
            
        print(f"Processing audio file to create profile for {name}...")
        
        # Check if file is m4a format and convert if needed
        temp_file = None
        try:
            if (audio_file.lower().endswith('.m4a')):
                print("Converting M4A file to WAV format...")
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                audio = AudioSegment.from_file(audio_file, format="m4a")
                audio.export(temp_file.name, format="wav")
                audio_file = temp_file.name
                print(f"Converted to temporary file: {audio_file}")
            
            # Load audio file
            audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            
            # Extract speaker embedding
            embedding = self.speaker_recognizer.encode_batch(torch.tensor(audio).unsqueeze(0))
            embedding_np = embedding.squeeze().cpu().detach().numpy()
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("INSERT OR REPLACE INTO speakers (name, embedding) VALUES (?, ?)", 
                          (name, embedding_np.tobytes()))
            
            conn.commit()
            conn.close()
            
            # Update local cache
            self.speakers[name] = embedding_np
            
            print(f"Successfully added {name} to the system")
            
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def identify_speaker(self, audio_segment):
        """Identify the speaker from an audio segment"""
        if not self.speakers:
            return None, 0
            
        # Convert audio to tensor and extract embedding
        audio_tensor = torch.tensor(audio_segment).unsqueeze(0)
        embedding = self.speaker_recognizer.encode_batch(audio_tensor)
        current_embedding = embedding.squeeze().cpu().detach().numpy()
        
        # Compare with known speakers
        best_score = -1
        best_speaker = None
        
        for name, speaker_embedding in self.speakers.items():
            # Calculate cosine similarity
            similarity = np.dot(current_embedding, speaker_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(speaker_embedding)
            )
            
            if similarity > best_score:
                best_score = similarity
                best_speaker = name
                
        # Return the best speaker only if confidence is above threshold
        if best_score > self.confidence_threshold:
            return best_speaker, best_score
        else:
            return None, best_score
    
    def detect_conversation(self, current_speaker, audio_segment):
        """
        Basic conversation detection - assumes person is talking to the 
        last identified different speaker
        """
        return None  # Placeholder for the listener
    
    def save_unknown_fragment(self, audio_chunk, transcription):
        """Save an audio fragment with unknown speaker for later review"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        filename = f"unknown_{timestamp.replace(' ', '_').replace(':', '-').replace('.', '-')}.wav"
        filepath = os.path.join(self.unknown_dir, filename)
        
        # Save the audio file
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_chunk * 32767).astype(np.int16).tobytes())
        
        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO unknown_fragments (filename, transcription, timestamp)
            VALUES (?, ?, ?)
        """, (filename, transcription, timestamp))
        
        conn.commit()
        conn.close()
        
        print(f"Saved unknown fragment to {filepath}")
        return filename
    
    def log_words(self):
        """Continuously listen and log words with speaker identification"""
        if not self.speakers:
            print("No speakers added yet. Please add at least one speaker first.")
            return
            
        print("Starting real-time voice logging. Press Ctrl+C to stop.")
        print(f"Known speakers: {', '.join(self.speakers.keys())}")
        print(f"Speech recognition confidence threshold: {self.speech_confidence_threshold}")
        print(f"Unknown speaker fragments will be saved to {self.unknown_dir}")
        
        try:
            last_speaker = None
            buffer = []
            stream_start_time = datetime.now()
            last_chunk_time = stream_start_time
            
            def audio_callback(indata, frames, time, status):
                """Callback for audio streaming"""
                if status:
                    print(f"Status: {status}")
                buffer.extend(indata[:, 0])  # Take the first channel if stereo
            
            # Start streaming from microphone
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=self.sample_rate):
                while True:
                    # Process in chunks
                    if len(buffer) >= self.sample_rate * self.buffer_duration:
                        audio_chunk = np.array(buffer[:self.sample_rate * self.buffer_duration])
                        buffer = buffer[self.sample_rate * self.buffer_duration:]
                        
                        # Get current time at the beginning of processing this chunk
                        chunk_start_time = datetime.now()
                        
                        # Calculate chunk offset from stream start (for word timestamp calculation)
                        chunk_offset = (chunk_start_time - stream_start_time).total_seconds()
                        
                        # Transcribe speech with detailed segment info to get confidence scores
                        result = self.speech_recognizer.transcribe(audio_chunk.astype(np.float32))
                        text = result["text"].strip()
                        
                        if text:  # Only process if there's actual speech
                            # Identify speaker
                            speaker, confidence = self.identify_speaker(audio_chunk)
                            
                            if speaker:
                                print(f"[{speaker} ({confidence:.2f})]: {text}")
                                
                                # Identify potential listener
                                listener = self.detect_conversation(speaker, audio_chunk) if speaker != last_speaker else None
                                
                                # Process each segment with its confidence score
                                if 'segments' in result:
                                    for segment in result['segments']:
                                        # Check if segment meets confidence threshold
                                        segment_confidence = segment.get('confidence', 0)
                                        if segment_confidence < self.speech_confidence_threshold:
                                            print(f"  Skipping low confidence segment: '{segment['text']}' ({segment_confidence:.2f})")
                                            continue
                                        
                                        # Process words in this segment
                                        segment_text = segment['text'].strip()
                                        segment_words = segment_text.split()
                                        segment_start = segment.get('start', 0) + chunk_offset
                                        segment_end = segment.get('end', self.buffer_duration) + chunk_offset
                                        
                                        # Calculate approximate word positions within segment
                                        word_count = len(segment_words)
                                        for i, word in enumerate(segment_words):
                                            # Calculate word timestamp
                                            word_time_offset = segment_start + (i / max(1, word_count-1)) * (segment_end - segment_start)
                                            word_timestamp = (stream_start_time + 
                                                           timedelta(seconds=word_time_offset))
                                            
                                            # Format timestamp with millisecond precision
                                            timestamp_str = word_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                            
                                            # Log word with confidence score
                                            self.log_word_to_database(speaker, word.lower(), timestamp_str, 
                                                                  listener, segment_confidence)
                                            
                                            print(f"  Word: '{word}' (Confidence: {segment_confidence:.2f})")
                                else:
                                    # Fallback for old API or if segments not available
                                    words = text.split()
                                    for i, word in enumerate(words):
                                        # Calculate approximate timestamp for each word
                                        word_time_offset = chunk_offset + (i / max(1, len(words)-1)) * self.buffer_duration
                                        word_timestamp = (stream_start_time + 
                                                        timedelta(seconds=word_time_offset))
                                        timestamp_str = word_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                        # Use a default confidence of 1.0 if not available
                                        self.log_word_to_database(speaker, word.lower(), timestamp_str, listener, 1.0)
                                
                                last_speaker = speaker
                            else:
                                print(f"[Unknown Speaker ({confidence:.2f})]: {text}")
                                # Save fragment for manual review
                                self.save_unknown_fragment(audio_chunk, text)
                        
                        # Update last chunk time
                        last_chunk_time = chunk_start_time
                        
                    time.sleep(0.1)  # Small delay to prevent CPU hogging
                        
        except KeyboardInterrupt:
            print("\nStopping voice logging.")
    
    def log_word_to_database(self, speaker, word, timestamp, listener=None, confidence=1.0, conn=None):
        """Log a single word with timestamp and confidence to the database"""
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
            
        cursor = conn.cursor()
        
        # Clean the word (remove punctuation)
        word = re.sub(r'[^\w\s]', '', word).strip()
        if not word:  # Skip empty strings
            if close_conn:
                conn.close()
            return
        
        # Get speaker ID
        cursor.execute("SELECT id FROM speakers WHERE name = ?", (speaker,))
        speaker_id = cursor.fetchone()[0]
        
        # Insert word with timestamp and confidence
        cursor.execute("""
            INSERT INTO word_logs (speaker_id, word, timestamp, confidence)
            VALUES (?, ?, ?, ?)
        """, (speaker_id, word, timestamp, confidence))
        
        # Update word frequency (separate query)
        cursor.execute("""
            UPDATE word_logs 
            SET frequency = (
                SELECT COUNT(*) 
                FROM word_logs 
                WHERE speaker_id = ? AND word = ?
            )
            WHERE speaker_id = ? AND word = ? AND rowid = last_insert_rowid()
        """, (speaker_id, word, speaker_id, word))
        
        # If this word is part of a conversation, log it
        if listener:
            cursor.execute("SELECT id FROM speakers WHERE name = ?", (listener,))
            result = cursor.fetchone()
            if result:
                listener_id = result[0]
                
                cursor.execute("""
                    INSERT INTO conversations (speaker_id, listener_id, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (speaker_id, listener_id, word, timestamp))
        
        conn.commit()
        if close_conn:
            conn.close()
    
    def review_unknown_fragments(self):
        """Review and label unknown speech fragments"""
        # Use a single connection throughout the method
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get list of unlabeled fragments
            cursor.execute("SELECT id, filename, transcription, timestamp FROM unknown_fragments WHERE is_labeled = 0")
            fragments = cursor.fetchall()
            
            if not fragments:
                print("No unlabeled fragments found.")
                return
            
            print(f"Found {len(fragments)} unlabeled fragments. Ready to review.")
            
            # Get list of speakers
            speaker_names = list(self.speakers.keys())
            for i, name in enumerate(speaker_names, 1):
                print(f"{i}. {name}")
            
            for frag_id, filename, transcription, timestamp in fragments:
                filepath = os.path.join(self.unknown_dir, filename)
                if not os.path.exists(filepath):
                    print(f"Warning: File {filepath} not found, skipping.")
                    continue
                
                print(f"\nPlaying fragment: {filename}")
                print(f"Timestamp: {timestamp}")
                print(f"Transcription: {transcription}")
                
                # Play the audio (this relies on system audio player)
                os.system(f"play {filepath}" if os.name != 'nt' else f"start {filepath}")
                
                # Ask for speaker identification
                while True:
                    choice = input(f"Who is speaking? (1-{len(speaker_names)}, 'n' for new speaker, 's' to skip): ")
                    
                    if choice.lower() == 's':
                        break
                        
                    if choice.lower() == 'n':
                        # Add new speaker - need to close our current connection since add_person will create its own
                        conn.close()
                        name = input("Enter name for new speaker: ")
                        self.add_person(filepath, name)
                        # Reopen our connection
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        speaker_id = cursor.execute("SELECT id FROM speakers WHERE name = ?", (name,)).fetchone()[0]
                        
                        # Update fragment
                        cursor.execute("""
                            UPDATE unknown_fragments 
                            SET is_labeled = 1, assigned_speaker_id = ? 
                            WHERE id = ?
                        """, (speaker_id, frag_id))
                        
                        # Also log the words with original timestamps - using our connection
                        words = transcription.split()
                        for i, word in enumerate(words):
                            word_timestamp = (datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f") + 
                                            timedelta(milliseconds=i*100)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            self.log_word_to_database(name, word.lower(), word_timestamp, conn=conn)
                        
                        conn.commit()
                        break
                        
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(speaker_names):
                            name = speaker_names[idx]
                            speaker_id = cursor.execute("SELECT id FROM speakers WHERE name = ?", (name,)).fetchone()[0]
                            
                            # Update fragment
                            cursor.execute("""
                                UPDATE unknown_fragments 
                                SET is_labeled = 1, assigned_speaker_id = ? 
                                WHERE id = ?
                            """, (speaker_id, frag_id))
                            
                            # Also log the words with original timestamps - using our connection
                            words = transcription.split()
                            for i, word in enumerate(words):
                                word_timestamp = (datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f") + 
                                                timedelta(milliseconds=i*100)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                self.log_word_to_database(name, word.lower(), word_timestamp, conn=conn)
                            
                            # Update the speaker model with this new sample
                            print(f"Updating speaker model for {name} with this sample...")
                            audio, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
                            
                            # We could do a more sophisticated model update here, but for now
                            # we'll just add this as additional training data by averaging embeddings
                            new_embedding = self.speaker_recognizer.encode_batch(
                                torch.tensor(audio).unsqueeze(0)
                            ).squeeze().cpu().detach().numpy()
                            
                            # Simple averaging of embeddings (could be improved)
                            updated_embedding = (self.speakers[name] + new_embedding) / 2
                            self.speakers[name] = updated_embedding
                            
                            # Save updated embedding
                            cursor.execute("""
                                UPDATE speakers SET embedding = ? WHERE name = ?
                            """, (updated_embedding.tobytes(), name))
                            
                            conn.commit()
                            print(f"Updated speaker model for {name}")
                            break
                        else:
                            print("Invalid selection.")
                    except ValueError:
                        print("Invalid input. Please enter a number or 's' to skip.")
            
        finally:
            # Make sure to close the connection no matter what
            conn.commit()
            conn.close()
            print("Review completed.")

def main():
    parser = argparse.ArgumentParser(description="Voice Logger - Real-time speech recognition with speaker identification")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add person command
    add_parser = subparsers.add_parser("add-person", help="Add a new person to the system")
    add_parser.add_argument("audio_file", help="Audio file (MP3 or WAV) containing the person's voice")
    add_parser.add_argument("name", help="Name to associate with the voice")
    
    # Log words command
    log_parser = subparsers.add_parser("log-words", help="Start logging spoken words")
    
    # Review fragments command
    review_parser = subparsers.add_parser("review", help="Review and label unknown speech fragments")
    
    args = parser.parse_args()
    
    voice_logger = VoiceLogger()
    
    if args.command == "add-person":
        voice_logger.add_person(args.audio_file, args.name)
    elif args.command == "log-words":
        voice_logger.log_words()
    elif args.command == "review":
        voice_logger.review_unknown_fragments()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()