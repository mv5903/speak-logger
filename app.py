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
from pydub import AudioSegment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import tempfile
import json
from colorama import init as colorama_init
from colorama import Fore

COLOR_INFO = Fore.CYAN
COLOR_ERROR = Fore.RED
COLOR_WARNING = Fore.YELLOW
COLOR_SUCCESS = Fore.GREEN
COLOR_TRANSCRIPTIONS = Fore.MAGENTA
COLOR_QUESTION = Fore.BLUE

class VoiceLogger:
    def __init__(self, db_path="voice_logs.db", unknown_dir="unknown_speakers", config_path="config.json"):
        # Config settings
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize database
        self.db_path = db_path
        self.setup_database()
        
        # Create directory for unknown speaker fragments
        self.unknown_dir = unknown_dir
        os.makedirs(self.unknown_dir, exist_ok=True)
        
        # Check for GPU availability
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available else 'cpu')
        if not cuda_available:
            print(COLOR_WARNING + "No dedicated GPU available, running on CPU. Expect slower performance.")
        else:
            print(COLOR_INFO + f"Using device: {self.device} - {torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'}")
        
        # Initialize models with GPU support if available
        print(COLOR_INFO + "Loading Whisper model...")
        self.speech_recognizer = whisper.load_model("medium", device=self.device)
        print(COLOR_INFO + "Whisper model loaded.")
        
        print(COLOR_INFO + "Loading Pyannote speaker embedding model...")
        # Use the newer model that works better for speaker identification
        self.speaker_embedding = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )
        print(COLOR_INFO + "Pyannote speaker embedding model loaded.")
        
        # Speaker embeddings storage
        self.speakers = {}
        self.load_speakers()
        
        # Audio settings
        self.sample_rate = 16000
        self.buffer_duration = 5  # seconds
        self.confidence_threshold = 0.65 
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
        print(COLOR_INFO + f"Loaded {len(self.speakers)} speakers: {', '.join(self.speakers.keys())}")
    
    def add_person(self, audio_file, name):
        """Add a new person based on audio recording"""
        if not os.path.exists(audio_file):
            print(COLOR_ERROR + f"Error: File {audio_file} not found")
            return
            
        print(COLOR_INFO + f"Processing audio file to create profile for {name}...")
        
        # Check if file is m4a or mp3 format and convert if needed
        temp_file = None
        try:
            if audio_file.lower().endswith(('.m4a', '.mp3')):
                format_type = 'm4a' if audio_file.lower().endswith('.m4a') else 'mp3'
                print(COLOR_WARNING + f"Converting {format_type} file to WAV format...")
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                try:
                    audio = AudioSegment.from_file(audio_file, format=format_type)
                    audio.export(temp_file.name, format="wav")
                    audio_file = temp_file.name
                    print(COLOR_INFO + f"Converted to temporary file: {audio_file}")
                except Exception as e:
                    print(COLOR_ERROR + f"Error converting file: {e}")
                    print(COLOR_ERROR + "Trying to process the original file")
            
            # Load audio file
            audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            
            # Ensure enough audio data (at least 2 seconds)
            if len(audio) < 2 * self.sample_rate:
                print(COLOR_WARNING + "Warning: Audio sample is too short. Repeating to create longer sample.")
                # Repeat the audio to make it longer
                repeats = max(1, int(2 * self.sample_rate / len(audio)) + 1)
                audio = np.tile(audio, repeats)[:2 * self.sample_rate]
            
            # Extract speaker embedding (using Pyannote model)
            with torch.no_grad():
                # PyAnnote expects tensor with shape [batch_size, channels, samples]
                waveform_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
                embedding_tensor = self.speaker_embedding(waveform_tensor)
                
                # Handle the case where the embedding is already a numpy array
                if isinstance(embedding_tensor, np.ndarray):
                    embedding_np = embedding_tensor
                else:
                    embedding_np = embedding_tensor.cpu().numpy()
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("INSERT OR REPLACE INTO speakers (name, embedding) VALUES (?, ?)", 
                          (name, embedding_np.tobytes()))
            
            conn.commit()
            conn.close()
            
            # Update local cache
            self.speakers[name] = embedding_np
            
            print(COLOR_SUCCESS + f"Successfully added {name} to the system")
            
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                try:
                    # Close any handles to the file first
                    import gc
                    gc.collect()  # Force garbage collection to close any handles
                    os.unlink(temp_file.name)
                except Exception as e:
                    print(COLOR_WARNING + f"Error removing temp file: {e}")
                    print(COLOR_WARNING + "This is non-critical, the file will be cleaned up automatically later.")
                
        print(COLOR_INFO + "TIP: For better results, record at least 10 seconds of clear speech")
        print(COLOR_INFO + f"TIP: The first few seconds of recognition may be less accurate")
    
    def identify_speaker(self, audio_segment):
        """Identify the speaker from an audio segment"""
        if not self.speakers:
            return None, 0
            
        # Convert audio to tensor and extract embedding
        with torch.no_grad():
            # PyAnnote expects tensor with shape [batch_size, channels, samples]
            audio_tensor = torch.tensor(audio_segment).unsqueeze(0).unsqueeze(0).to(self.device)
            embedding_tensor = self.speaker_embedding(audio_tensor)
            
            # Handle the case where the embedding is already a numpy array
            if isinstance(embedding_tensor, np.ndarray):
                current_embedding = embedding_tensor
            else:
                current_embedding = embedding_tensor.cpu().numpy()
        
        # Compare with known speakers
        best_score = -1
        best_speaker = None
        
        for name, speaker_embedding in self.speakers.items():
            # Calculate cosine similarity
            similarity = np.dot(current_embedding.flatten(), speaker_embedding.flatten()) / (
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
        
        print(COLOR_INFO + f"Saved unknown fragment to {filepath}")
        return filename
    
    def log_words(self):
        """Continuously listen and log words with speaker identification"""
        if not self.speakers:
            print("No speakers added yet. Please add at least one speaker first.")
            return
            
        print(COLOR_SUCCESS + "Starting real-time voice logging. Press Ctrl+C to stop.")
        print(COLOR_INFO + f"Known speakers: {', '.join(self.speakers.keys())}")
        print(COLOR_INFO + f"Speaker recognition confidence threshold: {self.confidence_threshold}")
        print(COLOR_INFO + f"Speech recognition confidence threshold: {self.speech_confidence_threshold}")
        print(COLOR_INFO + f"Unknown speaker fragments will be saved to {self.unknown_dir}")
        
        # Get the selected microphone device
        mic_device = self.config["microphone"]["device"]
        mic_name = self.config["microphone"]["name"]
        
        if mic_device is not None:
            print(COLOR_INFO + f"Using microphone: {mic_name} (device {mic_device})")
        else:
            print(COLOR_WARNING + "Using default system microphone. If you feel the wrong microphone is selected, use 'select-mic' command.")
        
        print(COLOR_SUCCESS + "Listening...")

        try:
            last_speaker = None
            buffer = []
            stream_start_time = datetime.now()
            
            def audio_callback(indata, frames, time, status):
                """Callback for audio streaming"""
                if status:
                    print(f"Status: {status}")
                buffer.extend(indata[:, 0])  # Take the first channel if stereo
            
            # Start streaming from microphone
            with sd.InputStream(callback=audio_callback, device=mic_device, channels=1, samplerate=self.sample_rate):
                while True:
                    # Process in chunks
                    if len(buffer) >= self.sample_rate * self.buffer_duration:
                        audio_chunk = np.array(buffer[:self.sample_rate * self.buffer_duration])
                        buffer = buffer[self.sample_rate * self.buffer_duration:]
                        
                        # Get current time at the beginning of processing this chunk
                        chunk_start_time = datetime.now()
                        
                        # Calculate chunk offset from stream start (for word timestamp calculation)
                        chunk_offset = (chunk_start_time - stream_start_time).total_seconds()
                        
                        # Detect speech activity to avoid processing silence
                        rms = np.sqrt(np.mean(audio_chunk**2))
                        if rms < 0.01:  # Skip processing if audio is too quiet
                            continue
                        
                        try:
                            # Move model inference to GPU if available
                            with torch.cuda.device(0) if torch.cuda.is_available() else torch.device("cpu"):
                                # Transcribe speech with detailed segment info to get confidence scores
                                # Use faster options with small model for real-time performance
                                result = self.speech_recognizer.transcribe(
                                    audio_chunk.astype(np.float32),
                                    language="en",
                                    task="transcribe",
                                    fp16=torch.cuda.is_available()
                                )
                                
                            text = result["text"].strip() # type: ignore
                            
                            if text:  # Only process if there's actual speech
                                # Apply repetition detection and fix
                                text = self.detect_repetition(text)
    
                                # Identify speaker
                                speaker, confidence = self.identify_speaker(audio_chunk)
                                
                                if speaker:
                                    print(COLOR_SUCCESS + f"[{speaker} ({confidence:.2f})]: {COLOR_TRANSCRIPTIONS} {text}")
                                    
                                    # Identify potential listener
                                    listener = self.detect_conversation(speaker, audio_chunk) if speaker != last_speaker else None
                                    
                                    # Process each segment with its confidence score
                                    # if 'segments' in result:
                                    #     print(f" Segments: ")
                                    #     for segment in result['segments']:
                                    #         # Make sure segment is a dictionary
                                    #         if not isinstance(segment, dict):
                                    #             print(f"  Skipping invalid segment: {segment}")
                                    #             continue
                                            
                                    #         # Check if segment meets confidence threshold
                                    #         segment_confidence = segment.get('confidence', 0)
                                    #         if segment_confidence < self.speech_confidence_threshold:
                                    #             print(f"  Skipping low confidence segment: '{segment['text']}' ({segment_confidence:.2f})")
                                    #             continue
                                            
                                    #         # Process words in this segment
                                    #         segment_text = segment['text'].strip()
                                    #         segment_words = segment_text.split()
                                    #         segment_start = segment.get('start', 0) + chunk_offset
                                    #         segment_end = segment.get('end', self.buffer_duration) + chunk_offset
                                            
                                    #         # Calculate approximate word positions within segment
                                    #         word_count = len(segment_words)
                                    #         for i, word in enumerate(segment_words):
                                    #             # Calculate word timestamp
                                    #             word_time_offset = segment_start + (i / max(1, word_count-1)) * (segment_end - segment_start)
                                    #             word_timestamp = (stream_start_time + 
                                    #                            timedelta(seconds=word_time_offset))
                                                
                                    #             # Format timestamp with millisecond precision
                                    #             timestamp_str = word_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                                
                                    #             # Log word with confidence score
                                    #             self.log_word_to_database(speaker, word.lower(), timestamp_str, 
                                    #                                   listener, segment_confidence)
                                                
                                    #             print(f"  Word: '{word}' (Confidence: {segment_confidence:.2f})")
                                    # else:
                                    # Fallback for old API or if segments not available. TODO: Figure out a better way to handle
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
                                    print(COLOR_WARNING + f"[Unknown Speaker ({confidence:.2f})]: {text}")
                                    # Save fragment for manual review
                                    self.save_unknown_fragment(audio_chunk, text)
                        except Exception as e:
                            print(COLOR_ERROR + f"Error during speech processing: {e}")
                        
                    time.sleep(0.01)  # Small delay to prevent CPU hogging
                        
        except KeyboardInterrupt:
            print(COLOR_INFO + "\nStopping voice logging.")
    
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

        clips_assigned_to_speakers = []
        
        try:
            # Get list of unlabeled fragments
            cursor.execute("SELECT id, filename, transcription, timestamp FROM unknown_fragments WHERE is_labeled = 0")
            fragments = cursor.fetchall()
            
            if not fragments:
                print(COLOR_WARNING + "No unlabeled fragments found.")
                return
            
            print(COLOR_SUCCESS + f"Found {len(fragments)} unlabeled fragments. Ready to review.")
            
            # Get list of speakers
            speaker_names = list(self.speakers.keys())
            for i, name in enumerate(speaker_names, 1):
                print(COLOR_INFO + f"{i}. {name}")
            
            for frag_id, filename, transcription, timestamp in fragments:
                filepath = os.path.join(self.unknown_dir, filename)
                if not os.path.exists(filepath):
                    print(COLOR_WARNING + f"Warning: File {filepath} not found, skipping.")
                    continue
                
                print(COLOR_INFO + f"\nPlaying fragment: {filename}")
                print(COLOR_INFO + f"Timestamp: {timestamp}")
                print(COLOR_TRANSCRIPTIONS + f"Transcription: {transcription}")
                
                # Play the audio (this relies on system audio player)
                if os.name == 'nt':  # Windows
                    import winsound
                    winsound.PlaySound(filepath, winsound.SND_FILENAME)
                else:  # Linux/Mac
                    os.system(f"play {filepath}")
                
                # Ask for speaker identification
                while True:
                    choice = input(COLOR_QUESTION + f"Who is speaking? (1-{len(speaker_names)}, 'n' for new speaker, 's' to skip): ")
                    
                    if choice.lower() == 's':
                        break
                        
                    if choice.lower() == 'n':
                        # Add new speaker - need to close our current connection since add_person will create its own
                        conn.close()
                        name = input(COLOR_QUESTION + "Enter name for new speaker: ")
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
                        clips_assigned_to_speakers.append(filepath)
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
                            print(COLOR_INFO + f"Updating speaker model for {name} with this sample...")
                            audio, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
                            
                            # Extract speaker embedding with Pyannote model
                            with torch.no_grad():
                                # PyAnnote expects tensor with shape [batch_size, channels, samples]
                                waveform_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
                                new_embedding_tensor = self.speaker_embedding(waveform_tensor)
                                
                                # Handle the case where the embedding is already a numpy array
                                if isinstance(new_embedding_tensor, np.ndarray):
                                    new_embedding = new_embedding_tensor
                                else:
                                    new_embedding = new_embedding_tensor.cpu().numpy()
                            
                            # Simple averaging of embeddings (could be improved)
                            updated_embedding = (self.speakers[name] + new_embedding) / 2
                            self.speakers[name] = updated_embedding
                            
                            # Save updated embedding
                            cursor.execute("""
                                UPDATE speakers SET embedding = ? WHERE name = ?
                            """, (updated_embedding.tobytes(), name))
                            
                            conn.commit()
                            print(COLOR_INFO + f"Updated speaker model for {name}.")
                            clips_assigned_to_speakers.append(filepath)

                            break
                        else:
                            print(COLOR_ERROR + "Invalid selection.")
                    except ValueError:
                        print(COLOR_ERROR + "Invalid input. Please enter a number or 's' to skip.")
            
        finally:
            # Make sure to close the connection no matter what
            conn.commit()
            conn.close()
            print(COLOR_SUCCESS + "Review completed.")
            print(COLOR_INFO + "Cleaning up...")
            for clip in clips_assigned_to_speakers:
                os.unlink(clip)


    def update_speaker_profile(self, name, audio_files):
        """Update speaker profile with multiple audio samples for better recognition"""
        if name not in self.speakers:
            print(COLOR_ERROR + f"Speaker {name} not found in database")
            return
            
        print(COLOR_INFO + f"Updating profile for {name} with {len(audio_files)} audio samples...")
        
        all_embeddings = []
        
        # Process each audio file
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                print(COLOR_WARNING + f"Warning: File {audio_file} not found, skipping.")
                continue
                
            # Check if file is m4a or mp3 format and convert if needed
            temp_file = None
            try:
                if audio_file.lower().endswith(('.m4a', '.mp3')):
                    format_type = 'm4a' if audio_file.lower().endswith('.m4a') else 'mp3'
                    print(COLOR_INFO + f"Converting {format_type} file to WAV format...")
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    try:
                        audio = AudioSegment.from_file(audio_file, format=format_type)
                        audio.export(temp_file.name, format="wav")
                        audio_file = temp_file.name
                    except Exception as e:
                        print(COLOR_ERROR + f"Error converting file {audio_file}: {e}")
                        continue
                
                # Load audio file
                audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                
                # Extract speaker embedding
                with torch.no_grad():
                    # PyAnnote expects tensor with shape [batch_size, channels, samples]
                    waveform_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
                    embedding_tensor = self.speaker_embedding(waveform_tensor)
                    
                    # Handle the case where the embedding is already a numpy array
                    if isinstance(embedding_tensor, np.ndarray):
                        embedding_np = embedding_tensor
                    else:
                        embedding_np = embedding_tensor.cpu().numpy()
                
                all_embeddings.append(embedding_np)
                print(COLOR_INFO + "  Processed {audio_file}")
                
            finally:
                # Clean up temporary file if created
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        print(COLOR_ERROR + f"Error removing temp file: {e}")
        
        if not all_embeddings:
            print(COLOR_WARNING + "No valid audio files processed")
            return
            
        # Compute average embedding
        avg_embedding = np.mean(all_embeddings, axis=0)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE speakers SET embedding = ? WHERE name = ?", 
                      (avg_embedding.tobytes(), name))
        
        conn.commit()
        conn.close()
        
        # Update local cache
        self.speakers[name] = avg_embedding
        
        print(COLOR_SUCCESS + f"Successfully updated profile for {name} with {len(all_embeddings)} samples")
        
    def reset_profile(self, name):
        """Reset a speaker profile by removing it from the database"""
        if name not in self.speakers:
            print(COLOR_ERROR + f"Speaker {name} not found in database")
            return
            
        # Delete from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get speaker ID
        cursor.execute("SELECT id FROM speakers WHERE name = ?", (name,))
        result = cursor.fetchone()
        if not result:
            print(COLOR_ERROR + f"Speaker {name} not found in database")
            conn.close()
            return
            
        speaker_id = result[0]
        
        # Delete from speakers table
        cursor.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
        
        # Delete related word logs
        cursor.execute("DELETE FROM word_logs WHERE speaker_id = ?", (speaker_id,))
        
        # Update unknown fragments
        cursor.execute("UPDATE unknown_fragments SET is_labeled = 0, assigned_speaker_id = NULL WHERE assigned_speaker_id = ?", 
                      (speaker_id,))
        
        conn.commit()
        conn.close()
        
        # Remove from local cache
        if name in self.speakers:
            del self.speakers[name]
        
        print(COLOR_SUCCESS + f"Speaker profile for {name} has been reset")
        print(COLOR_INFO + f"Use add-person to create a new profile")


    def load_config(self):
        """Load configuration from file or create default config"""
        default_config = {
            "microphone": {
                "device": None,  # Default to system default
                "name": "Default"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(COLOR_INFO + f"Loaded configuration from {self.config_path}")
                # Ensure all required keys exist by combining with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
            except Exception as e:
                print(COLOR_ERROR + f"Error loading config: {e}")
                print(COLOR_ERROR + "Using default configuration")
                return default_config
        else:
            print(COLOR_WARNING + "No configuration file found. Using default settings.")
            return default_config

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(COLOR_INFO + f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(COLOR_ERROR + f"Error saving config: {e}")

    def list_microphones(self):
        """List all available microphones"""
        devices = sd.query_devices()
        print(COLOR_INFO + "\nAvailable microphones:")
        print(COLOR_INFO + "----------------------")
        
        # Track input devices for selection
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # type: ignore # It's an input device
                input_devices.append((i, device))
                is_default = " (default)" if device.get('default_input') else "" # type: ignore
                # Truncate long names
                name = device['name'] # type: ignore
                if len(name) > 40:
                    name = name[:37] + "..."
                print(COLOR_INFO + f"{len(input_devices)}: {name}{is_default}")
        
        # Check if current device is still valid
        current_device = self.config["microphone"]["device"]
        current_name = self.config["microphone"]["name"]
        
        if current_device is not None:
            # Verify current device still exists
            device_exists = False
            for device_id, _ in input_devices:
                if device_id == current_device:
                    device_exists = True
                    break
                    
            if device_exists:
                print(COLOR_INFO + f"\nCurrent microphone: {current_name} (device {current_device})")
            else:
                print(COLOR_WARNING + "\nPreviously selected microphone no longer available.")
                self.config["microphone"]["device"] = None
                self.config["microphone"]["name"] = "Default"
                print(COLOR_WARNING + "Reverted to system default.")
        else:
            print(COLOR_WARNING + "\nCurrent microphone: System Default")
        
        return input_devices

    def select_microphone(self):
        """Allow user to select a microphone from the available options"""
        input_devices = self.list_microphones()
        
        if not input_devices:
            print(COLOR_ERROR + "No input devices found. Program cannot continue without one!")
            return
            
        print(COLOR_INFO + "\nOptions:")
        print(COLOR_INFO + "  0: Use system default")
        print(COLOR_INFO + "  1-{}: Select specific microphone".format(len(input_devices)))
        print(COLOR_INFO + "  c: Cancel and keep current setting")
        
        while True:
            choice = input("\nSelect microphone: ").strip().lower()
            
            if choice == 'c':
                print(COLOR_WARNING + "Selection cancelled.")
                return
                
            if choice == '0':
                self.config["microphone"]["device"] = None
                self.config["microphone"]["name"] = "Default"
                self.save_config()
                print(COLOR_INFO + "Microphone set to system default.")
                return
                
            try:
                idx = int(choice)
                if 1 <= idx <= len(input_devices):
                    device_id, device = input_devices[idx-1]
                    self.config["microphone"]["device"] = device_id
                    self.config["microphone"]["name"] = device['name']
                    self.save_config()
                    print(COLOR_INFO + f"Microphone set to: {device['name']} (device {device_id})")
                    return
                else:
                    print(COLOR_WARNING + f"Please enter a number between 0 and {len(input_devices)}")
            except ValueError:
                print(COLOR_ERROR + "Invalid input. Please enter a number or 'c' to cancel.")

    def detect_repetition(self, text):
        """Detect and fix repetitive text patterns"""
        if not text:
            return text
            
        # Look for simple repetition patterns
        words = text.split()
        if len(words) < 5:  # Don't process short texts
            return text
            
        # Check for repeating patterns of 3+ words
        cleaned_text = []
        i = 0
        while i < len(words):
            cleaned_text.append(words[i])
            
            # Look for repetition patterns starting at current word
            for pattern_len in range(3, min(10, len(words) - i)):
                pattern = words[i:i+pattern_len]
                # Check if this pattern repeats immediately after
                if i + pattern_len*2 <= len(words) and words[i+pattern_len:i+pattern_len*2] == pattern:
                    # Skip ahead past repetitions
                    repeat_count = 1
                    pos = i + pattern_len
                    while pos + pattern_len <= len(words) and words[pos:pos+pattern_len] == pattern:
                        repeat_count += 1
                        pos += pattern_len
                    
                    print(COLOR_WARNING + f"Detected {repeat_count} repetitions of pattern: '{' '.join(pattern)}'")
                    i = pos - 1  # -1 because of the i++ at the end of the loop
                    break
            i += 1
        
        return ' '.join(cleaned_text)

def main():
    colorama_init(autoreset=True)
    parser = argparse.ArgumentParser(description="Voice Logger - Real-time speech recognition with speaker identification")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add person command
    add_parser = subparsers.add_parser("add-person", help="Add a new person to the system")
    add_parser.add_argument("audio_file", help="Audio file (MP3, WAV, etc.) containing the person's voice")
    add_parser.add_argument("name", help="Name to associate with the voice")
    
    # Log words command
    subparsers.add_parser("log-words", help="Start logging spoken words")
    
    # Review fragments command
    subparsers.add_parser("review", help="Review and label unknown speech fragments")
    
    # Update profile command
    update_parser = subparsers.add_parser("update-profile", help="Update a speaker profile with multiple audio samples")
    update_parser.add_argument("name", help="Name of the speaker to update")
    update_parser.add_argument("audio_files", nargs="+", help="Audio files to use for the update")
    
    # Reset profile command
    reset_parser = subparsers.add_parser("reset-profile", help="Reset a speaker profile")
    reset_parser.add_argument("name", help="Name of the speaker to reset")
    
    # Microphone selection command
    subparsers.add_parser("select-mic", help="Select which microphone to use for voice logging")
    
    # List microphones command
    subparsers.add_parser("list-mics", help="List all available microphones")
    
    args = parser.parse_args()
    
    voice_logger = VoiceLogger()
    
    if args.command == "add-person":
        voice_logger.add_person(args.audio_file, args.name)
    elif args.command == "log-words":
        voice_logger.log_words()
    elif args.command == "review":
        voice_logger.review_unknown_fragments()
    elif args.command == "update-profile":
        voice_logger.update_speaker_profile(args.name, args.audio_files)
    elif args.command == "reset-profile":
        voice_logger.reset_profile(args.name)
    elif args.command == "select-mic":
        voice_logger.select_microphone()
    elif args.command == "list-mics":
        voice_logger.list_microphones()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()