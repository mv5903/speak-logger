# Voice Logger Configuration Reference

This document explains all configuration parameters in the `config.json` file.

## Microphone Settings (don't modify these directly)

| Parameter | Description |
|-----------|-------------|
| `device`  | Device ID for the microphone to use (null for system default) |
| `name`    | Human-readable name of the selected microphone |

## Models

### Whisper Speech Recognition Model

| Parameter    | Description |
|--------------|-------------|
| `model_name` | Whisper model size (tiny/base/small/medium/large) |
| `language`   | Language code for speech recognition (e.g., "en" for English) |
| `task`       | Task type ("transcribe" or "translate") |

### Speaker Embedding Model

| Parameter    | Description |
|--------------|-------------|
| `model_path` | Path to speaker embedding model in HuggingFace format |

## Audio Settings

| Parameter         | Description |
|-------------------|-------------|
| `sample_rate`     | Audio sample rate in Hz (16000 recommended) |
| `buffer_duration` | Duration of audio buffer in seconds |
| `min_audio_length`| Minimum audio length in seconds for speaker profiles |
| `overlap_seconds` | Overlap between audio chunks in seconds (prevents missed words) |

## Threshold Settings

| Parameter           | Description |
|---------------------|-------------|
| `speaker_confidence`| Minimum confidence score for speaker identification |
| `speech_confidence` | Minimum confidence score for speech recognition |
| `silence_rms`       | RMS threshold below which audio is considered silence |

## Processing Settings

| Parameter                     | Description |
|-------------------------------|-------------|
| `max_silence_count`           | Number of silent chunks before resetting overlap buffer |
| `min_text_length_for_repetition`| Minimum word count to check for repetitions |
| `min_pattern_length`          | Minimum length of repeating word patterns to detect |
| `max_pattern_length`          | Maximum length of repeating word patterns to detect |

## Path Settings

| Parameter    | Description |
|--------------|-------------|
| `db_path`    | Path to SQLite database file |
| `unknown_dir`| Directory for storing unknown speaker audio fragments |
