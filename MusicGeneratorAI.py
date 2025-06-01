import music21 as m21
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import random

# Function to load MIDI files
def load_songs_in_midi(directory):
	songs = []
	for path, _, files in os.walk(directory):
		for file in files:
			if file.endswith(".mid"):
				song = m21.converter.parse(os.path.join(path, file))
				songs.append(song)
	return songs

# Function to preprocess MIDI files
def preprocess(song, sequence_length):
	notes = []
	durations = []
	for element in song.flat.notesAndRests:
		if isinstance(element, m21.note.Note) or isinstance(element, m21.chord.Chord):
			notes.append(str(element.pitch)) if isinstance(element, m21.note.Note) else notes.append('.'.join(str(n) for n in element.normalOrder))
			durations.append(element.quarterLength)  # Capture the duration of each note/chord
		elif isinstance(element, m21.note.Rest):
			notes.append('rest')
			durations.append(element.quarterLength)
	return notes, durations

# Load and preprocess MIDI files
directory = "data"
songs = load_songs_in_midi(directory)
sequences = [preprocess(song, sequence_length=64) for song in songs]

# Create sequences for training
def create_sequences(notes, sequence_length):
	sequences = []
	for i in range(len(notes) - sequence_length):
		seq = notes[i:i + sequence_length]
		sequences.append(seq)
	return sequences

# Process notes and durations
all_sequences_notes = []
all_sequences_durations = []
for song, durations in sequences:
	all_sequences_notes.extend(create_sequences(song, sequence_length=64))
	all_sequences_durations.extend(create_sequences(durations, sequence_length=64))

unique_notes = list(set([note for seq in all_sequences_notes for note in seq]))
note_to_int = {note: number for number, note in enumerate(unique_notes)}
int_sequences_notes = [[note_to_int[note] for note in seq] for seq in all_sequences_notes]

unique_durations = list(set([dur for seq in all_sequences_durations for dur in seq]))
duration_to_int = {dur: number for number, dur in enumerate(unique_durations)}
int_sequences_durations = [[duration_to_int[dur] for dur in seq] for seq in all_sequences_durations]

X_notes = np.array([seq[:-1] for seq in int_sequences_notes])
y_notes = np.array([seq[-1] for seq in int_sequences_notes])
X_durations = np.array([seq[:-1] for seq in int_sequences_durations])
y_durations = np.array([seq[-1] for seq in int_sequences_durations])

# Check if models exist and load them, otherwise train and save
if os.path.exists("model_notes.h5"):
	model_notes = load_model("model_notes.h5")
else:
	# Build the neural network for notes
	model_notes = Sequential()
	model_notes.add(LSTM(512, input_shape=(X_notes.shape[1], 1), return_sequences=True))  # Increased complexity
	model_notes.add(Dropout(0.3))
	model_notes.add(LSTM(512))
	model_notes.add(Dense(256, activation='relu'))
	model_notes.add(Dense(len(unique_notes), activation='softmax'))

	model_notes.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	y_notes_categorical = to_categorical(y_notes, num_classes=len(unique_notes))
	X_notes = np.reshape(X_notes, (X_notes.shape[0], X_notes.shape[1], 1))

	# Train the model for notes
	model_notes.fit(X_notes, y_notes_categorical, epochs=1, batch_size=64)
	model_notes.save("model_notes.h5")

if os.path.exists("model_durations.h5"):
	model_durations = load_model("model_durations.h5")

else:
	# Build the neural network for durations
	model_durations = Sequential()
	model_durations.add(LSTM(512, input_shape=(X_durations.shape[1], 1), return_sequences=True))  # Increased complexity
	model_durations.add(Dropout(0.3))
	model_durations.add(LSTM(512))
	model_durations.add(Dense(256, activation='relu'))
	model_durations.add(Dense(len(unique_durations), activation='softmax'))

	model_durations.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	y_durations_categorical = to_categorical(y_durations, num_classes=len(unique_durations))
	X_durations = np.reshape(X_durations, (X_durations.shape[0], X_durations.shape[1], 1))

	# Train the model for durations
	model_durations.fit(X_durations, y_durations_categorical, epochs=1, batch_size=64)
	model_durations.save("model_durations.h5")


def generate_chords(notes, int_to_note, scale):
	"""
	Generate chord progressions based on the notes and the provided scale.
	"""
	chords = []
	for note in notes:
		if note == 'rest':
			chords.append('rest')
		else:
			root = int_to_note[note].split('.')[0]
			if root in scale:
				chord_notes = [root]
				third = scale[(scale.index(root) + 2) % len(scale)]
				fifth = scale[(scale.index(root) + 4) % len(scale)]
				chord_notes.extend([third, fifth])
				chords.append('.'.join(chord_notes))
			else:
				chords.append(root)
	return chords

def generate_music(
	model_notes, model_durations, start_sequence_notes, start_sequence_durations, int_to_note, int_to_duration, 
	duration_seconds, diversity=0.9, seed=None, scale=None, start_from='random', tempo=160
):
	"""
	Generate music with rhythmic, harmonic coherence, dynamics, and natural phrasing.
	"""
	if seed is not None:
		np.random.seed(seed)

	# Define a default scale (C major) if none is provided
	if scale is None:
		scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

	def is_in_scale(note):
		"""Check if a note is in the provided scale."""
		if note == 'rest':  # Rests are always allowed
			return True 
		# Allow only single notes in the scale (e.g., "C4", not "C4.E4.G4")
		return note.split('.')[0] in scale

	# Calculate the number of beats based on the duration in seconds and tempo
	beats_per_second = tempo / 60  # Increased default tempo for more energetic feel
	total_beats = duration_seconds * beats_per_second
	
	# Enhanced musical parameters for more interesting output
	phrase_lengths = [4, 6, 8]  # Variable phrase lengths
	dynamic_range = (60, 110)  # Wider dynamic range
	rhythmic_patterns = [0.25, 0.5, 1, 1.5, 2]  # More diverse rhythm options

	generated_notes = start_sequence_notes[:]
	generated_durations = start_sequence_durations[:]
	current_duration = sum(generated_durations)

	# Enhanced musical phrasing parameters
	phrase_length = random.choice(phrase_lengths)
	current_phrase_position = 0
	phrase_direction = 1
	last_note_direction = 0
	melodic_tension = 0  # Track musical tension
	rhythm_complexity = random.random()  # Varies rhythmic complexity

	while current_duration < total_beats:
		# Dynamic musical development
		current_phrase_position = (current_phrase_position + 1) % phrase_length
		if current_phrase_position == 0:
			phrase_direction *= -1
			phrase_length = random.choice(phrase_lengths)
			melodic_tension = min(1.0, melodic_tension + random.uniform(0.1, 0.3))
			rhythm_complexity = random.uniform(0.3, 0.9)

		# Predict the next note with enhanced musical context
		X_note_pred = np.reshape(generated_notes[-64:], (1, 64, 1))
		note_prediction = model_notes.predict(X_note_pred, verbose=0)[0]
		note_prediction = np.log(note_prediction + 1e-9) / diversity
		note_prediction = np.exp(note_prediction)
		note_prediction = note_prediction / np.sum(note_prediction)

		# Enhanced note selection with musical context and tension
		valid_indices = [idx for idx, note in int_to_note.items() if is_in_scale(note)]
		if not valid_indices:
			valid_indices = list(range(len(note_prediction)))

		# Dynamic note selection based on melodic tension and phrase position
		if len(generated_notes) > 0:
			prev_note = generated_notes[-1]
			# Increase probability of larger intervals during high tension
			interval_bias = melodic_tension * 0.3
			# Add variation based on phrase position
			phrase_bias = abs(current_phrase_position - phrase_length/2) / (phrase_length/2)
			
			for idx in valid_indices:
				# Calculate interval size
				interval = abs(idx - prev_note)
				# Favor interesting intervals during high tension
				if interval > 2 and interval < 7:
					note_prediction[idx] *= (1 + interval_bias)
				# Add phrase-based variations
				if phrase_bias > 0.7:
					note_prediction[idx] *= (1 + phrase_bias * 0.2)

		filtered_probs = np.array([note_prediction[idx] if idx in valid_indices else 0 for idx in range(len(note_prediction))])
		if np.sum(filtered_probs) == 0:
			filtered_probs = np.ones(len(filtered_probs))
		filtered_probs = filtered_probs / np.sum(filtered_probs)

		note_index = np.random.choice(range(len(filtered_probs)), p=filtered_probs)

		# Update melodic direction
		if len(generated_notes) > 0:
			last_note_direction = np.sign(note_index - generated_notes[-1])

		# Advanced rhythm generation with pattern variations
		X_dur_pred = np.reshape(generated_durations[-64:], (1, 64, 1))
		dur_prediction = model_durations.predict(X_dur_pred, verbose=0)[0]
		dur_prediction = np.log(dur_prediction + 1e-9) / (diversity * (1 + rhythm_complexity))
		dur_prediction = np.exp(dur_prediction)
		dur_prediction = dur_prediction / np.sum(dur_prediction)
		
		# Favor more interesting rhythmic patterns based on complexity
		for i, dur in enumerate(rhythmic_patterns):
			if dur in int_to_duration.values():
				dur_idx = list(int_to_duration.values()).index(dur)
				dur_prediction[dur_idx] *= (1 + rhythm_complexity)

		# Adjust duration probabilities based on phrase position
		if current_phrase_position == phrase_length - 1:
			# Encourage longer notes at phrase endings
			dur_prediction *= np.array([d > np.median(list(int_to_duration.values())) for d in int_to_duration.values()])
			dur_prediction = dur_prediction / np.sum(dur_prediction)

		dur_index = np.random.choice(range(len(dur_prediction)), p=dur_prediction)

		# Handle note coherence and transitions with enhanced expression
		if len(generated_notes) < 64:
			note_index = generated_notes[len(generated_notes)]
			dur_index = generated_durations[len(generated_durations)]
		else:
			# Implement more sophisticated note transition logic
			if len(generated_notes) > 0:
				previous_note_index = generated_notes[-1]
				if abs(note_index - previous_note_index) > 5:
					# Create smoother transitions for large intervals
					transition_steps = abs(note_index - previous_note_index) // 2
					step_size = (note_index - previous_note_index) / (transition_steps + 1)
					for step in range(transition_steps):
						transition_note = int(previous_note_index + step_size * (step + 1))
						generated_notes.append(transition_note)
						# Add subtle timing variations
						actual_duration = int_to_duration[dur_index] * random.uniform(0.95, 1.05)
						generated_durations.append(actual_duration)
						current_duration += actual_duration

		# Enhanced chord and rest handling
		note = int_to_note[note_index]
		if '.' in note:
			# More sophisticated chord probability based on phrase position
			chord_prob = 0.3 if current_phrase_position in [0, phrase_length-1] else 0.1
			if np.random.random() > chord_prob:
				continue
			else:
				note_parts = note.split('.')
				if len(note_parts) > 3:  # Limit chord complexity
					note_index = valid_indices[0]

		# Dynamic rest probability based on phrase position
		rest_prob = 0.2 if current_phrase_position == phrase_length-1 else 0.05
		if note == 'rest' and np.random.random() > rest_prob:
			continue

		generated_notes.append(note_index)
		generated_durations.append(dur_index)
		current_duration += int_to_duration[dur_index]

	# Generate more musically coherent chord progressions
	generated_chords = generate_chords(generated_notes, int_to_note, scale)

	return [int_to_note[i] for i in generated_notes], [int_to_duration[i] for i in generated_durations], generated_chords




# Function to handle user input for MIDI file
def handle_user_midi(file_path):
	"""
	Handles user-provided MIDI files, preprocesses them, and creates input sequences for model generation.
	Falls back to a random sequence if the file is invalid or incomplete.
	"""
	try:
		# Load and parse the user MIDI file
		print(f"Attempting to load MIDI file: {file_path}")
		user_song = m21.converter.parse(file_path)
		print("Successfully loaded MIDI file.")

		# Preprocess the MIDI file into notes and durations
		user_notes, user_durations = preprocess(user_song, sequence_length=64)

		# Ensure there are enough notes and durations for a 64-length sequence
		if len(user_notes) < 64 or len(user_durations) < 64:
			raise ValueError("The provided MIDI file has insufficient notes or durations for processing.")

		# Convert to integer sequences using mapping
		start_sequence_notes = [note_to_int[note] for note in user_notes[:64]]
		start_sequence_durations = [duration_to_int[dur] for dur in user_durations[:64]]

		print("Preprocessed MIDI file successfully.")
		return start_sequence_notes, start_sequence_durations

	except FileNotFoundError:
		print("Error: The specified MIDI file was not found. Falling back to a random sequence.")
	except ValueError as ve:
		print(f"Error: {ve}. Falling back to a random sequence.")
	except Exception as e:
		print(f"Unexpected error while processing MIDI file: {e}. Falling back to a random sequence.")

	# Fallback to a random sequence from the dataset
	print("Selecting a random sequence as fallback.")
	random_song_idx = random.randint(0, len(sequences) - 1)
	random_start_idx = random.randint(0, len(sequences[random_song_idx][0]) - 65)

	start_sequence_notes = [note_to_int[note] for note in sequences[random_song_idx][0][random_start_idx:random_start_idx + 64]]
	start_sequence_durations = [duration_to_int[dur] for dur in sequences[random_song_idx][1][random_start_idx:random_start_idx + 64]]

	print("Fallback sequence selected successfully.")
	return start_sequence_notes, start_sequence_durations


int_to_note = {number: note for note, number in note_to_int.items()}
int_to_duration = {number: dur for dur, number in duration_to_int.items()}

# User can input their own MIDI file
user_midi_file = "C:/Users/audri/Documents/Personal/Music_AI/data/MicnVenus.mid"  # Replace with the actual path to the user's MIDI file
start_sequence_notes, start_sequence_durations = handle_user_midi(user_midi_file)

SCALES = {
	'C Major': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
	'G Major': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
	'D Major': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
	'A Minor': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
	'E Minor': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
	'B Minor': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
	'F Major': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
}

def choose_random_scale():
	scale_name, scale_notes = random.choice(list(SCALES.items()))
	print(f"Selected Scale: {scale_name} ({', '.join(scale_notes)})")
	return scale_notes

# Generate music
generated_notes, generated_durations, generated_chords = generate_music(
	model_notes, model_durations, start_sequence_notes, start_sequence_durations, 
	duration_seconds=500, int_to_note=int_to_note, int_to_duration=int_to_duration, 
	seed=random.randint(0, 1000000000), scale=choose_random_scale(), start_from='random'
)

# Save the generated music
from music21 import stream, note, chord

def save_midi(notes, durations, chords, file_name="generated_music_with_chords.mid"):
	midi_stream = stream.Stream()
	phrase_length = random.randint(4, 8)  # Random phrase length for natural phrasing
	current_phrase = 0
	base_velocity = 80
	
	for i, (pitch, duration, chord_notes) in enumerate(zip(notes, durations, chords)):
		# Dynamic velocity control based on musical phrasing
		current_phrase = (current_phrase + 1) % phrase_length
		phrase_position = current_phrase / phrase_length
		
		# Create natural dynamics with crescendos and diminuendos
		velocity = int(base_velocity + 20 * np.sin(2 * np.pi * phrase_position))
		velocity = max(40, min(110, velocity))  # Keep velocity in reasonable range
		
		# Add slight random variations for humanization
		velocity += random.randint(-5, 5)
		duration_variation = duration * (1 + random.uniform(-0.05, 0.05))  # Subtle timing variations
		
		# Create and configure the note/chord
		if pitch == 'rest':
			midi_note = note.Rest(quarterLength=duration_variation)
		elif '.' in pitch or pitch.isdigit():
			chord_pitches = [int(p) for p in pitch.split('.')]
			midi_note = chord.Chord(chord_pitches, quarterLength=duration_variation)
			midi_note.volume.velocity = velocity
		else:
			midi_note = note.Note(pitch, quarterLength=duration_variation)
			midi_note.volume.velocity = velocity
			
			# Add articulation for more expressive playing
			if random.random() < 0.2:  # 20% chance for articulation
				if random.random() < 0.5:
					midi_note.articulations.append(m21.articulations.Accent())
				else:
					midi_note.articulations.append(m21.articulations.Staccato())
		
		midi_stream.append(midi_note)
		
		# Handle accompanying chords with slightly lower velocity
		if chord_notes != 'rest':
			try:
				chord_pitches = [note.Note(n) for n in chord_notes.split('.')]
				midi_chord = chord.Chord(chord_pitches, quarterLength=duration_variation)
				midi_chord.volume.velocity = int(velocity * 0.8)  # Slightly softer accompaniment
				midi_stream.append(midi_chord)
			except Exception as e:
				print(f"Error creating chord: {e}")
	
	midi_stream.write("midi", fp=file_name)

save_midi(generated_notes, generated_durations, generated_chords)

print("Generated music with chords saved as generated_music_with_chords.mid")


