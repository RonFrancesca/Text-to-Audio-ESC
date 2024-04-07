import os
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from IPython.display import Audio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write


audio_folder = <insert path to output folder>
os.makedirs(audio_folder, exist_ok=True)


classes = {
    "a dog barking": 1000, 
    "kids playing on the street": 1000, 
    "an air conditioner": 1000, 
    "music on the street": 1000,
    "a jackhammer": 1000, 
    "an engine idling": 1000,
    "a drilling": 1000, 
    "a siren": 929,
    "a car horning": 429,
    "a gun shot": 374
}

id = <Insert class id>
n_file = <Insert number of files to generate>

classes_fold = [
    "dog_bark", 
    "children_playing", 
    "air_conditioner", 
    "street_music",
    "jackhammer",
    "engine_idling",
    "drilling", 
    "siren",
    "car_horn", 
    "gun_shot",
]

print(f"Total number of files to generate: {sum(classes.values())}")


duration = 4
model_name = 'facebook/audiogen-medium'


# prompt used for class sound generation
prompt = <Insert prompt>
print(f"Prompt used: {prompt}")

# load the model
model = AudioGen.get_pretrained(model_name)
model.set_generation_params(duration=duration)  # generate 4 seconds.

class_folder_path = os.path.join(audio_folder, classes_fold[id])
os.makedirs(class_folder_path, exist_ok=True)


for sound in range(n_file):
        
    audio = model.generate(prompt)  # generates 1 samples.

    audio_file_out = f"{sound}"
    audio_file_out_path = os.path.join(class_folder_path, audio_file_out)
        
    print(f"File {sound} saved in {audio_file_out_path}")
    audio_write(audio_file_out_path, audio[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    
    
    
    
    
    
        