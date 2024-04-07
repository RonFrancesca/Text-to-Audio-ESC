import os
import scipy
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from diffusers import AudioLDMPipeline, AudioLDM2Pipeline

audio_folder = <Insert path to folder output>
os.makedirs(audio_folder, exist_ok=True)

# equivalent of audioldm2-full
model = "audioldm2"
repo_id = f"cvssp/{model}"

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


# following tips from: https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#tips

sample_rate = 16000
audio_length_in_s = 4.0
num_inference_steps = 200

# prompt used for class sound generation
prompt = <Insert prompt>
negative_prompt = <Insert negative prompt>
    
# load the model and move it to GPU
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# set the seed for the generator
generator = torch.Generator(device).manual_seed(20)
    
class_folder_path = os.path.join(audio_folder, classes_fold[3])
os.makedirs(class_folder_path, exist_ok=True)

#for sound in enumerate(range(classes[class_sound])):
for sound in range(1000):
        
    audio = pipe(prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, 
            num_waveforms_per_prompt=1, 
            audio_length_in_s=4.0
        ).audios[0]

    audio_file_out = f"{sound}.wav"
    audio_file_out_path = os.path.join(class_folder_path, audio_file_out)
        
    print(f"File {sound} saved in {audio_file_out_path}")
    scipy.io.wavfile.write(audio_file_out_path, rate=sample_rate, data=audio)

    
    
    
    
    
    
        