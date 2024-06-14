---
layout: default
title:  "Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification"
---

Accompanying website to the paper _Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification, Francesca Ronchini, Luca Comanducci, Fabio Antonacci, submitted at DCASE 2024_

## Abstract
In the past few years, text-to-audio models have emerged as a significant advancement in automatic audio generation. Although they represent impressive technological progress, the effectiveness of their use in the development of audio applications remains uncertain. This paper aims to investigate these aspects, specifically focusing on the task of classification of environmental sounds. This study analyzes the performance of two different environmental classification systems when data generated from text-to-audio models is used for training. Two cases are considered: a) when the training dataset is augmented by data coming from two different text-to-audio models; and b) when the training dataset consists solely of synthetic audio generated. In both cases, the performance of the classification task is tested on real data. Results indicate that text-to-audio models are effective for dataset augmentation, whereas the performance of the models drops when relying on only generated audio.  

## Audio Examples

In this page, we present audio data generated using AudioLDM2 and MusicGen via simple prompt and via ChatGPT prompts  (namely AudioLDM2<sub>gpt</sub> and MusicGen<sub>gpt</sub>). We present results for each of the 10 classes contained in the [UrbanSound8K (US8K)](https://urbansounddataset.weebly.com/urbansound8k.html) dataset: <i>air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music</i>. For each class, we present three examples per each model.

### 1) air_conditioner

- <b>Simple prompt</b>: "A clear sound of an air conditiner in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of an air conditioner in a urban environment."

USK8 example: 
<audio src="audio/US8K/air_cond/100852-0-0-13.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/air_cond/109.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/air_cond/175.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2</h6>
     <audio src="audio/AudioLDM2/air_cond/143.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/air_cond/125.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/air_cond/217.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/air_cond/238.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/air_cond/243.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/air_cond/291.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/air_cond/53.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/air_cond/353.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/air_cond/343.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/air_cond/404.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>


### 2) car_horn

- <b>Simple prompt</b>: "A clear sound of an car horning in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of an car horning in a urban environment."

USK8 example: 
<audio src="audio/US8K/car_horn/100648-1-0-0.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/car_horn/1152.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/car_horn/138.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/car_horn/13.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/car_horn/176.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/car_horn/501.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/car_horn/195.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/car_horn/47.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/car_horn/292.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/car_horn/856.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/car_horn/62.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/car_horn/89.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/car_horn/444.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

### 3) children_playing

- <b>Simple prompt</b>: "A clear sound of a children playing between them in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of children playing between them in a urban environment."

USK8 example: 
<audio src="audio/US8K/kid_playing/100263-2-0-117.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/kid_playing/277.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/kid_playing/177.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/kid_playing/344.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/kid_playing/292.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/kid_playing/491.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/kid_playing/202.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/kid_playing/492.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/kid_playing/509.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/kid_playing/677.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/kid_playing/241.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/kid_playing/714.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/kid_playing/703.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>


### 4) dog_bark

- <b>Simple prompt</b>: "A clear sound of a dog barking in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of a dog barking in a urban environment."

USK8 example: 
<audio src="audio/US8K/dog_bark/100032-3-0-0.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/dog_bark/137.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/dog_bark/117.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/dog_bark/204.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/dog_bark/313.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/dog_bark/277.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/dog_bark/180.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/dog_bark/378.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/dog_bark/515.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/dog_bark/382.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/dog_bark/30.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/dog_bark/590.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/dog_bark/670.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

### 5) drilling

- <b>Simple prompt</b>: "A clear sound of a drilling in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of a drilling in a urban environment."

USK8 example: 
<audio src="audio/US8K/drilling/104625-4-0-3.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/drilling/322.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/drilling/298.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/drilling/410.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/drilling/342.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/drilling/411.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/drilling/441.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/drilling/565.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/drilling/532.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/drilling/495.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/drilling/621.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/drilling/796.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/drilling/815.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>


### 6) engine_idling

- <b>Simple prompt</b>: "A clear sound of an engine idling in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of an engine idling in a urban environment."

USK8 example: 
<audio src="audio/US8K/engine_idling/102857-5-0-24.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/engine_idling/109.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/engine_idling/283.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/engine_idling/126.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/engine_idling/256.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/engine_idling/317.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/engine_idling/481.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/engine_idling/177.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/engine_idling/489.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/engine_idling/508.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/engine_idling/630.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/engine_idling/48.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/engine_idling/765.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

### 7) gun_shot

- <b>Simple prompt</b>: "A clear sound of a gun shot in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of a gun shot in a urban environment."

USK8 example: 
<audio src="audio/US8K/gun_shot/145611-6-0-0.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/gun_shot/210.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/gun_shot/107.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/gun_shot/171.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/gun_shot/356.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/gun_shot/24.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/gun_shot/486.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/gun_shot/348.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/gun_shot/592.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/gun_shot/360.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/gun_shot/718.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/gun_shot/519.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/gun_shot/779.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

### 8) jackhammer

- <b>Simple prompt</b>: "A clear sound of a jackhammer in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of a jackhammer in a urban environment."

USK8 example: 
<audio src="audio/US8K/jackhammer/125678-7-0-3.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/jackhammer/256.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/jackhammer/203.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/jackhammer/100.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/jackhammer/271.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/jackhammer/454.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/jackhammer/373.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/jackhammer/162.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/jackhammer/483.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/jackhammer/694.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/jackhammer/515.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/jackhammer/230.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/jackhammer/729" controls preload style="width: 200px;"></audio>
   </div>
</div>


### 9) siren

- <b>Simple prompt</b>: "A clear sound of a siren coming from an emergency vehicle in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of the sound of a siren coming from an emergency vehicle in a urban environment." 

USK8 example: 
<audio src="audio/US8K/siren/105289-8-0-1.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/siren/148.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/siren/306.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/siren/175.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/siren/1028.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/siren/454.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/siren/450.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/siren/311.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/siren/379.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/siren/694.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/siren/545.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/siren/443.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/siren/667.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

### 10) street_music

- <b>Simple prompt</b>: "A clear sound of street music in a urban context." <br> 
- <b>ChatGPT prompt</b>: "Generate a realistic audio representation of street music in a urban environment." 

USK8 example: 
<audio src="audio/US8K/street_music/14527-9-0-6.wav" controls preload style="width: 150px;"></audio>

<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/AudioGen/street_music/187.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/AudioGen_gpt/street_music/194.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2/street_music/308.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioLDM2<sub>gpt</sub></h6>
     <audio src="audio/AudioLDM2_gpt/street_music/506.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/street_music/287.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/street_music/305.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/street_music/468.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/street_music/767.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/AudioGen/street_music/409.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/AudioGen_gpt/street_music/378.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/AudioLDM2/street_music/675.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/AudioLDM2_gpt/street_music/982.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>
