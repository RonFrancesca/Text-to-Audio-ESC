---
layout: default
title:  "Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification"
---

Accompanying website to the paper _Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification, Francesca Ronchini, Luca Comanducci, Fabio Antonacci, submitted at EUSIPCO 2024_

## Abstract
In the past few years, text-to-audio models have emerged as a significant advancement in automatic audio generation. Although they represent impressive technological progress, the effectiveness of their use in the development of audio applications remains uncertain. This paper aims to investigate these aspects, specifically focusing on the task of classification of environmental sounds. This study analyzes the performance of two different environmental classification systems when data generated from text-to-audio models is used for training. Two cases are considered: a) when the training dataset is augmented by data coming from two different text-to-audio models; and b) when the training dataset consists solely of synthetic audio generated. In both cases, the performance of the classification task is tested on real data. Results indicate that text-to-audio models are effective for dataset augmentation, whereas the performance of the models drops when relying on only generated audio.  

## Audio Examples

Here we present audio data generated using AudioLDM2, MusicGen via simple prompt and via ChatGPT prompts, namely AudioLDM2<sub>gpt</sub> and MusicGen<sub>gpt</sub>. We present results for each of the 10 classes contained in the [UrbanSound8K (US8K)](https://urbansounddataset.weebly.com/urbansound8k.html) dataset: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_drilling, gun_shot, jackhammer, siren, street_music. For each class, we present three examples per each model.

### 1) air_conditioner

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

### 4) dog_bark

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
### 6) engine_drilling
### 7) gun_shot
### 8) jackhammer
### 9) siren
### 10) street_music
