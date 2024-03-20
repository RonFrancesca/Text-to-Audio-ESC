---
layout: default
title:  "Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification"
---

Accompanying website to the paper _Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification, Francesca Ronchini, Luca Comanducci, Fabio Antonacci, submitted at EUSIPCO 2024_

## Abstract
In the past few years, text-to-audio models have emerged as a significant advancement in automatic audio generation. Although they represent impressive technological progress, the effectiveness of their use in the development of audio applications remains uncertain. This paper aims to investigate these aspects, specifically focusing on the task of classification of environmental sounds. This study analyzes the performance of two different environmental classification systems when data generated from text-to-audio models is used for training. Two cases are considered: a) when the training dataset is augmented by data coming from two different text-to-audio models; and b) when the training dataset consists solely of synthetic audio generated. In both cases, the performance of the classification task is tested on real data. Results indicate that text-to-audio models are effective for dataset augmentation, whereas the performance of the models drops when relying on only generated audio.  

## Audio Examples

Here we present audio data generated using AudioLDM2, MusicGen via simple prompt and via ChatGPT prompts, namely AudioLDM2<sub>gpt</sub> and MusicGen<sub>gpt</sub>. We present results for each of the 10 classes contained in the [UrbanSound8K (US8K)](https://urbansounddataset.weebly.com/urbansound8k.html) dataset: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_drilling, gun_shot, jackhammer, siren, street_music. For each class, we present three examples per each model.

### air_conditioner
<div class="container">
   <div class="column-1">
     <h6>AudioGen</h6>
     <audio src="audio/example0/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <h6>AudioLDM2</h6>
     <audio src="audio/example0/dpai.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-3">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/example0/caw.wav" controls preload style="width: 200px;"></audio>
   </div>
   <div class="column-4">
     <h6>AudioGen<sub>gpt</sub></h6>
     <audio src="audio/example0/sga.wav" controls preload style="width: 200px;"></audio>
   </div>
     <div class="column-5">
     <h6>US8K</h6>
     <audio src="audio/example0/sga.wav" controls preload style="width: 200px;"></audio>
   </div>
</div>


### car_horn
### children_playing
### dog_bark
### drilling
### engine_drilling
### gun_shot
### jackhammer
### siren
### street_music
