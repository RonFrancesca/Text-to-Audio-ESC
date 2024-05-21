<div align="center">

# Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification

<!-- <img width="700px" src="docs/new-generic-style-transfer-headline.svg"> -->
 
[Francesca Ronchini](https://www.linkedin.com/in/francesca-ronchini/)<sup>1</sup>, [Luca Comanducci](https://lucacoma.github.io/)<sup>1</sup>, and [Fabio Antonacci](https://www.deib.polimi.it/ita/personale/dettagli/573870)<sup>1</sup>

<sup>1</sup> Dipartimento di Elettronica, Informazione e Bioingegneria - Politecnico di Milano<br>
    
[![arXiv](https://img.shields.io/badge/arXiv-2403.17864-b31b1b.svg)](https://arxiv.org/abs/2403.17864)

</div>


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Abstract](#abstract)
- [Install & Usage](#install--usage)
- [Link to additional material](#link-to-additional-material)
- [Additional information](#additional-information)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
    
## Abstract

In the past few years, text-to-audio models have emerged as a significant advancement in automatic audio generation. Although they represent impressive technological progress, the effectiveness of their use in the development of audio applications remains uncertain. This paper aims to investigate these aspects, specifically focusing on the task of classification of environmental sounds. This study analyzes the performance of two different environmental classification systems when data generated from text-to-audio models is used for training. Two cases are considered: a) when the training dataset is augmented by data coming from two different text-to-audio models; and b) when the training dataset consists solely of synthetic audio generated. In both cases, the performance of the classification task is tested on real data. Results indicate that text-to-audio models are effective for dataset augmentation, whereas the performance of the models drops when relying on only generated audio.


## Install & Usage

For generating the data, we used AudioLDM2 and AudioGen. 

### Intalling AudioLDM2

Please refer to the [AudioLDM2 GitHub repo](https://github.com/haoheliu/AudioLDM2?tab=readme-ov-file#hugging-face--diffusers) and follow the installation instructions. For this study, we used the official checkpoints available in the Hugging Face 🧨 Diffusers and the <i>audioldm</i> checkpoint. 

When AudioLDM2 has been installed, you can generate the audio files running the script <i>audio_generation/class_generation_audioldm.py</i>
Before running the script, you need to specify the path to the output folder, the audio class to generate, the prompt to use to generate the files, and the number of files to generate in the <i>audio_generation/class_generation_audiogen.py</i>.

After that, you can run the script with the command: 

```
cd audio_generation
python class_generation_audioldm.py
```


### Intalling AudioGen

Please refer to the [AudioGen GitHub repo](https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md#installation) and follow the installation instructions. 

When AudioGen has been installed, you can generate the audio files running the script <i>audio_generation/class_generation_audiogen.py</i>.
Before running the script, you need to specify the path to the output folder, the audio class to generate, the prompt to use to generate the files, and the number of files to generate in the <i>audio_generation/class_generation_audiogen.py</i>. 

```
cd audio_generation
python class_generation_audiogen.py
```

### Run the code
When all the data have been generated, you can reproduce the experiments. 

First, install all the packages required by the system. Run the following command on your terminal to install all the packages needed:

```
pip install -r requirements.txt
```

When all packages have been installed, you need to specify which dataset to use following the instructions on the <i>config/default.yaml</i> file. 

After all the parameters have been defined, you can run the code with the following command:

```
python main.py
```

## Link to additional material

Additional material and audio samples are available on the [companion website](https://ronfrancesca.github.io/Text-to-Audio-ESC/). 


## Additional information

For more details:
"[Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification](https://arxiv.org/abs/2403.17864)", [Francesca Ronchini](https://www.linkedin.com/in/francesca-ronchini/), [Luca Comanducci](https://lucacoma.github.io/), and [Fabio Antonacci](https://www.deib.polimi.it/ita/personale/dettagli/573870) - arXiv, 2024. 


If you use code or comments from this work, please cite our paper:

```BibTex
@article{ronchini2024synthesizing,
  title={Synthesizing Soundscapes: Leveraging Text-to-Audio Models for Environmental Sound Classification},
  author={Ronchini, Francesca and Comanducci, Luca and Antonacci, Fabio},
  journal={arXiv preprint arXiv:2403.17864},
  year={2024}
}
```

