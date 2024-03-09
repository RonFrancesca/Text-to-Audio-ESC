todo:

understad why the network is not working: 
- normalization: how do ohter people do


Notice: 
- the shuffle in the dataset split is only for make it run. It will need to be modified or handle in a diferent way for the validation or for a real run
Important:
- now there is only the normalization spec by spec. It will need to be modified afterwards if also the other normalization need to tested

# 
1. Data augmentation: implementare le tecniche di Salamon nel paper oltre questa come data augmentation e le varie combinazioni (PS, TS, Mixup, GaussianNoise)
2. Dataset da soli e dataset concatenati
3. 100, 200, 300 dati generati etc.
4. Rete 1, Rete 2 e Rete3: baseline, dcase 
5. Aumentando il dataset, possiamo avere delle reti più piccole?

Things to do:
- code need to be cleaned and the project need to be better written
- create a single file for the inference process
- try with different networks
- DCASE baseline to try: https://github.com/marmoi/dcase2022_task1_baseline
- ResNet: reti piu piccole per performance migliori? Possiamo avere con un dataset migliore delle reti piu piccoli?
- set the patience into the default parrameters
- how the concatenation between csbv files and dataset is handled need to be changed 
- fine-tuning of parameters 
- need to find a way to do not copy and paste all the data but directly cpy and paste the X runs of of the model
- find a way to add the information related to the energy consumption of the models
- main need to be a function that i will call so that everything will be better
- change how the data augmentation need to be saved. In the way it is now, I need to manually change the number of the value of the data augmentation otherwise it won't save the values correctly on the CVS file. 
The other way around, loop first trought the values and then tworught the files is the best way. 
it could also simply create a single csv file to get all of them


Experiments to do:
- change the parameters regularization for the CVNN as done on the paper

Future works:
- different networks
- fine tune the models with the sounds we want



