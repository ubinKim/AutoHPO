# AutoHPO
This repository contains the code and some materials used for the Bachelor's project thesis conducted by students of Ghent University.
This project aims to compare the performance of three automated hyperparameter optimization (AutoHPO) algorithms on the given CNN model and dataset for the translation initiation site (TIS) detection.

- __Dissertation title__: _A Comparative Study of Grid Search, Random Search, and Particle Swarm Optimization for Automated Hyperparameter Optimization_
- __Academic year__: 2022-2023
- __Research centre__: Centre for Biosystems and Biotech Data Science, GUGC


## What is HPO?
Hyperparameter optimization (HPO), also known as hyperparameter tuning, is the process of finding the best hyperparameter configurations for a specific problem and dataset. It is essential for achieving optimal model performance, including accuracy, generalization ability, and computational efficiency. However, traditional approaches to HPO have become increasingly challenging because they are very time-consuming and computationally intensive as the complexity of the model increases. These limitations raise the need for AutoHPO algorithms to explore the hyperparameter space more efficiently and automate the HPO process without human interference.

```AutoHPO``` uses accuracy as a measure for the performance evaluation.
To determine the most efficient algorithm, computational costs are measured in terms of the execution time and the number of function evaluations.


## Data and Model
TIS datasets used in this project can be found in the ```/data``` folder.

- tr_5prime_utr.pos: TIS-positive sequences used for training
- tr_5prime_utr.neg: TIS-negative sequences used for training
- val_5prime_utr.pos: TIS-positive sequences used for validation
- val_5prime_utr.neg: TIS-negative sequences used for validation

The codes for pre-processing (```cls_TIS_dataset.py```) and the simple CNN model (```cls_TIS_model.py```) can be found under the ```/src``` folder.

Those are provided by the following paper: <br>
Utku Ozbulak, Hyun Jung Lee, Jasper Zuallaert, Wesley De Neve, Stephen Depuydt, Joris Vankerschaver. [Mutate and Observe: Utilizing Deep Neural Networks to Investigate the Impact of Mutations on Translation Initiation](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btad338/7177993 "Mutate_and_Observe"), Bioinformatics, 2023.


## Code 
__1. AutoHPO algorithms <br>__
For this project, three AutoHPO algorithms were implemented:

- Grid Search (GS)
- Random Search (RS)
- Particle Swarm Optimization (PSO) 

To train and evaluate the model and implement AutoHPO algorithms, use ```main_inference.py``` and ```main_functions.py``` located in the ```/src``` folder.

__2. Visualizations <br>__
The codes for the data visualization can be found in the ```/visualizations``` folder.


## Results
Some results of this project can be found under the ```/results``` folder.

(* The remaining results will be uploaded shortly.)


## Acknowledgement
We would like to express our sincere gratitude to our supervisor and counselors for their invaluable support and guidance throughout the completion of our bachelor's project thesis.

- Prof. Dr. Joris Vankerschaver (@jvkersch)
- Ms. Negin Harandi (@negin17h)
- Mr. Utku Özbulak (@utkuozbulak)
