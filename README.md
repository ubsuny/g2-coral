# Final Project

This repository is for the final project of PHY 506 Computational Physics II. The ultimate goal is to implement the CNN model on Google Coral Dev Board / Google Edge TPU for g2 temporal correlation data analysis. Beyond the final project, this will still be worked on since there're several parts that could be improved or need to be solved.

# Contents

+ `Model` : It contains the CNN model we developed, could be visualized on the jupyter notebook, and a `.h5` file saving a Keras model.
+ `Quantization` : We're not able to implement the model on TPUs, due to some layers in the model are not fully quantized. This will be further investigated and several testing conveted models and a ongoing notebook about it could be found.
+ `plots` : Including plots for midterm milestone, though ther're not related to the final.
+ `py_programme` : Python files including main implementations for this project and a linting report from Pylint.
+ `sdt_data` : Actual measurement data files.
+ `simulation` : Several notebooks represent different simulation methods. We finally adopt the `sdt_simulation` which uses Monte Carlo Method on Tensorflow. Simulated data are included.
+ `unittest` : A `.ipynb` to see the visualization, simulated data file and a test kears model file.
+ `wiki_figures` : Figures and some sketches for wiki.
