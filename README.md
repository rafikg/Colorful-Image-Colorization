TF2 implementation of  [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) 

The workflow normally has to be followed is 
1. Build the model.
2. Data preparation.
3. Train the model using Pascal dataset.
4. Evaluate the model on static image.
5. Evalaute the model on streaming input.
6. Create an API of the model using `Flask`.
7. Write the `Dockerfile`
8. Continous Integration with travis-CI.

9. `Extra:` deployment with kubeflow

