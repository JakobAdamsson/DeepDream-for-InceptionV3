# DeepDream Model
This is a project for the course in AI security.

DeepDream is based on gradient ascent where the goal is to maximize the loss function with respect to the input image to modify the image in the direction where the neurons in the selected layer react most strongly.

DeepDream is a computer vision program created by Google that uses a convolutional neural network to find and enhance patterns in images, creating a dream-like, surreal appearance. It's based on the InceptionV3 architecture, which is a model trained on the ImageNet dataset for image classification.

The purpose of providing this implementation is to facilitate the reader and to avoid having to implement it yourself. If the reader wants to explore the model on their own, they can easily change the hyperparameters:
1. STEPS
2. STEPSIZE
3. OCTAVE_SCALE

Where OCTAVE_SCALE is a scalar that enlarges the image, thereby creating stronger patterns, while a smaller value creates a blurrier and less dream-like image.

## Running the Model
Start by creating a virtual environment in the directory of your choice, then run
```bash
python -m venv myenv
```

to create a virtual environment. Activate it by entering
```
myenv\Scripts\activate
```

Finally, you need to install all the packages to run the model, which is done through

```
pip install -r requirements.txt
```

To run the model, you need to process one layer at a time, that is either layer0, layer5, or layer10. Once you have chosen which layer you want to run, adjust the hyperparameters as desired.
Then start the program and wait for the cell to complete. The image seen after the program has finished executing is the mutated image where, depending on the layer, different patterns can be seen.