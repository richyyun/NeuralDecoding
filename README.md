# NeuralDecoding

\**Work in progress*\*

## Goal of the project
My goal is to decode data from a Utah array implanted in primary motor cortex of an animal performing a 2-dimensional center out task with wrist torques. The task first prompts the animal to relax and have the cursor in the middle of the screen. Then a target appears on the edge of the screen (one of 8 possible locations: left, right, up, down, and each corner) which cues the animal to move the cursor into the target. After the cursor reaches the target, the target moves to the center so the animal relaxes again. The data consists of 96 channel LFP data of 0.5 seconds before to 2.5 seconds after trial completion sampled at 500 Hz. 

## To do
- ~~Curate the data into sections of LFP for each trial~~
- Normalize the data and plot to see trial-dependent changes **&larr; Currently in progress**
- Implement an RNN to ensure the raw LFP or the spectra can accurately predict trial types
- Implement the simaese neural network - start with simple LSTM, possibly with an initial 1D convolution to further compress the data.  
  - Determine an effective triplet loss implementation
    - Triplet loss can be tricky, and may require online mining to find semi-hard triplets
  - Train on all trial types
  - Train limited to 4 trial types
  - Compare the two networks and their outputs 
- Implement zero-shot learning with the outputs of the siamese neural network  

## Approach
As I am now much more comfortable with neural networks and machine learning, I wanted to tackle a problem that is both more complex and within the BCI field. A large part of data analysis in neuroscience both in research and industry is neural decoding, or deciphering what neural signals mean. Machine learning is clearly a useful tool for neural decoding, but there are two main issues that need to be solved for proper commercialization:
1. **Neural data is highly variable.** Neural data changes drastically on a day-to-day basis even with implanted invasive electrodes, not to mention inter-subject variability or changes due to non-invasive headset positioning. Traditional neural networks will have to be constantly retrained or somehow incorporate extreme generalizability. The data in this project was collected from a single animal but across 8 months, which means we need to figure out how to tackle this problem.
2. **Decoding a large space requires an incredible amount of data.** As an example, decoding the range of motion for a single arm would require data for each degree of freedom and possibly each direction. Neural signals are also often not linear, so data each different combination of movements may be necessary for proper decoding. Our data consists of 8 specific targets, but a network trained on those targets could not easily be expanded to include more targets.

A simple recurrent neural network would likely be able to accurately decode the neural data per session, but I wanted to approach the problems outlined above. To that end, I want to implement two solutions: 
1. **One-shot learning.** The problem of generalizability is also highly prevalent in facial recognition as it is impossible to capture every angle of a face with different hair, makeup, clothes, or accessories. A common solution is a siamese neural network, which consists of two identical parallel networks trained to recognize the similarity between inputs. We can expand this to neural decoding by training a neural network to discern between different neural signals. 
2. **Zero-shot learning.** The problem of identifying a novel input into potentially a novel class exists in both natural language processing and computer vision. As a network may not be trained with all possible inputs and classes, a neural network may be trained to learn the embeddings of each input and  process how similar a novel input is to those embeddings to determine a new class. Zero-shot learning has already been used on decoding fMRI signals when participants are shown different images. A simple proof of concept for our purposes would consist of classifying a subset of trial types that was not used in initial training.     

To that end, I will train a siamese neural network to discern between different types of input data consisting of left, right, up, or down targets. The output of the network is essentially an "embedding" of the data, which can then be used to determine whether novel data belongs to one of the original trial types or a new trial type. The zero-shot learning can be implemented in a smalle trained neural network with fully connected layers or a simple heuristic. 

<p align="center">
  <img src="https://github.com/richyyun/NeuralDecoding/blob/master/Approach.png" />
</p>
