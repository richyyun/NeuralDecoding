As I am now much more comfortable with neural networks and machine learning, I wanted to tackle a problem that is both more complex and within the BCI field. 
A large part data analysis in neuroscience both in research and industry is neural decoding, or determining intention or motion from recorded neural data. 
Machine learning is clearly a useful tool for neural decoding, but there are two main issues with currently no good solutions:
1. Neural data is highly variable. Neural data changes drastically on a day-to-day basis even with implanted invasive electrodes, not to mention inter-subject variability. Traditional neural networks will have to be retrained every once in a while or somehow incorporate extreme generalizability which goes against the purpose of decoding.  
2. Decoding a large space requires an incredible amount of data. As an example, decoding the range of motion for a single arm would require data for each degree of freedom and possibly direction for a traditional neural network. Neural signals are also often not linear, so each different combination of movement may be necessary for proper decoding. 

I want to approach these problems in two steps: 
1. One-shot learning. The problem of generalizability is also highly prevalent in facial recognition as it is impossible to capture every angle of a face with different makeup, hair, clothes, or accessories. A solution is a siamese neural network, which consists of two identical parallel networks processing two inputs, which is trained to detect the similarity between the inputs. A siamese network can be trained to discern between different neural signals 
2. Zero-shot learning. 
