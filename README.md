As a part of the project for CS549:
worked on HAN - https://arxiv.org/pdf/1903.07293 on improving the perfomance. original code is found here. https://github.com/Jhy1993/HAN 

Result:
Hyperparameter Tuning:,
parser.add_argument('--lr', type=float, default=0.001,  # reduced from 0.005
parser.add_argument('--weight_decay', type=float, default=0.0005,  # reduced from 0.001
parser.add_argument('--hidden_dim', type=int, default=64,  # increased from 8
parser.add_argument('--dropout', type=float, default=0.5,  # reduced from 0.6
parser.add_argument('--alpha', type=float, default=0.1,  # reduced from 0.2
parser.add_argument('--q_vector', type=int, default=256,  # increased from 128

Added  Label Smoothing (nn.CrossEntropyLoss(label_smoothing=0.1)),
Label smoothing is a regularization technique that prevents the model from becoming overconfident in its predictions by "softening" the hard target labels.

Added Learning rate scheduling,
This learning rate scheduler reduces the learning rate when a metric (typically validation loss) stops improving.

Added More Evaluation Metrics,
Precision and recall. 


Tried: Modifying the Model Architecure by adding batch normalisation, additng additional hidden layer and initialising weights and adding non-linearity like leaky_relu. 
Also tried adding gradient clipping.  – no better result instead the result was a bit worst.

![image](https://github.com/user-attachments/assets/277c48b7-acc5-4da2-b5bc-89c17fd77ace)
