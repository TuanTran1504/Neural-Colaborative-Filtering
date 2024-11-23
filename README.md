This project is aim to reimplement the concept of Neural Matrix Factorization in this paper https://arxiv.org/abs/1708.05031

Collaborative Filtering is used widely across Recommendation Systems as it's capable to predict user preferences and make personalized recommendations by leveraging patterns and relationships found in user-item interaction data. Its popularity stems from its ability to:

* Make recommendations without needing detailed content information about items - it only requires user interaction data like ratings, clicks, or purchases
* Capture complex patterns and similarities between users and items that might not be obvious from item features alone
* Discover latent relationships in the data, meaning it can find hidden factors that influence user preferences
* Scale well to large datasets, especially when using modern matrix factorization techniques
* Adapt and improve as more user interaction data becomes available

In the paper, it is justified that Matrix Factorization (MF) can also be interpreted as a special case of their NCF framework. They have fused GMF and MLP under the NCF framework by letting the 2 model share the same embedding layer, and then combine the outputs of their interaction function
Here is their proposed model and I will try to recreate that
<img width="273" alt="image" src="https://github.com/user-attachments/assets/29b4c9c8-f1bd-4424-bc76-bd1dd1393460">


Dataset:
For the dataset that i used in this project, I will trying to used their framework but on a different data that they used in the paper.
Netflix Prize Dataset can be found here https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

Overall, it is very similar to the original dataset. While the Netflix data set has over 100 millions entries and the original has only 1 million, I will only use 1 million entries of the Netflix dataset for this project


Pre-trained model:
According to the paper, due to the non-convexity of the objective function of NeuMF, gradient-based optimization methods only find locally-optimal solutions. Therefore, intialization plays an important role for the convergence and performance of deep leaning model
The key points in this stage is the optimizer, they have adopted Adam for pretraining and SGD for the actual learning. This is because Adam needs to save momentum
information for updating parameters properly. As we initialize NeuMF with pre-trained model parameters only and forgo saving the momentum information, it is unsuitable tofurther optimize NeuMF with momentum-based methods.

Training and Validation:

After applying all the element as in the paper, including a pretrained GMF and MLP with intial weights, I have achieved a performance like below
<img width="343" alt="image" src="https://github.com/user-attachments/assets/d88f2c1e-18f7-49a0-9a59-89b94f673406">

The model above is the result of serveral tuning of the optimizer, lr and scheduler. The optimal values, as in the graph, is weight_decay= 1e-4, lr=0.01, momentum=0.9. Parameter of scheduler iclude factor=0.3, patience=5. 
Other layers or parameters of the model can be found in the code in this repo.

Conclusion:

The model is effective for reccomending task and used only two features users and items

Limitation:

I only test the model with 1 million entries, which might draw some limitation in a larger scale company. However, it could be a baseline for evaluation of different models in the future


