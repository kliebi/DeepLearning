# DeepLearning

1. implement a function that explores the VAE latent space and visualise the grid (20x20) of the latent space (MNIST)
2. compare latent spaces of AE and VAE
3. (AE and VAE) select an encoder layer (2nd or 3rd) and a channel in this layer. Then generate / find a dream image for the selected channel in the selected layer.


Task 1) 	
Start python program "mnist_vae.py" 
Use function "train_and_store_VAE(50)" for training. Parameter is amount of epochs - here in this example 50. 
While applying the function following pictures will be generated:
	Shows the clusters of the VAE Latent space - mnist_vae_label_clusters.jpg
	Visualise the grid of the latent space (MNIST) - mnist_vae_latent_space.jpg
	
VAE Model will be trained and the VAE model including the encoder and decoder with the weights will be saved.
	
	
Task 2) 
Start python program "mnist_ae_latent_space.py" 
Use function "train_and_store_AE(50)" for training. Parameter is amount of epochs - here in this example 50. While applying the function several pictures will be generated:

Shows the clusters of the AE Latent space - AE_mnist_latent_space-clusters.jpg
Visualise the grid of the latent space (MNIST) - AE_mnist_latent_space.jpg
Visualise the predictions of the AE model of train data (MNIST) - AE_mnist-train_data-predictions.jpg

AE Model will be trained and the AE model including the encoder and decoder with the weights will be saved the following files:
	ae_encoder.json
	ae_encoder_weights.h5
	ae_decoder.json
	ae_decoder_weights.h5
	ae_model.json
	ae_model_weights.h5
To load the trained model and to viszualize the latent space of the AE Model with the loaded model use funtion >>> load_and_predict_AE()

The follwoing pictures will be generated:
	AE_mnist_latent_space-clusters_loaded.jpg --> Shows the clusters of the VAE Latent space generated from the loaded AE Encoder
	AE_mnist_latent_space_loaded.jpg --> Visualise the grid of the latent space (MNIST) from the loaded AE Decoder
  
After running 1) and 2) latent Spaces of AE and VAE trained on the MNIST Data set can be compared. 
