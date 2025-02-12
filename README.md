1. Data Loading & Preprocessing: Reads a CSV file, extracts features and labels, normalizes the data, and splits it into train/test sets.
2. Data Visualization: Uses Matplotlib, Seaborn, and Plotly for visualizing dataset distributions and sample plots.
3. Autoencoder Model:  with an encoder (compressing the input) and a decoder (reconstructing the input). Trained using Mean Squared Error (MSE) loss to minimize reconstruction errors. 
4. Anomaly Detection Part: The autoencoder is trained only on normal data(Oscillation). The reconstruction loss is calculated for both normal(Oscillation) and anomalous(Other) data. If the reconstruction loss exceeds the computed threshold, the instance is classified as anomalous.
5. Last Part: performance matrix and confusion matrix


**Reconstruction Loss**
° The reconstruction loss is calculated using the Mean Absolute Error (MAE)
               loss = tf.keras.losses.mae(test_data, reconstructed)
° Here, test_data is the original input data. reconstructed is the output of the autoencoder after encoding and decoding the input. The MAE computes the absolute difference between each original and reconstructed value, then takes the average.

**Threshold**
° The threshold is set based on the training reconstruction loss:
                threshold = np.mean(train_loss) + np.std(train_loss)
°train_loss is the reconstruction loss computed for normal (Oscillation) training data. The threshold is set as the mean reconstruction loss plus one standard deviation. Any test sample with a reconstruction loss above this threshold is classified as an anomaly(Non-Oscillation).
