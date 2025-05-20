import torch
import pandas as pd


def predict(preprocessor, model, data):
    """
    Predict using the trained model and preprocessor.
    This function takes a preprocessor, a model, and data as input,
    applies the preprocessor to the data, and then uses the model to make predictions.
    Args:
        preprocessor: scikit-learn preprocessor
        model: pytorch model
        data : pd.DataFrame
            The input data to be preprocessed and predicted on.
            It should be a pandas DataFrame containing the features used for prediction.
    The DataFrame should have the same structure as the training data used to fit the preprocessor.

    Returns:
        np.ndarray: predictions
            The predicted values as a numpy array.
            The shape of the output will depend on the model's output layer.
    """

    X = preprocessor.transform(data)
    if isinstance(X, pd.DataFrame):
        X = X.values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
    predictions = y_pred.cpu().numpy()
    # If the model has a single output, y_pred will be 2D with shape (n_samples, 1)
    # If the model has multiple outputs, y_pred will be 2D with shape (n_samples, n_outputs)
    # We can flatten it to 1D if needed
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    elif predictions.ndim > 1:
        predictions = predictions.reshape(predictions.shape[0], -1)

    return predictions
