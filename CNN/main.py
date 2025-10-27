from train_util import train_model
from data_utils_cnn import load_constellaration_dataset, ConstellerationCNNDataset
from model_cnn import CombinedModel
from config_cnn import config, device

def main():
    """
    Load datasets, initialize the CombinedModel, and train it using the train_model utility.
    """
    # Load training and testing datasets
    train_hf, test_hf = load_constellaration_dataset(subset_hyperparam=None)
    train_dataset = ConstellerationCNNDataset(train_hf, device=device)
    test_dataset = ConstellerationCNNDataset(test_hf, device=device)

    # Initialize the model
    model = CombinedModel(
        n_theta=config["n_theta"],
        n_phi=config["n_phi"],
        mlp_input_dim=config["mlp_input_dim"],
        hidden_dim=config["hidden_dim"],
        num_outputs=config["num_outputs"]
    )

    # Train the model
    trained_model = train_model(
        model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=config["batch_size"],
        num_epochs=3,
        lr=config["learning_rate"],
        device=device,
        weight_decay=config["weight_decay"]
    )

    return trained_model

if __name__ == "__main__":
    model = main()



 