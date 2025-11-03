from train_util import train_model
from data_utils_cnn import load_constellaration_dataset, ConstellerationCNNDataset
from model_cnn import CombinedModel
from config_cnn import config, device


train_hf, test_hf = load_constellaration_dataset()
print("Train size:", len(train_hf))
print("Test size:", len(test_hf))




def main():
    

    
    train_hf, test_hf = load_constellaration_dataset(subset_hyperparam=None)
    train_dataset = ConstellerationCNNDataset(train_hf, device=device)
    test_dataset = ConstellerationCNNDataset(test_hf, device=device)

    
    model = CombinedModel(
        n_theta=config["n_theta"],
        n_phi=config["n_phi"],
        mlp_input_dim=config["mlp_input_dim"],
        hidden_dim=config["hidden_dim"],
        num_outputs=config["num_outputs"]
    )

    
    trained_model = train_model(
        model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=config["batch_size"],
        num_epochs=config['num_epochs'],
        lr=config["learning_rate"],
        device=device,
        weight_decay=config["weight_decay"]
    )

    return trained_model

if __name__ == "__main__":
    print("Starting training...")
    model = main()
    print("Training completed.")





 