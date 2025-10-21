import config
from data_utils import load_constellaration_dataset, prepare_tensor_dataset
from model import StellaratorNet
from train import train_model

def main():
    # --- Load data ---
    train_data, test_data = load_constellaration_dataset(subset_hyperparam=config.subset_hyperparam)
    train_ds = prepare_tensor_dataset(train_data, config.device, config.fixed_len)
    test_ds = prepare_tensor_dataset(test_data, config.device, config.fixed_len)
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # --- Build model ---
    input_size = train_ds[0][0].shape[0]
    model = StellaratorNet(input_size, config.dropout_prob).to(config.device)

    # --- Train ---
    train_model(model, train_ds, test_ds, config)

if __name__ == "__main__":
    main()




