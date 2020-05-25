
from utils import preprocess_adj,plot_embeddings, load_data_v1


if __name__ == "__main__":
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1('cora')