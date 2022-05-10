import os


def get_path():
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle-training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


path = get_path()


def test_ingest():
    datapath = "dataset/raw/housing"
    os.system(f"python3 src/housing/ingest_data.py --datapath {datapath}")
    print(f"{path}{datapath}/housing.csv")
    assert os.path.isfile(f"{path}{datapath}/housing.csv")
    assert os.path.isfile(f"{path}data/processed/train_X.csv")
    assert os.path.isfile(f"{path}data/processed/train_y.csv")


def test_train():
    models = "outputs/artifacts"
    dataset = "data/processed"
    model_names = ["lin_model", "tree_model", "forest_model", "grid_search_model"]
    os.system(
        f"python src/housing/train.py --inputpath {dataset} --outputpath {models}"
    )
    print(f"{path}{models}/{model_names[0]}/model.pkl")
    assert os.path.isfile(f"{path}{models}/{model_names[0]}/model.pkl")
    assert os.path.isfile(f"{path}{models}/{model_names[1]}/model.pkl")
    assert os.path.isfile(f"{path}{models}/{model_names[2]}/model.pkl")
    assert os.path.isfile(f"{path}{models}/{model_names[3]}/model.pkl")


test_ingest()
test_train()
