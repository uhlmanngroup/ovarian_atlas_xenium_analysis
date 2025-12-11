import pickle
import typing


def save_pickle(save_path: str, save_object: typing.Any) -> None:
    with open(save_path, "wb") as handle:
        pickle.dump(save_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_picke(load_path: str):
    with open(load_path, "rb") as handle:
        loaded_object = pickle.load(handle)
    return loaded_object
