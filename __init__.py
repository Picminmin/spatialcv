# __init__.pyがあるディレクトリはPythonパッケージとして認識されるようになる。
# spatialcvディレクトリ内に__init__.pyがあるので, spatialcvを読み込める！
from .spatial_train_test_split import spatial_train_test_split, __version__
__all__ = ["spatial_train_test_split", "__version__"]
