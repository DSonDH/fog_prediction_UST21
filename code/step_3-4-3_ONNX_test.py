import os
import numpy
import matplotlib.pyplot as plt
import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnxruntime as rt
import sklearn
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost
from xgboost import XGBClassifier
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
import onnxmltools
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa
import onnxmltools.convert.common.data_types

data = load_iris()
X = data.data[:, :2]
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

pipe = Pipeline([('scaler', StandardScaler()),
                 ('xgb', XGBClassifier(n_estimators=3))])
pipe.fit(X, y)

update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

model_onnx = convert_sklearn(
    pipe, 'pipeline_xgboost',
    [('input', FloatTensorType([None, 2]))],
    target_opset={'': 12, 'ai.onnx.ml': 2})

# And save.
with open("pipeline_xgboost.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

print("predict", pipe.predict(X[:5]))
print("predict_proba", pipe.predict_proba(X[:1]))

sess = rt.InferenceSession("pipeline_xgboost.onnx")
pred_onx = sess.run(None, {"input": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])