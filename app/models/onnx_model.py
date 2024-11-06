import onnxruntime
import numpy as np

from app.common.constants import DEFAULT_BATCH_SIZE, MNASNET


class ONNXModel:
    def __init__(self):
        self.session = onnxruntime.InferenceSession(MNASNET)

    def __call__(self, data, *args, **kwargs):
        input_name = self.session.get_inputs()[0].name
        if data.shape[0] <= DEFAULT_BATCH_SIZE:
            out = self.session.run(None, {input_name: data})
            return out[0]

        # split data into batch
        num_splits = int(data.shape[0] / DEFAULT_BATCH_SIZE)
        results = []
        for batch in np.array_split(data, num_splits, axis=0):
            out = self.session.run(None, {input_name: batch})
            results.append(out)
        return np.hstack(results).squeeze(0)

    def forward(self, data):
        return self.__call__(data)
