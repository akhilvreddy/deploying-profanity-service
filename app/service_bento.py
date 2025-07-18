# app/service_bento.py

from __future__ import annotations
import os
import joblib
import bentoml

@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class ProfanityService:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "profanity.joblib")

    def __init__(self) -> None:
        self.pipeline = joblib.load(self.MODEL_PATH)

    @bentoml.api
    def predict(self, text: str) -> dict[str, object]:
        """
        Accepts a text string, returns JSON:
        {
            "is_profane": bool,
            "confidence": float
        }
        """

        proba = self.pipeline.predict_proba([text])[0][1]
        is_profane = bool(proba > 0.5)
        confidence = float(round(proba, 4))
        return {
            "is_profane": is_profane,
            "confidence": confidence,
        }


# app/service_bento.py

# import os
# import joblib
# import bentoml
# from bentoml.io import JSON

# @bentoml.service(
#     resources={"cpu": "1"},
#     traffic={"timeout": 10},
# )
# class ProfanityService:
#     MODEL_PATH = os.path.join(os.path.dirname(__file__), "profanity.joblib")

#     def __init__(self) -> None:
#         self.pipeline = joblib.load(self.MODEL_PATH)

#     @bentoml.api(JSON(), JSON())
#     def predict(self, payload) -> dict[str, object]:
#         # payload is the parsed JSON body; expecting {"text": "..."}
#         text = payload["text"]
#         # get the probability as a Python float
#         proba_np = self.pipeline.predict_proba([text])[0][1]
#         proba: float = float(proba_np)
#         is_profane: bool = proba > 0.5
#         return {
#             "is_profane": is_profane,
#             "confidence": proba,
#         }