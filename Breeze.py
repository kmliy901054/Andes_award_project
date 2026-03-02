# Breeze.py  (Ollama version)
import requests
import time


class BreezeCoach:
    def __init__(
        self,
        model="jcai/breeze-7b-instruct-v1_0:q4_0",
        host="http://localhost:11434",
        timeout=120,
        enable_warmup=False,     # ✅ 預設關閉：避免一開始就「講太多」
    ):
        self.model = model
        self.url = f"{host}/api/generate"
        self.timeout = timeout
        self._warmed = False
        self.enable_warmup = enable_warmup

    def _warmup(self):
        # ✅ 只讓模型「載入/熱起來」，不拿回應、不丟 UI、不 TTS
        # ✅ num_predict=1，極短
        try:
            requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": "你好",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=60,
            )
        except Exception:
            pass

    def get_advice(self, exercise, geometry_status, lstm_status, max_new_tokens=80):
        if self.enable_warmup and (not self._warmed):
            self._warmup()
            self._warmed = True

        prompt = f"""你現在是一位專業、認真負責的女性健身教練。
使用者的動作是「{exercise}」，目的是矯正他的姿勢，並且你要告訴他哪邊用力太多或太少，還有哪邊要出力。
系統偵測數據：
1. 外在姿勢：{geometry_status}
2. 內在發力：{lstm_status}

請給使用者一個即時回饋（50字內，口語化），請盡量講重點就好。
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "num_predict": int(max_new_tokens),
                "temperature": 0.7,
            },
            "stream": False,
        }

        t0 = time.time()
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        dt = time.time() - t0

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        text = (data.get("response") or "").strip()
        print(f"[OLLAMA] dt={dt:.2f}s")
        return text
