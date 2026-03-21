import json
import os

folder = "/Users/banan/Documents/voice cli/model_source/Abdul Halim Bin Abdul Rahman - group1_vlad_EN_48k"

params = {
  "slotIndex": 0,
  "name": "Vlad RVC",
  "description": "",
  "credit": "",
  "termsOfUseUrl": "",
  "iconFile": "",
  "speakers": {
    "0": "target"
  },
  "modelFile": "TimbreNode/G_Final_EMA.pth",
  "indexFile": "faiss_index/added_IVF1381_Flat_nprobe_1_v2.index",
  "defaultTrans": 0,
  "defaultTune": 0,
  "defaultIndexRatio": 1.0,
  "defaultProtect": 0.5,
  "isONNX": False,
  "modelType": "PyTorch",
  "samplingRate": 48000,
  "f0": True,
  "embChannels": 768,
  "embOutputLayer": 12,
  "useSR": True,
  "deprecated": False
}

with open(os.path.join(folder, "params.json"), "w") as f:
    json.dump(params, f, indent=4)
print("done")
