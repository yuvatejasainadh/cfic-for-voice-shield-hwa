# Cloud Inference Subsystem — Inter-Service Contracts

The Cloud Inference Subsystem operates entirely within private Docker bridge networking (`cfic-internal-network`).

## Microservice Registry

| Microservice | Internal Port | Primary Endpoint | Responsibility |
| :--- | :--- | :--- | :--- |
| **Inference Orchestrator** | `8200` | `POST /v1/inference` | End-to-end model coordination & result aggregation |
| **Model LID** | `8301` | `POST /identify` | Fast language identification |
| **IndicConformer English** | `8311` | `POST /transcribe` | English speech-to-text |
| **IndicConformer Telugu** | `8312` | `POST /transcribe` | Telugu speech-to-text |
| **IndicConformer Hindi** | `8313` | `POST /transcribe` | Hindi speech-to-text |
| **VMC-South Spoof Detector** | `8321` | `POST /predict` | South Indian acoustic deepfake detection (AASIST-L) |
| **AASIST-L Fallback** | `8322` | `POST /predict` | English & general acoustic deepfake detection |

---

## Detailed Specifications
Refer to the dedicated contract documents in:
- [Orchestrator Contract](../../cloud-inference/shared/contracts/inference_api_contract.md)
- [LID Contract](../../cloud-inference/shared/contracts/lid_contract.md)
- [ASR Contract](../../cloud-inference/shared/contracts/asr_contract.md)
- [Spoof Detector Contract](../../cloud-inference/shared/contracts/spoof_contract.md)
