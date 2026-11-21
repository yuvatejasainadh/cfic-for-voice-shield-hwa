# PellaV2 Quantized

## Model artifact

- Model: PellaV2
- Variant directory: 293mb
- File: `293mb/pellav2_k12_int8_final.pt`
- Format: PyTorch checkpoint (`.pt`)
- Quantization: INT8, as indicated by the supplied filename.

## Calibration

See `config/k12_calibration.json` for the original uploaded calibration parameters.

## Repository contents

- `293mb/` — original quantized model checkpoint
- `config/` — model metadata and calibration
- `tokenizer/` — reserved for tokenizer artifacts
- `benchmark/` — benchmark documentation and results status
- `model_card.md` — model information and limitations
- `checksums.txt` — verified model checksum

## Benchmark status

No benchmark measurements were supplied. Performance and accuracy claims have not been made.

## Missing information

Architecture, parameter count, base model, tokenizer requirements, and runtime compatibility have not been independently established.

The original model weights are preserved without re-quantization.
