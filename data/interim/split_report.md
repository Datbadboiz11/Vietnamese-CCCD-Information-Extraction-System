# Split Report

## Ket qua split 70/15/15
| Split | So anh | % | So annotation |
|---|---|---|---|
| train | 3074 | 69.9% | 21535 |
| val   | 699 | 15.9% | 4905 |
| test  | 626 | 14.2% | 4414 |

## Verify
- Leakage train-val: NONE
- Leakage train-test: NONE
- Leakage val-test: NONE
- Tong so anh khop: 4399 == 3074+699+626

## Config
- Seed: 42
- Stratified by: difficulty score (fields + bbox_size + brightness)
- Cluster strategy: base_name grouping (Roboflow augmented copies)