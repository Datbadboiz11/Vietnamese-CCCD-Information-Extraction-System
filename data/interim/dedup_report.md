# Deduplication Report

## Tong quan
- Tong so anh: 4399
- So anh bi loi hash: 0
- Tong so cluster: 769
  - Cluster >= 2 anh (co ban trung/augment): 698
  - Cluster don le (unique): 71
- Anh nam trong cluster trung: 4328 (98.4%)

## pHash (Hamming <= 8)
- Cap trung: 4244
- Cluster pHash >= 2: 648

## Base-name grouping (Roboflow augmented copies)
- So base name unique (anh goc): 769
- Base name co >= 2 ban: 698

## Phan bo kich thuoc cluster
- Cluster size 1: 71 cluster
- Cluster size 2: 1 cluster
- Cluster size 3: 172 cluster
- Cluster size 4: 6 cluster
- Cluster size 5: 103 cluster
- Cluster size 6: 5 cluster
- Cluster size 7: 229 cluster
- Cluster size 9: 182 cluster

## Cross-split leakage (cluster co anh o nhieu split khac nhau)
- Cluster bi leakage: 347
- -> Phai re-split theo cluster de tranh data leakage