# PathSnap Phase 2 Final Evaluation

## Setup
- Test trips: 10
- Candidate radius: 50 m
- Viterbi Top-K: 8
- Beta values tested: 10, 25, 50

## Accuracy

| Method | Accuracy | Improvement |
|---|---:|---:|
| Greedy | 82.65% | - |
| Viterbi beta 10 | 83.87% | +1.21 pp |
| Viterbi beta 25 | 84.75% | +2.09 pp |
| Viterbi beta 50 | 85.00% | +2.35 pp |

## Selected Model: Viterbi beta 50
- Accuracy: 85.00%
- Improvement: +2.35 percentage points

## Candidate Coverage
- Candidate miss rate: 0.00% on labelled observations.

## Limitations
- Reference routes may be imperfect.
- GPS gaps can affect alignment.
- Junction observations may match either adjoining arc.
- Unlabelled points are excluded.

## Results Files
- gt_fast_per_trip.csv
- gt_fast_summary.csv
- viterbi_gt_per_trip.csv
- viterbi_gt_summary.csv
- phase2_final_gt.csv