# AUDIT A — MOTORCYCLE DETECTION VS E2E RESCUE AUDIT
## INVESTIGATION OF INCONSISTENCY IN HYBRID R1 FRESH HOLDOUT REPORT

**Date:** 2026-09-25  
**Audit Target:** `reports/fresh_holdout_500_results.csv` ($N = 100$ motorcycle samples)  
**Report Audited:** `reports/HYBRID_R1_FRESH_HOLDOUT_VALIDATION_REPORT.md`  

---

## 1. AUDIT SUMMARY & CORE STATUS

- **DETECTION RESCUES:** **11 plates** (Direct: 38 / 100 $\rightarrow$ R1: 49 / 100)
- **E2E RESCUES:** **10 plates** (Direct: 30 / 100 $\rightarrow$ R1: 40 / 100)
- **INCONSISTENCY STATUS:** **RESOLVED**
- **ROOT CAUSE:** A manual reporting/transcription error in the markdown text of `reports/HYBRID_R1_FRESH_HOLDOUT_VALIDATION_REPORT.md` where Direct E2E was erroneously transcribed as `27/100` and R1 E2E as `39/100` (+12 gain), whereas the actual per-sample ground truth in `fresh_holdout_500_results.csv` and the automated analysis dictionary strictly contain **Direct E2E = 30 / 100** and **R1 E2E = 40 / 100** (+10 gain).

---

## 2. SAMPLE-LEVEL PER-SAMPLE RECOMPUTATION

From the canonical per-sample evaluation file `reports/fresh_holdout_500_results.csv`:

### Detection States ($N = 100$ Motorcycles):
- `BOTH_DETECT`: **38 samples**
- `DIRECT_ONLY_DETECT` (Regressions): **0 samples**
- `R1_ONLY_DETECT` (Rescues): **11 samples**
- `NEITHER_DETECT`: **51 samples**
- **Direct Detection Recall:** $38 / 100 = \mathbf{38.00\%}$
- **R1 Detection Recall:** $(38 + 11) / 100 = 49 / 100 = \mathbf{49.00\%}$
- **Detection Delta:** $\mathbf{+11.00\text{ pp}}$ (11 net plates rescued)

### End-to-End Exact States ($N = 100$ Motorcycles):
- `BOTH_E2E`: **30 samples**
- `DIRECT_ONLY_E2E` (Regressions): **0 samples**
- `R1_ONLY_E2E` (Rescues): **10 samples**
- `NEITHER_E2E`: **60 samples**
- **Direct E2E Exact:** $30 / 100 = \mathbf{30.00\%}$
- **R1 E2E Exact:** $(30 + 10) / 100 = 40 / 100 = \mathbf{40.00\%}$
- **E2E Delta:** $\mathbf{+10.00\text{ pp}}$ (10 net plates rescued)

---

## 3. AUDIT OF THE 11 RESCUED SAMPLES

The 11 motorcycle samples where Direct failed but R1 successfully detected the plate:

| Sample ID | GT Plate Text | Direct Det | Direct E2E | R1 Det | R1 Source | R1 IoU | R1 OCR Prediction | R1 E2E Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `FH500_0021` | `36P30137` | False | False | True | CASCADE | 0.884 | `36P30137` | **Exact Hit** |
| `FH500_0053` | `29P151023` | False | False | True | CASCADE | 0.929 | `25G151023` | *OCR Miss (`29P` $\rightarrow$ `25G`)* |
| `FH500_0062` | `37F155306` | False | False | True | CASCADE | 0.865 | `37F155306` | **Exact Hit** |
| `FH500_0123` | `29B185992` | False | False | True | CASCADE | 0.871 | `29B185992` | **Exact Hit** |
| `FH500_0192` | `29C151116` | False | False | True | CASCADE | 0.852 | `29C151116` | **Exact Hit** |
| `FH500_0260` | `29X121540` | False | False | True | CASCADE | 0.890 | `29X121540` | **Exact Hit** |
| `FH500_0276` | `38C116816` | False | False | True | CASCADE | 0.843 | `38C116816` | **Exact Hit** |
| `FH500_0284` | `28L104337` | False | False | True | CASCADE | 0.892 | `28L104337` | **Exact Hit** |
| `FH500_0327` | `29X504646` | False | False | True | CASCADE | 0.866 | `29X504646` | **Exact Hit** |
| `FH500_0348` | `18N17020` | False | False | True | CASCADE | 0.875 | `18N17020` | **Exact Hit** |
| `FH500_0446` | `29L500295` | False | False | True | CASCADE | 0.888 | `29L500295` | **Exact Hit** |

---

## 4. RESOLUTION & CONCLUSION

1. **Logical Consistency Confirmed:**
   - Out of the 11 detection rescues, **10 resulted in an exact OCR match**.
   - Exactly **1 detection rescue** (`FH500_0053`) failed OCR due to character confusion (`29P151023` recognized as `25G151023`).
   - Therefore, the true empirical gain is **+11 detection rescues** and **+10 E2E rescues**, which satisfies strict logical monotonicity ($10 \le 11$).
2. **Evaluator Code Status:**
   - The automated evaluator `tools/analyze_fresh_holdout_500.py` and `reports/fresh_holdout_500_results.csv` contain the exact, correct numbers.
   - The markdown report `reports/HYBRID_R1_FRESH_HOLDOUT_VALIDATION_REPORT.md` will be updated to reflect Direct E2E = 30 / 100 (30.00%) and R1 E2E = 40 / 100 (40.00%).
