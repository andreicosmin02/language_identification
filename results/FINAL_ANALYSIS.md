# Final Analysis

## Summary

This file summarizes the experimental results for language identification on very short texts.

## Average Test Performance

| model | accuracy | macro_f1 | examples_per_sec |
| --- | --- | --- | --- |
| char_svm | 0.9601 | 0.9601 | 38644.7738 |
| fasttext | 0.9553 | 0.9554 | 99264.0637 |
| tfidf_lr | 0.9058 | 0.9083 | 224343.8572 |

## Main Findings

- Performance improves clearly when moving from `3` to `5` and then to `10` words.
- Noise affects the `3`-word setting the most.
- `char_svm` and `fasttext` are much more robust than `tfidf_lr` on very short texts.
- `tfidf_lr` remains the fastest at inference time, but it pays for that speed in robustness and accuracy on short noisy text.

## Best Model for Each Test Condition

- `3` words + `clean`: the best model is `fasttext` with accuracy `0.9340`
- `3` words + `no_diacritics`: the best model is `fasttext` with accuracy `0.9110`
- `3` words + `typo`: the best model is `fasttext` with accuracy `0.9090`
- `5` words + `clean`: the best model is `char_svm` with accuracy `0.9820`
- `5` words + `no_diacritics`: the best model is `char_svm` with accuracy `0.9730`
- `5` words + `typo`: the best model is `char_svm` with accuracy `0.9720`
- `10` words + `clean`: the best model is `char_svm` with accuracy `0.9940`
- `10` words + `no_diacritics`: the best model is `char_svm` with accuracy `0.9930`
- `10` words + `typo`: the best model is `char_svm` with accuracy `0.9930`

## Easier and Harder Languages

- `fr` has average accuracy `0.9196` across models and conditions
- `en` has average accuracy `0.9365` across models and conditions
- `es` has average accuracy `0.9365` across models and conditions
- `ro` has average accuracy `0.9515` across models and conditions
- `de` has average accuracy `0.9580` across models and conditions

## Conclusion

For this setup, `char_svm` and `fasttext` are better choices than `tfidf_lr` if the main goal is robust short-text language identification. If absolute inference speed is the main priority, `tfidf_lr` remains attractive. If robustness on very short and noisy texts matters most, `char_svm` and `fasttext` are the strongest options among the current baselines.
