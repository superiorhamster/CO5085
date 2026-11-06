Train class counts min/max: 450/450. Balanced=True; no weighting/sampler correction.
32px images: keep low-resolution content; crop32/reflection-pad4/flip before resize224.
Hypothesis only: compare scratch augmentation vs noaug for each family, same budget.
ImageNet normalization is fixed to match pretrained inputs, not estimated using test.
Exact cross-split duplicate groups=12; no automatic deduplication.
