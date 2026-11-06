# CIFAR-100 measured progress

Validated main: 8/8; ablation pairs: 2/2.

Missing/incomplete: 

Training times are concurrent queue costs, not isolated GPU speed.

One seed and a fixed budget do not establish convergence or statistical significance.

Only scratch augmentation effects were tested; test did not select the recipe.

Highest measured validation accuracy: deit_tiny_full: 0.8266; corresponding test=0.8259.

resnet18: best by validation = resnet18_full; never assume full fine-tune wins.

deit_tiny: best by validation = deit_tiny_full; never assume full fine-tune wins.