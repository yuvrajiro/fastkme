

## fastkme : Fast Kaplan-Meier Estimator

This work implements two function `kaplan_meier_estimator` and `kaplan_meier_estimator_w`
The former one speed up the scikit survival kaplan meier curve using numba, the latter one is a weighted version of the former one.
to be used for Nearest Neighbors Weighted or Kenel Survival Estimator.

## Installation

```bash
pip install fastkme
```

## Usage

```python
from fastkme import kaplan_meier_estimator, kaplan_meier_estimator_w
import numpy as np
np.random.seed(0)
time = np.random.randint(0, 100, 100)
event = np.random.randint(0, 2, 100)
unique_time, survival_prob = kaplan_meier_estimator(time, event)
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

If you use this work, please cite the following paper:

```
@misc{goswami2023areanorm,
      title={Area-norm COBRA on Conditional Survival Prediction}, 
      author={Rahul Goswami and Arabin Kr. Dey},
      year={2023},
      eprint={2309.00417},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgments

* This work is inspired by the scikit-survival package