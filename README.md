

## fastkme : Fast Kaplan-Meier Estimator

This work implements two function `kaplan_meier_estimator` and `kaplan_meier_estimator_w`
The former one speed up the scikit survival kaplan meier curve using numba, the latter one is a weighted version of the former one.
to be used for Nearest Neighbors Weighted or Kenel Survival Estimator.

## Documentation

A [blog](https://www.statml.in/2024-03-11-faster-kme/)

## Installation

```bash
pip install git+https://github.com/yuvrajiro/fastkme
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


```

## Acknowledgments

* This work is inspired by the scikit-survival package
