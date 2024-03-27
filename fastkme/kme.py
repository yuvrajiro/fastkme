import numpy as np
import numba

@numba.jit(nopython=True)
def kaplan_meier_estimator(event, time_exit):
    uniq_times, n_events, n_at_risk, n_censored = _compute_counts(event, time_exit)

    ratio = divide(n_events, n_at_risk)
    values = 1.0 - ratio
    prob_survival = np.cumprod(values)
    return uniq_times, prob_survival


@numba.jit(nopython=True)
def kaplan_meier_estimator_w(event, time_exit, weight):
    uniq_times, n_events, n_at_risk, n_censored = _compute_counts_w(event, time_exit, weight)

    ratio = divide(n_events, n_at_risk + 1e-160)
    values = 1.0 - ratio
    prob_survival = np.cumprod(values)
    return uniq_times, prob_survival


@numba.jit(nopython=True)
def _compute_counts_w(event, time, weight, order=None):
    """
    Compute counts for weighted kaplan mier
    :param event:
    :param time:
    :param weight:
    :param order:
    :return:
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=np.float64)
    uniq_counts = np.empty(n_samples, dtype=np.float64)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += weight[order[i]]

            count += weight[order[i]]
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = resize(uniq_times, j)
    n_events = resize(uniq_events, j)
    total_count = resize(uniq_counts, j)
    n_censored = total_count - n_events
    total_count = append_zero(total_count)
    n_at_risk = weight.sum() - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored


@numba.jit(nopython=True)
def divide(n_events, n_at_risk):
    """
    Do a simple division
    :param n_events:
    :param n_at_risk:
    :return:
    """
    out = np.zeros(n_events.shape[0], dtype=float)
    for i in range(n_events.shape[0]):
        if n_events[i] != 0:
            out[i] = n_events[i] / n_at_risk[i]
    return out


@numba.jit(nopython=True)
def _compute_counts(event, time, order=None):
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=np.int32)
    uniq_counts = np.empty(n_samples, dtype=np.int32)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = resize(uniq_times, j)
    n_events = resize(uniq_events, j)
    total_count = resize(uniq_counts, j)
    n_censored = total_count - n_events
    total_count = append_zero(total_count)
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored


@numba.jit(nopython=True)
def append_zero(times):
    """
    This function concatenates 0 to the times array
    :param times:
    :return: return the times array with 0 concatenated
    """
    return np.concatenate((np.array([0]), times))


@numba.jit(nopython=True)
def resize(a, new_size):
    """
    Resize an flattened array to new_size
    :param a: Array
    :param new_size: new size
    :return: array tr
    """
    new = np.zeros(new_size, a.dtype)
    idx = 0
    while True:
        newidx = idx + a.size
        if newidx > new_size:
            new[idx:] = a[:new_size - newidx]
            break
        new[idx:newidx] = a
        idx = newidx
    return new