


def calibration_test_split(samples,test_ratio=0.2):
    test_start_index = samples.index[int(samples.shape[0] * (1-test_ratio))]
    calibration_samples = samples.loc[samples.index<test_start_index].copy()
    test_samples = samples.loc[samples.index>=test_start_index].copy()
    return calibration_samples,test_samples