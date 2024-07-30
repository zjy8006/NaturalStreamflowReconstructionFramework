def StandardScale(calibration_samples,test_samples,target_name):
    target_mean = calibration_samples[target_name].mean()
    target_std = calibration_samples[target_name].std()
    for c in calibration_samples.columns:
        mean = calibration_samples[c].mean()
        std = calibration_samples[c].std()
        calibration_samples[c] = (calibration_samples[c] - mean) / std
        test_samples[c] = (test_samples[c] - mean) / std
    return calibration_samples,test_samples,target_mean,target_std

def MinMaxScale(calibration_samples,test_samples,target_name,scale_range=(0,1)):
    target_min = calibration_samples[target_name].min()
    target_max = calibration_samples[target_name].max()
    if scale_range == (0,1):
        for c in calibration_samples.columns:
            min_ = calibration_samples[c].min()
            max_ = calibration_samples[c].max()
            calibration_samples[c] = (calibration_samples[c] - min_) / (max_ - min_)
            test_samples[c] = (test_samples[c] - min_) / (max_ - min_)
    elif scale_range == (-1,1):
        for c in calibration_samples.columns:
            min_ = calibration_samples[c].min()
            max_ = calibration_samples[c].max()
            calibration_samples[c] = 2*(calibration_samples[c] - min_) / (max_ - min_)  - 1
            test_samples[c] = 2*(test_samples[c] - min_) / (max_ - min_)  - 1
    return calibration_samples,test_samples,target_min,target_max
        

def MaxAbsScale(calibration_samples,test_samples,target_name):
    target_max = calibration_samples[target_name].max()
    for c in calibration_samples.columns:
        max_ = calibration_samples[c].max()
        calibration_samples[c] = calibration_samples[c] / max_
        test_samples[c] = test_samples[c] / max_
    return calibration_samples,test_samples,target_max