"""Optional RAPIDS acceleration, enabled before importing scikit-learn."""
import importlib.util
import sys


def enable_cuml_accel():
    if importlib.util.find_spec('cuml') is None:
        print('RAPIDS cuML is not installed; scikit-learn models will use CPU.', flush=True)
        return False

    import cupy

    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        print(f'RAPIDS CUDA check failed; scikit-learn models will use CPU: {exc}', flush=True)
        return False
    if count == 0:
        print('No CUDA device available for RAPIDS; scikit-learn models will use CPU.', flush=True)
        return False

    if 'sklearn' in sys.modules:
        raise RuntimeError('Restart the notebook kernel before enabling RAPIDS; scikit-learn is already imported.')

    import cuml

    cuml.accel.install(log_level='info')
    print('RAPIDS cuML accelerator enabled; unsupported operations fall back to CPU.', flush=True)
    return True
