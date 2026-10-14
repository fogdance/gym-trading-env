"""Process-local diagnostic switch; no synthetic claims of GPU acceptance."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('b2_diagnostics',Path(__file__).parents[2]/'tools/fg_s3_b2_diagnostics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_off_reads_original_state_without_rewriting_and_restores_method():
    class Schedule:
        value=.75
        reads=0
        def read(self):self.reads+=1;return np.float32(self.value)
    s=Schedule();original=Schedule.read
    with module.adapter_off(Schedule,np.zeros_like):
        assert s.read()==0 and s.value==.75 and s.reads==1
    assert Schedule.read is original and s.read()==.75
    with pytest.raises(RuntimeError):
        with module.adapter_off(Schedule,np.zeros_like):raise RuntimeError('exit')
    assert Schedule.read is original
