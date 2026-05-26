"""Check sageattention + triton after setup_sage_windows.ps1."""
from externals.image2video.wan_i2v import _sage_status

ok, detail = _sage_status()
print("sage ready:", ok, "-", detail)
