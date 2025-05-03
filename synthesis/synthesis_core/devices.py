from dataclasses import dataclass
import jax

@dataclass
class DeviceMesh:
    devices: list
    n_devices: int

def get_tpu_mesh():
    devices = jax.devices()
    return DeviceMesh(devices=devices, n_devices=len(devices))
