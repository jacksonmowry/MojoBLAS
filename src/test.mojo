from sys import has_accelerator
from gpu.host import DeviceContext

def main():
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("GPU:", ctx.name())