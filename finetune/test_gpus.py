"""
Quick diagnostic: test all GPUs on the instance.
Run FIRST before training to verify hardware.
Usage: python test_gpus.py
"""
import torch

def main():
    n = torch.cuda.device_count()
    print(f"CUDA devices found: {n}")
    if n == 0:
        print("ERROR: No GPUs detected!")
        return

    for i in range(n):
        try:
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_mem / 1e9:.1f} GB")
            t = torch.randn(1024, 1024, device=f"cuda:{i}", dtype=torch.bfloat16)
            _ = t @ t
            torch.cuda.synchronize(i)
            print(f"    -> OK (compute works)")
            del t
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    -> FAILED: {e}")

    if n > 1:
        print("\nNCCL test (GPU 0 <-> GPU 1)...")
        try:
            import torch.distributed as dist
            import os
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            # Can't test full NCCL without torchrun, but we can test peer access
            for i in range(n):
                for j in range(n):
                    if i != j:
                        can = torch.cuda.can_device_access_peer(i, j)
                        if not can:
                            print(f"  WARNING: GPU {i} cannot peer-access GPU {j}")
            print("  Peer access check done.")
        except Exception as e:
            print(f"  NCCL test error: {e}")

    print(f"\nAll {n} GPUs checked.")

if __name__ == "__main__":
    main()
