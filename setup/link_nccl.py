"""将 PyTorch 使用的 NCCL 开发文件暴露为 NCCL_HOME，避免另装不同版本。"""
from pathlib import Path
import sys
import sysconfig

source = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "nccl"
target = Path(sys.prefix) / "opt" / "nccl"
assert (source / "include" / "nccl.h").is_file(), f"NCCL header missing: {source}"
assert (source / "lib" / "libnccl.so.2").is_file(), f"NCCL library missing: {source}"
target.mkdir(parents=True, exist_ok=True)
(target / "lib").mkdir(exist_ok=True)
links = {target / "include": source / "include"}
links.update({target / "lib" / p.name: p for p in (source / "lib").glob("libnccl.so*")})
links[target / "lib" / "libnccl.so"] = source / "lib" / "libnccl.so.2"
for link, destination in links.items():
    if link.is_symlink():
        assert link.resolve() == destination.resolve(), f"Unexpected existing link: {link}"
    elif link.exists():
        raise RuntimeError(f"Refusing to replace existing file: {link}")
    else:
        link.symlink_to(destination)
print("NCCL_HOME:", target)
