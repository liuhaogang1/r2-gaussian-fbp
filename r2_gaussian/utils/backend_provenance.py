"""Source/build identities. Importing this module never imports torch or CUDA."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime, timezone

SOURCE_SUFFIXES = {".py", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".inl"}
EXCLUDED_DIRS = {"build", "dist", "__pycache__", ".git", ".pytest_cache"}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(backend):
    backend = Path(backend).resolve()
    repo = backend.parents[2]
    directories = [backend]
    if backend.name.startswith("self-"):
        directories.append(backend.parent / "xray-gaussian-rasterization-voxelization" / "third_party" / "glm")
    files = set()
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing native build input directory: {directory}")
        files.update(path for path in directory.rglob("*") if path.is_file()
                     and not EXCLUDED_DIRS.intersection(path.relative_to(directory).parts)
                     and not any(part.endswith(".egg-info") for part in path.parts)
                     and path.suffix.lower() in SOURCE_SUFFIXES)
    inventory = [{"path": path.relative_to(repo).as_posix(), "sha256": file_digest(path)} for path in files]
    inventory.append({"path": "r2_gaussian/utils/backend_provenance.py", "sha256": file_digest(__file__)})
    return sorted(inventory, key=lambda item: item["path"])


def source_fingerprint(backend):
    return _inventory_fingerprint(source_inventory(backend))


def _inventory_fingerprint(inventory):
    value = json.dumps({"fingerprint_version": 1, "files": inventory}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_record_path(binary):
    binary = Path(binary)
    return binary.with_name(binary.name + ".build.json")


def validate_extension_build(backend, extension):
    backend = Path(backend).resolve()
    binary = Path(extension.__file__).resolve()
    if backend not in binary.parents:
        raise RuntimeError(f"CUDA extension loaded from another checkout: {binary}")
    expected = source_fingerprint(backend)
    compiled = getattr(extension, "build_source_hash", None)
    if compiled != expected:
        raise RuntimeError(f"CUDA source/build mismatch: {binary}. Rebuild in {backend} with "
                           "python setup.py build_ext --inplace, then restart the process.")
    record_path = build_record_path(binary)
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Missing/invalid CUDA build record: {record_path}; rebuild the extension") from error
    binary_hash = file_digest(binary)
    if (not isinstance(record, dict) or record.get("format_version") != 1
            or record.get("source_hash") != expected or record.get("binary_sha256") != binary_hash):
        raise RuntimeError(f"CUDA build record does not match source/binary: {record_path}; rebuild the extension")
    return {"extension_path": str(binary), "binary_sha256": binary_hash,
            "source_hash": expected, "build_record_path": str(record_path),
            "build_record_sha256": file_digest(record_path), "build": record}


def _command_version(command):
    if not command:
        return None
    try:
        process = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=10)
        return {"command": command, "exit_code": process.returncode,
                "output": (process.stdout + process.stderr)[-6000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def provenance_build_ext(base_class, backend):
    """Wrap torch BuildExtension; write records only after a successful build."""
    backend = Path(backend).resolve()

    class ProvenanceBuildExtension(base_class):
        def run(self):
            source_files = source_inventory(backend)
            fingerprint = _inventory_fingerprint(source_files)
            # Wrapper/header-only changes must also update the compiled hash.
            # Ninja still handles unchanged compile commands incrementally.
            self.force = True
            for extension in self.extensions:
                extension.define_macros = [item for item in extension.define_macros
                                           if item[0] != "R2G_BUILD_SOURCE_HASH"]
                # Pass a single bare hexadecimal token. Windows' Ninja/MSVC
                # argument handling strips quotes embedded in -D values.
                # ext.cpp stringizes this token after preprocessing.
                extension.define_macros.append(("R2G_BUILD_SOURCE_HASH", fingerprint))
            super().run()
            if source_fingerprint(backend) != fingerprint:
                raise RuntimeError("Native sources changed during compilation; rebuild from a stable checkout")
            torch_module = sys.modules.get("torch")
            compiler = getattr(self, "compiler", None)
            compiler_command = getattr(compiler, "cc", None)
            if isinstance(compiler_command, str):
                compiler_command = [compiler_command, "/Bv"] if os.name == "nt" else [compiler_command, "--version"]
            else:
                command = getattr(compiler, "compiler", None)
                compiler_command = [command[0], "--version"] if isinstance(command, (list, tuple)) and command else None
            from torch.utils.cpp_extension import CUDA_HOME
            nvcc = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc") if CUDA_HOME else None
            for extension in self.extensions:
                binary = Path(self.get_ext_fullpath(extension.name)).resolve()
                record = {"format_version": 1, "source_hash": fingerprint,
                          "source_files": source_files, "binary_name": binary.name,
                          "binary_sha256": file_digest(binary), "extension_name": extension.name,
                          "built_at_utc": datetime.now(timezone.utc).isoformat(),
                          "python": platform.python_version(), "python_abi": sysconfig.get_config_var("SOABI"),
                          "torch": getattr(torch_module, "__version__", None),
                          "torch_cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
                          "host_compiler": _command_version(compiler_command),
                          "nvcc": _command_version([str(nvcc), "--version"]) if nvcc else None,
                          "compile_args": extension.extra_compile_args,
                          "define_macros": extension.define_macros,
                          "environment": {key: os.environ.get(key) for key in (
                              "CC", "CXX", "CFLAGS", "CXXFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                              "TORCH_CUDA_ARCH_LIST", "CUDA_HOME")}}
                descriptor, temporary = tempfile.mkstemp(dir=binary.parent, suffix=".json")
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
                    os.replace(temporary, build_record_path(binary))
                finally:
                    Path(temporary).unlink(missing_ok=True)
    return ProvenanceBuildExtension
