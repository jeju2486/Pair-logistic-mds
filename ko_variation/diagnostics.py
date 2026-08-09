from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable, Mapping


BLAS_ENVIRONMENT_VARIABLES = (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def guard_blas_environment(default_threads: int = 1) -> dict[str, str]:
    """Set conservative native-thread defaults before NumPy/SciPy import.

    Explicit user settings are preserved. The returned values are suitable for
    later reporting, but the loaded-library inspection in
    :func:`collect_parallel_runtime` is the authoritative runtime view.
    """

    if default_threads < 1:
        raise ValueError("default_threads must be positive")
    default = str(default_threads)
    configured: dict[str, str] = {}
    for variable in BLAS_ENVIRONMENT_VARIABLES:
        if not os.environ.get(variable, "").strip():
            os.environ[variable] = default
        configured[variable] = os.environ[variable]
    return configured


@dataclass(frozen=True)
class NativeLibrary:
    user_api: str
    internal_api: str
    prefix: str
    version: str
    num_threads: int

    @property
    def label(self) -> str:
        implementation = self.internal_api or self.prefix or "unknown"
        version = self.version or "unknown"
        return f"{self.user_api}:{implementation}:{version}:{self.num_threads}"


@dataclass(frozen=True)
class ParallelRuntime:
    worker_processes: int
    logical_cpus: int
    effective_blas_threads: int
    estimated_native_threads: int
    oversubscribed: bool
    libraries: tuple[NativeLibrary, ...]
    environment: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]

    def summary_fields(self) -> dict[str, Any]:
        return {
            "runtime_cpu_count": self.logical_cpus,
            "runtime_worker_processes": self.worker_processes,
            "runtime_blas_threads": self.effective_blas_threads,
            "runtime_estimated_native_threads": self.estimated_native_threads,
            "runtime_blas_oversubscribed": int(self.oversubscribed),
            "runtime_blas_libraries": ",".join(library.label for library in self.libraries) or "unresolved",
            "runtime_blas_environment": ",".join(f"{key}={value}" for key, value in self.environment),
            "runtime_warnings": ",".join(self.warnings) or "none",
        }


def _positive_environment_threads(environment: Mapping[str, str]) -> list[int]:
    values: list[int] = []
    for variable in BLAS_ENVIRONMENT_VARIABLES:
        raw = environment.get(variable, "").strip()
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


def assess_parallel_runtime(
    worker_processes: int,
    *,
    logical_cpus: int | None,
    libraries: Iterable[NativeLibrary],
    environment: Mapping[str, str],
) -> ParallelRuntime:
    """Combine process and native-thread settings into actionable diagnostics."""

    if worker_processes < 1:
        raise ValueError("worker_processes must be positive")

    library_tuple = tuple(libraries)
    blas_threads = [
        library.num_threads
        for library in library_tuple
        if library.user_api == "blas" and library.num_threads > 0
    ]
    thread_source = blas_threads or _positive_environment_threads(environment)
    effective_blas_threads = max(thread_source, default=1)
    estimated_native_threads = worker_processes * effective_blas_threads
    cpus = max(1, int(logical_cpus or 1))
    oversubscribed = estimated_native_threads > cpus

    warnings: list[str] = []
    if worker_processes > 1 and effective_blas_threads > 1:
        warnings.append("nested_blas_parallelism")
    if oversubscribed:
        warnings.append("native_thread_oversubscription")

    return ParallelRuntime(
        worker_processes=worker_processes,
        logical_cpus=cpus,
        effective_blas_threads=effective_blas_threads,
        estimated_native_threads=estimated_native_threads,
        oversubscribed=oversubscribed,
        libraries=library_tuple,
        environment=tuple((variable, environment.get(variable, "unset")) for variable in BLAS_ENVIRONMENT_VARIABLES),
        warnings=tuple(warnings),
    )


def collect_parallel_runtime(worker_processes: int) -> ParallelRuntime:
    """Inspect loaded BLAS/OpenMP libraries and the process-pool configuration."""

    loaded: list[NativeLibrary] = []
    try:
        from threadpoolctl import threadpool_info

        for information in threadpool_info():
            num_threads = information.get("num_threads")
            if not isinstance(num_threads, int):
                continue
            loaded.append(
                NativeLibrary(
                    user_api=str(information.get("user_api") or "unknown"),
                    internal_api=str(information.get("internal_api") or "unknown"),
                    prefix=str(information.get("prefix") or "unknown"),
                    version=str(information.get("version") or "unknown"),
                    num_threads=num_threads,
                )
            )
    except (ImportError, RuntimeError):
        # Environment reporting remains useful if library introspection is not
        # available in an unusual installation.
        pass

    return assess_parallel_runtime(
        worker_processes,
        logical_cpus=os.cpu_count(),
        libraries=loaded,
        environment=os.environ,
    )
