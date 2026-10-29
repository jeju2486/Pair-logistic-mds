"""Narrow restart migration from the released v0.9.0 execution implementation.

Only known source snapshots are accepted. Inputs, scientific settings, fitting
source and NumPy/SciPy versions must still match; no guessed cache compatibility.
"""
from .discovery_store import atomic_json


V090_CODE = {
    "discovery.py": {"58815668cd72808476938323038f40bf9a9857982199b1381d7a3266605227a7", "f7b73f6c0638da2e1667b918ba9bf04c418488c8144ddd141910e3da3d8e3414"},
    "discovery_store.py": {"84669b63f25bb42eff0573404b94c456066de20179322bb25c784e45c4b84364", "3d032490d2c9aeaf083c623cb5bfc7254da629a35cc07c03197761db74f3a358"},
    "discovery_score.py": {"2841afde850491a6bc071bdf5d8060110ccba469e4c0ffb515ce602943fe74df", "ad77b0edae27a4cbaf5aeefb30774f9a06800bb300f295091266bbeb13844193"},
    "null_identity.py": {"df135b2b7513b9029d1f39704d99be47a3e1d54575af49c947a5e88b6de0d4ec", "8498a97a83d407af0be6b218621e2a01d3c4cda72a58b451adba3dc9b7718ea2"},
    "glmm.py": {"ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4", "0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081"},
    "kinship.py": {"35ac6734cd8976956cbcaf5d6b6f91af09a234eca19c5ead7425dae9f7a5eef7", "e08b3b1a7926159b4a6d56b9a3600120df038c363f20934a9ebb89c9c43862c6"},
}


def migrate_identity(previous, expected, manifest):
    if previous == expected:
        return False
    if previous.get("version") != "0.9.0" or set(previous.get("code",{})) != set(V090_CODE):
        raise ValueError("Input/code/statistical settings changed; use a new output")
    if any(previous["code"][key] not in values for key,values in V090_CODE.items()):
        raise ValueError("Previous source is not a verified v0.9.0 release; migration rejected")
    scientific={key:value for key,value in previous.get("settings",{}).items() if key!="batch_size"}
    if scientific != expected["settings"]:
        raise ValueError("Scientific settings changed; migration cannot reuse this traversal")
    for key in ("inputs","numpy","scipy","null_model_source_sha256"):
        if previous.get(key)!=expected.get(key):
            raise ValueError(f"Restart {key} differs; migration rejected")
    backup=manifest.with_name("run_identity.v090.json")
    if not backup.exists():
        atomic_json(backup,previous)
    atomic_json(manifest,expected)
    return True
