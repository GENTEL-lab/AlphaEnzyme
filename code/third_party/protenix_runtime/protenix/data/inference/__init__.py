"""Inference helpers with lazy imports.

The original upstream module eagerly imported ``infer_dataloader``, which in turn
pulls the old ESM precompute stack at import time. In this bundle we also need
the newer ``esm==3`` package for residue-level ESMC extraction, so eager import
would fail even when the caller only needs JSON parsing or structure export.

Keep the public API unchanged, but resolve symbols lazily so non-ESM inference
utilities can run without importing the old ESM stack.
"""

from importlib import import_module

__all__ = [
    "get_inference_dataloader",
    "cif_to_input_json",
    "merge_covalent_bonds",
    "add_entity_atom_array",
    "lig_file_to_atom_info",
    "remove_leaving_atoms",
    "SampleDictToFeatures",
]


def __getattr__(name):
    if name == "get_inference_dataloader":
        return import_module(
            ".infer_dataloader", __name__
        ).get_inference_dataloader
    if name in {"cif_to_input_json", "merge_covalent_bonds"}:
        module = import_module(".json_maker", __name__)
        return getattr(module, name)
    if name in {
        "add_entity_atom_array",
        "lig_file_to_atom_info",
        "remove_leaving_atoms",
    }:
        module = import_module(".json_parser", __name__)
        return getattr(module, name)
    if name == "SampleDictToFeatures":
        return import_module(".json_to_feature", __name__).SampleDictToFeatures
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
