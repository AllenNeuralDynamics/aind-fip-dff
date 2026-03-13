"""Entry point for python -m aind_fip_dff"""

if __name__ == "__main__":
    import os
    import sys
    import runpy

    # Read version from the package
    from aind_fip_dff import __version__

    # Set default environment variables if not already set
    os.environ.setdefault("VERSION", __version__)
    os.environ.setdefault(
        "DFF_EXTRACTION_URL", "https://github.com/AllenNeuralDynamics/aind-fip-dff"
    )

    # Always add --serial flag if not already present
    if "--serial" not in sys.argv:
        sys.argv.append("--serial")

    # Run the main module
    runpy.run_module("aind_fip_dff.run_capsule", run_name="__main__")
