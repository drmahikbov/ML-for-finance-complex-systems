# jfb-for-implicit-oc

Python package for implicit-network optimal control (JFB / Full-AD).

## Documentation

| Document | Contents |
| -------- | -------- |
| [../README.md](../README.md) | Setup and the reference liquidation simulation |
| [../ORIGINAL-AUTHORS.md](../ORIGINAL-AUTHORS.md) | Upstream ICML JFB codebase (Gelphman et al.) — citation, quadcopter / bicycle / consumption examples |
| [../DEVELOPERS.md](../DEVELOPERS.md) | Refactored architecture and recipe for new problems |

## Quick start (ICML examples)

From this directory:

```bash
python examples/example_multibicycle.py
python examples/example_multi_quadcopter.py
python examples/example_multiConsumption.py
```

See [ORIGINAL-AUTHORS.md](../ORIGINAL-AUTHORS.md) for full usage, CLI flags, and BibTeX citations.

## Quick start (this project)

```bash
python examples/explicit_ustar/plot_liquidation_jfb.py --help
```

The flagship 20-asset stochastic run is documented at the end of the root [README.md](../README.md).
