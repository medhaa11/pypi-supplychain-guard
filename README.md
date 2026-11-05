# pypi-supplychain-guard

Detects likely malicious PyPI packages in pull requests, with explanations.

## Safety rules
- Never pip install, import, or run any package from the malicious dataset.
- Analyze source archives as text/AST only.
- Never commit downloaded samples (data/ is gitignored).

## Status
Phase 1: data collection (in progress)
