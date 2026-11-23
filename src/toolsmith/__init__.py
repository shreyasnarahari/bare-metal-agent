"""Toolsmith: function calling and constrained decoding for self-hosted models, from first principles.

Main pieces: ``api`` (OpenAI-compatible service), ``grammar`` (JSON Schema to GBNF),
``streaming``/``qwen35`` (native tool-call parsers), ``validation``, ``repair``,
``sampling`` (custom token-mask sampler), ``backends`` (llama.cpp), ``evaluation``,
and the Colab helpers ``hardware``, ``provision``, ``background``, and ``drive``.
"""

__version__ = "0.1.0"  # recorded in run manifests
