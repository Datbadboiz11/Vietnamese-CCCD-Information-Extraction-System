"""
src/pipeline — End-to-end CCCD extraction pipeline.

Exports:
    CCCDPipeline    - pipeline chính
    PipelineResult  - dataclass kết quả
"""
from .pipeline import CCCDPipeline, PipelineResult

__all__ = ["CCCDPipeline", "PipelineResult"]
