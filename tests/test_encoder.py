"""Tests for encoder communication."""

import pytest
from stack.data.encoder import EncoderReading


def test_encoder_reading_to_dict():
    reading = EncoderReading(
        timestamp_ms=1000,
        index_mcp=45.0,
        index_pip=30.0,
        three_finger_mcp=50.0,
        three_finger_pip=25.0,
        local_time=1234567890.0,
    )

    d = reading.to_dict()

    assert d["timestamp_ms"] == 1000
    assert d["index_mcp"] == 45.0
    assert d["index_pip"] == 30.0


def test_encoder_reading_to_array():
    reading = EncoderReading(
        timestamp_ms=1000,
        index_mcp=45.0,
        index_pip=30.0,
        three_finger_mcp=50.0,
        three_finger_pip=25.0,
        local_time=1234567890.0,
    )

    arr = reading.to_array()

    assert arr == [45.0, 30.0, 50.0, 25.0]
    assert len(arr) == 4
