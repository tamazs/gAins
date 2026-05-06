"""
test_text_splitter.py — Unit tests for tools/text_splitter.py

split_text is a pure function — no mocking needed.
"""

import pytest
from tools.text_splitter import split_text


class TestSplitText:
    def test_empty_string_returns_empty_list(self):
        assert split_text("") == []

    def test_text_shorter_than_chunk_size_returns_single_chunk(self):
        result = split_text("Hello world", chunk_size=500)
        assert result == ["Hello world"]

    def test_text_exactly_chunk_size_returns_single_chunk(self):
        text = "a" * 500
        result = split_text(text, chunk_size=500, overlap=0)
        assert len(result) == 1
        assert result[0] == text

    def test_text_longer_than_chunk_size_returns_multiple_chunks(self):
        text = "a" * 1000
        result = split_text(text, chunk_size=500, overlap=0)
        # step = 500-0 = 500; chunks: [0..499], [500..999]
        assert len(result) == 2

    def test_overlap_causes_content_repetition_at_boundary(self):
        # chunk 0: 500 "a"s,  chunk 1 starts at offset 450 → first 50 chars are "a"
        text = "a" * 500 + "b" * 500
        result = split_text(text, chunk_size=500, overlap=50)
        # chunk[1] starts at pos 450, so it begins with 50 "a"s
        assert result[1].startswith("a" * 50)

    def test_chunk_count_correct_for_known_input(self):
        # 1050 chars, chunk_size=500, overlap=50 → step=450
        # chunk 0: [0..499], chunk 1: [450..949], chunk 2: [900..1049]
        text = "x" * 1050
        result = split_text(text, chunk_size=500, overlap=50)
        assert len(result) == 3

    def test_strips_whitespace_from_chunk_edges(self):
        # Build a text where the first chunk is padded with trailing spaces
        text = "hello" + " " * 495 + "world" * 100
        result = split_text(text, chunk_size=500)
        for chunk in result:
            assert chunk == chunk.strip()

    def test_custom_chunk_size_respected(self):
        text = "a" * 200
        result = split_text(text, chunk_size=100, overlap=0)
        assert len(result) == 2
        assert all(len(c) <= 100 for c in result)

    def test_custom_overlap_respected(self):
        # chunk_size=100, overlap=20 → step=80
        # chunk 0: [0..99]  (100 "a"s)
        # chunk 1: [80..179] (20 "a"s then 80 "b"s)
        text = "a" * 100 + "b" * 100
        result = split_text(text, chunk_size=100, overlap=20)
        assert result[1].startswith("a" * 20)

    def test_no_empty_chunks_in_output(self):
        # A very short text followed by lots of whitespace could produce
        # empty stripped chunks — they must be filtered out.
        text = "hello" + " " * 500
        result = split_text(text, chunk_size=10, overlap=0)
        for chunk in result:
            assert chunk != ""

    def test_single_character_text(self):
        result = split_text("z", chunk_size=500)
        assert result == ["z"]
