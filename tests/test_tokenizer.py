from aether_v3.data.tokenizer import BLANK_ID, VOCAB_SIZE, byte_ids_to_text, text_to_byte_ids


def test_blank_id_and_vocab_size():
    assert BLANK_ID == 256
    assert VOCAB_SIZE == 257


def test_ascii_roundtrip():
    text = "hello world"
    ids = text_to_byte_ids(text)
    assert ids == [ord(c) for c in text]
    assert byte_ids_to_text(ids) == text


def test_multibyte_utf8_roundtrip():
    text = "Привет, world! 你好"
    ids = text_to_byte_ids(text)
    assert all(0 <= i <= 255 for i in ids)
    assert byte_ids_to_text(ids) == text


def test_empty_string():
    assert text_to_byte_ids("") == []
    assert byte_ids_to_text([]) == ""


def test_invalid_byte_sequence_is_replaced_not_raised():
    # A lone UTF-8 continuation byte (0x80) is invalid on its own.
    text = byte_ids_to_text([0x80, 0x41])
    assert "�" in text
    assert "A" in text
