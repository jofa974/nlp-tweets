from src.prepare import cleanup


def test_cleanup():
    text = "'By accident' they knew what was gon happen https://t.co/Ysxun5vCeh"
    result = cleanup(text)

    assert "http" not in result
