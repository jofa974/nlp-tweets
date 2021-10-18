from src.preprocessors.abstract import Preprocessor


def test_remove_html():
    text = "<div><h1>COUCOU</h1></div>"
    result = Preprocessor.remove_html(text)
    expected = "COUCOU"
    assert result == expected


def test_remove_emoji():
    text = "Heeyyy \U0001F600 \U0001F300 "
    result = Preprocessor.remove_emoji(text)
    expected = "Heeyyy EMOJI EMOJI "
    assert result == expected


def test_remove_url():
    text = "'By accident' they knew what was gon happen https://t.co/Ysxun5vCeh"
    result = Preprocessor.remove_url(text)
    expected = "'By accident' they knew what was gon happen "
    assert result == expected
