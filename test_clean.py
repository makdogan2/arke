import unittest
from textclean import EOT, clean, is_repetitive, reject_reason, strip_reference_sections

BODY_TR = ("Ankara, Türkiye'nin başkentidir. Şehir, İç Anadolu Bölgesi'nde yer alır.\n"
           "Nüfusu yaklaşık beş milyondur ve ülkenin ikinci büyük şehridir.")
BODY_EN = ("Ankara is the capital of Turkey. It is located in Central Anatolia.\n"
           "With a population of about five million, it is the second largest city.")


class TestStripReferenceSections(unittest.TestCase):
    def test_cuts_at_kaynakca(self):
        text = BODY_TR + "\n\nKaynakça\n\nDış bağlantılar\n Resmî site"
        self.assertEqual(strip_reference_sections(text).strip(), BODY_TR)

    def test_cuts_at_kaynaklar(self):
        text = BODY_TR + "\n\nKaynaklar\n\nYılmaz, A. (1999). Ankara Tarihi."
        self.assertEqual(strip_reference_sections(text).strip(), BODY_TR)

    def test_cuts_at_references(self):
        text = BODY_EN + "\n\n  References  \n\nSmith, J. (2001). A History of Ankara."
        self.assertEqual(strip_reference_sections(text).strip(), BODY_EN)

    def test_no_heading_unchanged(self):
        self.assertEqual(strip_reference_sections(BODY_EN), BODY_EN)

    def test_word_in_sentence_not_cut(self):
        text = "The book contains many references to the city.\n" + BODY_EN
        self.assertEqual(strip_reference_sections(text), text)

    def test_clean_strips_reference_section(self):
        text = BODY_TR + "\n\nKaynakça\nKaynak bir\nKaynak iki"
        self.assertEqual(clean(text), BODY_TR)


class TestIsRepetitive(unittest.TestCase):
    def test_normal_text_kept(self):
        self.assertFalse(is_repetitive(BODY_TR + "\n" + BODY_EN))

    def test_repeated_short_lines(self):
        # 3 of 6 lines are copies -> 50% > 30%
        text = "\n".join([BODY_EN] + ["Share", "Read more"] * 2 + ["Share"])
        self.assertTrue(is_repetitive(text))

    def test_one_long_paragraph_copied(self):
        # only 1 of 3 lines is a copy (33%... just above 30%), but it's ~half of the characters
        para = "This paragraph was pasted twice by the site template, which happens a lot on the web."
        self.assertTrue(is_repetitive("Short intro.\n" + para + "\n" + para))

    def test_blank_lines_ignored(self):
        self.assertFalse(is_repetitive(BODY_TR + "\n\n\n\n" + BODY_EN))

    def test_empty(self):
        self.assertFalse(is_repetitive(""))


class TestRejectReason(unittest.TestCase):
    def test_short(self):
        self.assertEqual(reject_reason("Kısa bir sayfa."), "short")

    def test_kept(self):
        # long enough (> MIN_CHARS) and every line is different
        text = "\n".join(f"Part {n}: " + BODY_EN.replace("\n", " ") for n in ("one", "two", "three", "four"))
        self.assertIsNone(reject_reason(text))

    def test_copied_text_rejected(self):
        self.assertEqual(reject_reason((BODY_EN + "\n") * 4), "repetitive")

    def test_clean_removes_eot(self):
        self.assertNotIn(EOT, clean("Bir" + EOT + "iki"))


if __name__ == "__main__":
    unittest.main()
