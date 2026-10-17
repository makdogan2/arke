import unittest
from data_download import clean, strip_reference_sections

BODY_TR = ("Ankara, Türkiye'nin başkentidir. Şehir, İç Anadolu Bölgesi'nde yer alır.\n"
           "Nüfusu yaklaşık beş milyondur ve ülkenin ikinci büyük şehridir.")
BODY_EN = ("Ankara is the capital of Turkey. It is located in Central Anatolia.\n"
           "With a population of about five million, it is the second largest city.")


class TestStripReferenceSections(unittest.TestCase):
    def test_cuts_at_kaynakca(self):
        text = BODY_TR + "\n\nKaynakça\n\nDış bağlantılar\n Resmî site"
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


if __name__ == "__main__":
    unittest.main()
