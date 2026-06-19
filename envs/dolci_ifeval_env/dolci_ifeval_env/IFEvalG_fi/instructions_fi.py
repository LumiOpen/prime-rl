# Finnish-adapted instruction checkers for IFEvalG.
#
# Every checker class overrides only build_description() with a Finnish
# description string. All check_following() logic is inherited unchanged from
# the English parent classes so scoring behaviour is identical.
#
# Strategy: call super().build_description(**kwargs) to let the parent set all
# self._xxx instance variables (thresholds, relations, keywords, …), then
# replace self._description_pattern and return the Finnish string.

import random
import re

import langdetect
from absl import logging

from ifeval_env.IFEvalG import instructions
from ifeval_env.IFEvalG.instructions import (
    Instruction,
    _COMPARISON_RELATION,
    _POSTSCRIPT_MARKER,
    _NUM_HIGHLIGHTED_SECTIONS,
    _LETTER_FREQUENCY,
)

# ---------------------------------------------------------------------------
# Finnish-specific constants
# ---------------------------------------------------------------------------

_CONSTRAINED_RESPONSE_OPTIONS_FI = (
    "Vastaukseni on kyllä.",
    "Vastaukseni on ei.",
    "Vastaukseni on ehkä.",
)

_STARTER_OPTIONS_FI = (
    "Sanoisin",
    "Vastaukseni on",
    "Uskon",
    "Mielestäni",
    "Minusta",
    "Arvelisin",
    "Minulla on tunne",
    "Näkökulmastani",
    "Näen asian niin",
    "Käsitykseni mukaan",
    "Minun kannaltani",
    "Ymmärrykseni mukaan",
    "Näkemykseni on",
    "Kantani on",
    "Havaintoni mukaan",
)

_ENDING_OPTIONS_FI = (
    "Onko sinulla muita kysymyksiä?",
    "Voinko auttaa sinua jollakin muulla?",
)

_PHRASES_FI = [
    "Tanssi kuin kukaan ei katsoisi",
    "Aikainen lintu madon saa",
    "Aika kuluu hauskaa pitäessä",
    "Jokaisella pilvellä on hopeareunus",
    "Teot puhuvat sanoja kovempaa",
    "Älä tuomitse kirjaa kannen perusteella",
    "Elä jokainen päivä täysillä",
    "Kaikki se mikä kiiltää ei ole kultaa",
    "Nauru on parasta lääkettä",
    "Kynä on miekkaa mightier",
]

# Finnish letters beyond ASCII a-z
_FI_LOWERCASE = "a-zäöå"
_FI_ALL = "a-zA-ZäöåÄÖÅ"

# Comparison relation display translations (stored kwargs stay in English)
_RELATION_FI = {
    "less than": "alle",
    "at least": "vähintään",
}

# Finnish language names keyed by ISO 639-1 code
_LANGUAGE_NAMES_FI = {
    "af": "afrikaans", "ar": "arabia", "bg": "bulgaria", "bn": "bengali",
    "ca": "katalaani", "cs": "tšekki", "cy": "kymri", "da": "tanska",
    "de": "saksa", "el": "kreikka", "en": "englanti", "eo": "esperanto",
    "es": "espanja", "et": "viro", "eu": "baski", "fa": "persia",
    "fi": "suomi", "fr": "ranska", "ga": "iiri", "gl": "galicia",
    "gu": "gudžarati", "he": "heprea", "hi": "hindi", "hr": "kroatia",
    "hu": "unkari", "hy": "armenia", "id": "indonesia", "is": "islanti",
    "it": "italia", "ja": "japani", "ka": "georgia", "kn": "kannada",
    "ko": "korea", "lt": "liettua", "lv": "latvia", "mk": "makedonia",
    "ml": "malajalam", "mr": "marathi", "ms": "malaiji", "mt": "malta",
    "nl": "hollanti", "no": "norja", "pl": "puola", "pt": "portugali",
    "ro": "romania", "ru": "venäjä", "sk": "slovakki", "sl": "sloveeni",
    "sq": "albania", "sr": "serbia", "sv": "ruotsi", "sw": "swahili",
    "ta": "tamili", "te": "telugu", "th": "thai", "tl": "tagalog",
    "tr": "turkki", "uk": "ukraina", "ur": "urdu", "vi": "vietnam",
    "zh": "kiina", "zh-cn": "kiina (yksinkertaistettu)",
    "zh-tw": "kiina (perinteinen)",
}


def _fi_lang(code: str) -> str:
    """Return the Finnish name for a language code, falling back to the code itself."""
    return _LANGUAGE_NAMES_FI.get(code, code)


def _rel(relation: str) -> str:
    """Return Finnish display form of a comparison relation."""
    return _RELATION_FI.get(relation, relation)


# ---------------------------------------------------------------------------
# Already-overridden checkers (kept from the original file)
# ---------------------------------------------------------------------------

class CapitalLettersFinnishChecker(Instruction):
    """Checks that the response is in Finnish and in all capital letters."""

    def build_description(self):
        self._description_pattern = "Koko vastauksesi tulee olla suomeksi ja kokonaan isoilla kirjaimilla."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        assert isinstance(value, str)
        try:
            return value.isupper() and langdetect.detect(value) == "fi"
        except langdetect.LangDetectException as e:
            logging.error("Unable to detect language for text %s due to %s", value, e)
            return True


class LowercaseLettersFinnishChecker(Instruction):
    """Checks that the response is in Finnish and in all lowercase letters."""

    def build_description(self):
        self._description_pattern = (
            "Koko vastauksesi tulee olla suomeksi ja kokonaan pienillä kirjaimilla. "
            "Isoja kirjaimia ei sallita."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        assert isinstance(value, str)
        try:
            return value.islower() and langdetect.detect(value) == "fi"
        except langdetect.LangDetectException as e:
            logging.error("Unable to detect language for text %s due to %s", value, e)
            return True


class ConstrainedResponseFinnishChecker(Instruction):
    """Checks the constrained response (Finnish options)."""

    def build_description(self):
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS_FI
        self._description_pattern = "Vastaa yhdellä seuraavista vaihtoehdoista: {response_options}"
        return self._description_pattern.format(response_options=self._constrained_responses)

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        value = value.strip()
        return any(option in value for option in self._constrained_responses)


class ConstrainedStartFinnishChecker(Instruction):
    """Checks the response start (Finnish starters)."""

    def build_description(self, *, starter=None):
        self._starter = starter.strip() if isinstance(starter, str) else starter
        if self._starter is None:
            self._starter = random.choice(_STARTER_OPTIONS_FI)
        self._description_pattern = (
            "Kun on sinun vuorosi keskustelussa, aloita aina lausumallasi {starter}"
        )
        return self._description_pattern.format(starter=self._starter)

    def get_instruction_args(self):
        return {"starter": self._starter}

    def get_instruction_args_keys(self):
        return ["starter"]

    def check_following(self, value):
        response_pattern = r"^\s*" + re.escape(self._starter) + r".*$"
        return bool(re.search(response_pattern, value, flags=re.MULTILINE))


class EndFinnishChecker(Instruction):
    """Checks that the response ends with a given phrase (Finnish defaults)."""

    def build_description(self, *, end_phrase=None):
        self._end_phrase = end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS_FI)
        self._description_pattern = (
            "Päätä vastauksesi tähän tarkkaan lauseeseen {ender}. "
            "Tämän lauseen jälkeen ei saa olla muita sanoja."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        return ["end_phrase"]

    def check_following(self, value):
        value = value.strip().strip('"').lower()
        return value.endswith(self._end_phrase.strip().lower())


class RepeatPhraseFinnishChecker(Instruction):
    """Repeat the phrase N times with a one-word variation (Finnish phrases)."""

    def build_description(self, phrase=None, small_n=None):
        if not phrase:
            self._phrase = random.choice(_PHRASES_FI)
        else:
            self._phrase = phrase.strip()
        if not small_n:
            self._small_n = random.randint(2, 3)
        else:
            self._small_n = small_n

        self._description_pattern = (
            "Toista lause {phrase} täsmälleen {small_n} kertaa, muuttaen sitä hieman "
            "joka kerta korvaamalla vain yksi sana lauseen keskeltä."
        )
        return self._description_pattern.format(phrase=self._phrase, small_n=self._small_n)

    def get_instruction_args(self):
        return {"phrase": self._phrase, "small_n": self._small_n}

    def get_instruction_args_keys(self):
        return ["phrase", "small_n"]

    def check_following(self, value):
        first_word = self._phrase.split()[0]
        last_word = self._phrase.split()[-1]
        found_phrases = re.findall(rf"{re.escape(first_word)} .*? {re.escape(last_word)}", value)
        if len(found_phrases) != self._small_n:
            return False
        num_satisfied = 0
        for phrase in found_phrases:
            phrase_words = phrase.split()
            ref_words = self._phrase.split()
            if len(phrase_words) != len(ref_words):
                return False
            differences = sum(a != b for a, b in zip(phrase_words, ref_words))
            if differences > 1:
                return False
            num_satisfied += differences == 1
        return num_satisfied == self._small_n


class LowercaseCountingFinnishChecker(Instruction):
    """Lowercase word count — extended regex covers Finnish letters ä, ö, å."""

    def build_description(self, N=None):
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        self._description_pattern = (
            "Vastauksessasi kaikkien pienten kirjainten sanojen tulee esiintyä enintään {N} kertaa."
        )
        return self._description_pattern.format(N=self._N)

    def get_instruction_args(self):
        return {"N": self._N}

    def get_instruction_args_keys(self):
        return ["N"]

    def check_following(self, value):
        lowercase_words = re.findall(r"\b[" + _FI_LOWERCASE + r"]+\b", value)
        return len(lowercase_words) <= self._N


class LetterCountingFinnishChecker(Instruction):
    """Letter count — extended regex covers Finnish diacritics ä, ö, å."""

    def build_description(self, N=None, relation=None):
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        if not relation:
            self._relation = random.choice(_COMPARISON_RELATION)
        else:
            self._relation = relation
        self._description_pattern = "Vastaa {relation} {N} kirjaimella."
        return self._description_pattern.format(N=self._N, relation=_rel(self._relation))

    def get_instruction_args(self):
        return {"N": self._N, "relation": self._relation}

    def get_instruction_args_keys(self):
        return ["N", "relation"]

    def check_following(self, value):
        letters = re.findall(r"[" + _FI_ALL + r"]", value)
        if self._relation == "at least":
            return len(letters) >= self._N
        elif self._relation == "less than":
            return len(letters) < self._N


# ---------------------------------------------------------------------------
# New Finnish checker overrides — only build_description is changed
# ---------------------------------------------------------------------------

class ResponseLanguageFinnishChecker(instructions.ResponseLanguageChecker):
    def build_description(self, *, language=None):
        super().build_description(language=language)
        lang_name = _fi_lang(self._language)
        self._description_pattern = (
            "Koko vastauksesi tulee olla {language}-kielellä. Muita kieliä ei sallita."
        )
        return self._description_pattern.format(language=lang_name)


class NumberOfSentencesFinnishChecker(instructions.NumberOfSentences):
    def build_description(self, *, num_sentences=None, relation=None):
        super().build_description(num_sentences=num_sentences, relation=relation)
        self._description_pattern = "Vastauksesi tulee sisältää {relation} {num_sentences} lausetta."
        return self._description_pattern.format(
            relation=_rel(self._comparison_relation),
            num_sentences=self._num_sentences_threshold,
        )


class PlaceholderFinnishChecker(instructions.PlaceholderChecker):
    def build_description(self, *, num_placeholders=None):
        super().build_description(num_placeholders=num_placeholders)
        self._description_pattern = (
            "Vastauksen tulee sisältää vähintään {num_placeholders} paikkamerkkiä "
            "hakasulkeissa, kuten [osoite]."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)


class BulletListFinnishChecker(instructions.BulletListChecker):
    def build_description(self, *, num_bullets=None):
        super().build_description(num_bullets=num_bullets)
        self._description_pattern = (
            "Vastauksesi tulee sisältää täsmälleen {num_bullets} luetelmapistettä. "
            "Käytä markdown-luettelomerkkejä kuten:\n"
            "* Tämä on kohta 1. \n"
            "* Tämä on kohta 2"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)


class HighlightSectionFinnishChecker(instructions.HighlightSectionChecker):
    def build_description(self, *, num_highlights=None):
        super().build_description(num_highlights=num_highlights)
        self._description_pattern = (
            "Korosta vähintään {num_highlights} osiota vastauksessasi markdownilla, "
            "eli *korostettu osio*."
        )
        return self._description_pattern.format(num_highlights=self._num_highlights)


class SectionFinnishChecker(instructions.SectionChecker):
    def build_description(self, *, section_spliter=None, num_sections=None):
        super().build_description(section_spliter=section_spliter, num_sections=num_sections)
        self._description_pattern = (
            "Vastauksessasi tulee olla {num_sections} osiota. Merkitse jokaisen osion alku "
            "merkinnällä {section_spliter} X, esimerkiksi:\n"
            "{section_spliter} 1\n"
            "[osion 1 sisältö]\n"
            "{section_spliter} 2\n"
            "[osion 2 sisältö]"
        )
        return self._description_pattern.format(
            num_sections=self._num_sections,
            section_spliter=self._section_spliter,
        )


class ParagraphFinnishChecker(instructions.ParagraphChecker):
    def build_description(self, *, num_paragraphs=None):
        super().build_description(num_paragraphs=num_paragraphs)
        self._description_pattern = (
            "Vastauksessa tulee olla {num_paragraphs} kappaletta. "
            "Kappaleet erotetaan toisistaan markdown-erottimella: ***"
        )
        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)


class PostscriptFinnishChecker(instructions.PostscriptChecker):
    def build_description(self, *, postscript_marker=None):
        super().build_description(postscript_marker=postscript_marker)
        self._description_pattern = (
            "Lisää vastauksesi loppuun jälkikirjoitus, joka alkaa merkinnällä {postscript}"
        )
        return self._description_pattern.format(postscript=self._postscript_marker)


class JsonFormatFinnishChecker(instructions.JsonFormat):
    def build_description(self):
        self._description_pattern = (
            "Koko tuloste tulee olla JSON-muodossa. Voit käyttää markdown-koodilohkoja kuten ```."
        )
        return self._description_pattern


class ParagraphFirstWordFinnishChecker(instructions.ParagraphFirstWordCheck):
    def build_description(self, num_paragraphs=None, nth_paragraph=None, first_word=None):
        super().build_description(
            num_paragraphs=num_paragraphs,
            nth_paragraph=nth_paragraph,
            first_word=first_word,
        )
        self._description_pattern = (
            "Vastauksessa tulee olla {num_paragraphs} kappaletta. "
            "Kappaleet erotetaan toisistaan kahdella rivinvaihdolla ('\\n\\n' pythonissa). "
            "Kappaleen {nth_paragraph} tulee alkaa sanalla {first_word}."
        )
        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )


class TwoResponsesFinnishChecker(instructions.TwoResponsesChecker):
    def build_description(self):
        self._description_pattern = (
            "Anna kaksi erilaista vastausta. Vastaukset ja vain vastaukset erotetaan "
            "toisistaan kuudella tähtimerkillä: ******."
        )
        return self._description_pattern


class RepeatPromptThenAnswerFinnishChecker(instructions.RepeatPromptThenAnswer):
    def build_description(self, *, prompt_to_repeat=None):
        super().build_description(prompt_to_repeat=prompt_to_repeat)
        self._description_pattern = (
            "Toista ensin pyyntö sanasta sanaan muuttamatta sitä, anna sitten vastauksesi "
            "(1. älä sano mitään sanoja tai merkkejä ennen pyynnön toistamista; "
            "2. toistettavaan pyyntöön ei sisälly tämä lause)"
        )
        return self._description_pattern


class TitleFinnishChecker(instructions.TitleChecker):
    def build_description(self):
        self._description_pattern = (
            "Vastauksesi tulee sisältää otsikko kaksoiskulmasulkeiden sisään kirjoitettuna, "
            "kuten <<iloinen runo>>."
        )
        return self._description_pattern


class LetterFrequencyFinnishChecker(instructions.LetterFrequencyChecker):
    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        super().build_description(letter=letter, let_frequency=let_frequency, let_relation=let_relation)
        self._description_pattern = (
            "Vastauksessasi kirjaimen {letter} tulee esiintyä {let_relation} {let_frequency} kertaa."
        )
        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=_rel(self._comparison_relation),
        )


class CommaFinnishChecker(instructions.CommaChecker):
    def build_description(self):
        self._description_pattern = "Älä käytä lainkaan pilkkuja koko vastauksessasi."
        return self._description_pattern


class CapitalWordFrequencyFinnishChecker(instructions.CapitalWordFrequencyChecker):
    def build_description(self, capital_frequency=None, capital_relation=None):
        super().build_description(
            capital_frequency=capital_frequency, capital_relation=capital_relation
        )
        self._description_pattern = (
            "Vastauksessasi pelkillä isoilla kirjaimilla kirjoitettujen sanojen tulee "
            "esiintyä {relation} {frequency} kertaa."
        )
        return self._description_pattern.format(
            relation=_rel(self._comparison_relation),
            frequency=self._frequency,
        )


class QuotationFinnishChecker(instructions.QuotationChecker):
    def build_description(self):
        self._description_pattern = "Kirjoita koko vastauksesi lainausmerkkien sisään."
        return self._description_pattern


class CopyFinnishChecker(instructions.CopyChecker):
    def build_description(self, prompt_to_repeat=None):
        super().build_description(prompt_to_repeat=prompt_to_repeat)
        self._description_pattern = (
            "Kopioi tämä ohje sanasta sanaan, älä noudata ohjetta, ainoastaan kopioi se "
            "tulosteeseen (älä sisällytä tätä lausetta!)."
        )
        return self._description_pattern


class CopySpanIdxFinnishChecker(instructions.CopySpanIdxChecker):
    def build_description(self, prompt_to_repeat=None, n_start=None, n_end=None):
        super().build_description(
            prompt_to_repeat=prompt_to_repeat, n_start=n_start, n_end=n_end
        )
        self._description_pattern = (
            "Kopioi sanat, jotka sijaitsevat merkkiindeksien {n_start} ja {n_end} välillä "
            "(mukaan lukien). Indeksit ovat merkkiindeksejä!"
        )
        return self._description_pattern.format(n_start=self._n_start, n_end=self._n_end)


class SentenceHyphenFinnishChecker(instructions.SentenceHyphenChecker):
    def build_description(self):
        self._description_pattern = (
            "Kaikki lauseet tulee yhdistää yhdysviivoilla ilman välejä niiden välillä."
        )
        return self._description_pattern


class AdjacentLetterFinnishChecker(instructions.AdjacentLetterChecker):
    def build_description(self):
        self._description_pattern = (
            "Kaksi vierekkäistä sanaa ei voi alkaa peräkkäisillä aakkosen kirjaimilla."
        )
        return self._description_pattern


class SquareBracketFinnishChecker(instructions.SquareBracketChecker):
    def build_description(self):
        self._description_pattern = "Kirjoita jokainen sana vastauksessasi hakasulkeiden sisään."
        return self._description_pattern


class KeywordFrequencyOnceFinnishChecker(instructions.KeywordFrequencyOnceChecker):
    def build_description(self, *, keyword=None):
        super().build_description(keyword=keyword)
        self._description_pattern = "Sisällytä avainsana {keyword} vastaukseesi."
        return self._description_pattern.format(keyword=self._keyword)


class KeywordFrequencyDifferentFinnishChecker(instructions.KeywordFrequencyCheckerDifferent):
    def build_description(self, *, keyword=None, frequency=None, relation=None):
        super().build_description(keyword=keyword, frequency=frequency, relation=relation)
        self._description_pattern = (
            "Vastauksessasi sanan {keyword} tulee esiintyä {frequency} kertaa."
        )
        return self._description_pattern.format(
            keyword=self._keyword, frequency=self._frequency
        )


class ExcludeWordHarderFinnishChecker(instructions.ExcludeWordHarderChecker):
    def build_description(self, keyword=None, instruction=None):
        super().build_description(keyword=keyword, instruction=instruction)
        self._description_pattern = "Älä sisällytä avainsanaa {keyword} vastaukseen."
        return self._description_pattern.format(keyword=self._keyword)


class ParagraphBasicFinnishChecker(instructions.ParagraphBasicChecker):
    def build_description(self):
        self._description_pattern = (
            "Vastauksessa tulee olla 2 kappaletta. "
            "Kappaleet erotetaan toisistaan markdown-erottimella: ***"
        )
        return self._description_pattern


class ParagraphBasic2FinnishChecker(instructions.ParagraphBasicChecker2):
    def build_description(self):
        self._description_pattern = (
            "Vastauksessa tulee olla 2 kappaletta. "
            "Kappaleet erotetaan toisistaan kahdella rivinvaihdolla."
        )
        return self._description_pattern


class FirstWordSentFinnishChecker(instructions.FirstWordSentChecker):
    def build_description(self, first_word=None):
        super().build_description(first_word=first_word)
        self._description_pattern = (
            "Jokaisen lauseen ensimmäisen sanan tulee olla sana {first_word}."
        )
        return self._description_pattern.format(first_word=self._first_word)


class FirstWordAnswerFinnishChecker(instructions.FirstWordAnswerChecker):
    def build_description(self, first_word=None):
        super().build_description(first_word=first_word)
        self._description_pattern = (
            "Vastauksesi ensimmäisen sanan tulee olla sana {first_word}."
        )
        return self._description_pattern.format(first_word=self._first_word)


class LastWordSentFinnishChecker(instructions.LastWordSentChecker):
    def build_description(self, last_word=None):
        super().build_description(last_word=last_word)
        self._description_pattern = (
            "Jokaisen lauseen viimeisen sanan tulee olla, ennen välimerkkiä, sana {last_word}."
        )
        return self._description_pattern.format(last_word=self._last_word)


class LastWordAnswerFinnishChecker(instructions.LastWordAnswerChecker):
    def build_description(self, last_word=None):
        super().build_description(last_word=last_word)
        self._description_pattern = (
            "Vastauksesi viimeisen sanan tulee olla sana {last_word}."
        )
        return self._description_pattern.format(last_word=self._last_word)


class BiGramWrappingFinnishChecker(instructions.BiGramWrappingChecker):
    def build_description(self):
        self._description_pattern = (
            "Kirjoita jokainen sanapari kaksoiskulmasulkeiden sisään, "
            "kuten <<olen kotona>> <<minun koirani>> <<on söpö>>."
        )
        return self._description_pattern


class CopyingSimpleFinnishChecker(instructions.CopyingSimpleChecker):
    def build_description(self, prompt_to_repeat=None):
        super().build_description(prompt_to_repeat=prompt_to_repeat)
        self._description_pattern = (
            "Toista pyyntö muuttamatta sitä (älä sano mitään ennen pyynnön toistamista; "
            "toistettavaan pyyntöön ei sisälly tämä lause) äläkä vastaa itse pyyntöön!"
        )
        return self._description_pattern


class CopyingMultipleFinnishChecker(instructions.CopyingMultipleChecker):
    def build_description(self, prompt_to_repeat=None, N=None):
        super().build_description(prompt_to_repeat=prompt_to_repeat, N=N)
        self._description_pattern = (
            "Toista pyyntö muuttamatta sitä {N} kertaa, eroteltuina kuudella tähtimerkillä "
            "(älä sano mitään ennen pyynnön toistamista; toistettavaan pyyntöön ei sisälly "
            "tämä lause) äläkä vastaa itse pyyntöön!"
        )
        return self._description_pattern.format(N=self._N)


class PunctuationDotFinnishChecker(instructions.PunctuationDotChecker):
    def build_description(self):
        self._description_pattern = (
            "Älä käytä pistettä (.) välimerkkeinä tai muutenkaan koko vastauksessasi."
        )
        return self._description_pattern


class PunctuationExclamationFinnishChecker(instructions.PunctuationExclamationChecker):
    def build_description(self):
        self._description_pattern = (
            "Älä käytä huutomerkkiä (!) välimerkkeinä tai muutenkaan koko vastauksessasi."
        )
        return self._description_pattern


class CountingCompositionFinnishChecker(instructions.CountingCompositionChecker):
    def build_description(self, n_sent=None, n_words=None):
        super().build_description(n_sent=n_sent, n_words=n_words)
        self._description_pattern = (
            "Kirjoita 3 kappaletta, jotka on erotettu markdown-erottimella: * * *, "
            "joissa on täsmälleen {n_sent} lausetta kussakin ja täsmälleen {n_words} "
            "sanaa jokaisessa lauseessa."
        )
        return self._description_pattern.format(n_sent=self._n_sent, n_words=self._n_words)


class CountUniqueFinnishChecker(instructions.CountUniqueChecker):
    def build_description(self):
        self._description_pattern = (
            "Käytä vastauksessasi vain yksilöllisiä sanoja, yhtäkään sanaa ei saa toistaa!"
        )
        return self._description_pattern


class CountIncrementWordFinnishChecker(instructions.CountIncrementWordChecker):
    def build_description(self, keyword1=None, keyword2=None):
        super().build_description(keyword1=keyword1, keyword2=keyword2)
        self._description_pattern = (
            "Sisällytä avainsana {keyword1} kerran vastaukseesi ja "
            "avainsana {keyword2} kahdesti vastaukseesi."
        )
        return self._description_pattern.format(
            keyword1=self._keyword1, keyword2=self._keyword2
        )


class PalindromeBasicFinnishChecker(instructions.PalindromeBasicChecker):
    def build_description(self):
        self._description_pattern = "Sisällytä palindromi vastaukseesi."
        return self._description_pattern


class KeywordSpecificPositionFinnishChecker(instructions.KeywordSpecificPositionChecker):
    def build_description(self, keyword=None, n=None, m=None):
        super().build_description(keyword=keyword, n=n, m=m)
        self._description_pattern = (
            "Sisällytä avainsana {keyword} {n}. lauseeseen sen {m}. sanaksi."
        )
        return self._description_pattern.format(
            keyword=self._keyword, n=self._n, m=self._m
        )


class StartEndFinnishChecker(instructions.StartEndChecker):
    def build_description(self):
        self._description_pattern = (
            "Aloita ja lopeta vastauksesi samalla sanalla "
            "(älä kirjoita mitään viimeisen sanan jälkeen, ei edes välimerkkiä)."
        )
        return self._description_pattern


class KeywordFinnishChecker(instructions.KeywordChecker):
    def build_description(self, *, keywords=None):
        super().build_description(keywords=keywords)
        self._description_pattern = "Sisällytä avainsanat {keywords} vastaukseen."
        return self._description_pattern.format(keywords=self._keywords)


class KeywordFrequencyFinnishChecker(instructions.KeywordFrequencyChecker):
    def build_description(self, *, keyword=None, frequency=None, relation=None):
        super().build_description(keyword=keyword, frequency=frequency, relation=relation)
        self._description_pattern = (
            "Vastauksessasi sanan {keyword} tulee esiintyä {relation} {frequency} kertaa."
        )
        return self._description_pattern.format(
            keyword=self._keyword,
            relation=_rel(self._comparison_relation),
            frequency=self._frequency,
        )


class ForbiddenWordsFinnishChecker(instructions.ForbiddenWords):
    def build_description(self, forbidden_words=None):
        super().build_description(forbidden_words=forbidden_words)
        self._description_pattern = "Älä sisällytä avainsanoja {forbidden_words} vastaukseen."
        return self._description_pattern.format(forbidden_words=self._forbidden_words)


class NumberOfWordsFinnishChecker(instructions.NumberOfWords):
    def build_description(self, *, num_words=None, relation=None):
        super().build_description(num_words=num_words, relation=relation)
        self._description_pattern = "Vastaa {relation} {num_words} sanalla."
        return self._description_pattern.format(
            relation=_rel(self._comparison_relation),
            num_words=self._num_words,
        )
