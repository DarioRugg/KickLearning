from os.path import join, exists
from os import mkdir
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
from polyglot.detect import Detector
from transformers.hf_api import HfApi
import torch
import pysbd
from polyglot.detect.base import logger as polyglot_logger
import time
import regex
import sys

polyglot_logger.setLevel("ERROR")


class TextAnalysis:
    def __init__(self, data_path, file_name, print_time=True, batch_size=60, device='auto'):
        self.file_name = file_name
        self.file_path = join(data_path, file_name)
        save_path = data_path.replace('Scraped', 'Translated')
        if not exists(save_path):
            mkdir(save_path)
        self.save_file = join(save_path, filename.replace('_scraped', ''))
        self.html_chars = regex.compile(
            r"\s*You\'ll\s*need\s*an\s*HTML5\s*capable\s*browser\s*to\s*see\s*this\s*content\s*\.\s*(\n\s)*\s*|\s*Play\s*(\n\s)+\s*|/\s*Indicator\s*Bar\s*\d\s*(\n\s)*|/\s*Animation\s*Variables\s*(\n\s)\s*|(Replay|Play)\s*with\s*sound\s*(\n\s)+|\xa0")
        self.df = pd.read_csv(self.file_path)
        self.bad_chars = regex.compile(r"\p{Cc}|\p{Cs}")
        self.df.loc[:, 'story'] = self.df.loc[:, 'story'].apply(self.__remove_html_and_special)
        self.df.loc[:, 'lang'] = self.df['story'].apply(self.detect_lang).apply(self.glob_lang)
        self.__nmt_languages()
        self.time = print_time

        self.batch = batch_size
        if device == 'auto' or device == 'gpu':
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f'using device: {self.device}')
        elif device == 'cpu':
            self.device = device

    def __nmt_languages(self):
        model_list = HfApi().model_list()
        org = "Helsinki-NLP"
        model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
        self.suffix = [x.split('/')[1] for x in model_ids]

    def detect_lang(self, inp):
        return Detector(self.__remove_bad_chars(str(inp)), quiet=True).languages

    def __remove_bad_chars(self, text):
        return self.bad_chars.sub("", text)

    def glob_lang(self, langs):
        lang1, lang2, lang3 = [{'code': x.code, 'conf': x.confidence} for x in langs]
        codes = [lang1['code'], lang2['code'], lang3['code']]
        confs = [lang1['conf'], lang2['conf'], lang3['conf']]
        if lang1['conf'] > 80 and lang2['conf'] <= 10:
            glob_l = lang1['code']
        elif lang2['conf'] > 10:
            glob_l = [lang for i, lang in enumerate(codes) if confs[i] > 10]
            if 'en' in glob_l:
                glob_l = 'multi_en'
            else:
                glob_l = f'multi_{lang1["code"]}'
        else:
            glob_l = 'unknown'
        return glob_l

    def __remove_html_and_special(self, text):
        return self.html_chars.sub("", str(text))

    def __primary_lang_filter(self, text, segmenter, lang):
        segmented = segmenter.segment(str(text))
        return ' '.join([segmented[i] for i, x in enumerate(list(map(self.detect_lang, segmented))) if
                         x[0].code == lang and x[0].confidence >= 80 and x[1].confidence < 5])

    def __catch_time(self, unit='s'):
        if unit == 's':
            return round((time.time() - self.start), 2)
        if unit == 'm':
            return round((time.time() - self.start) / 60, 2)
        if unit == 'h':
            return round((time.time() - self.start) / 3600, 2)
        else:
            raise ValueError(f"Unit must be a single letter between ('s','m','h'). Got {unit} instead")

    def filter_languages(self):
        self.start = time.time()
        multi = {x for x in set(self.df['lang']) if 'multi_' in x}
        for l in multi:
            lang = l.split('_')[1]
            seg = pysbd.Segmenter(language=lang if lang != 'sv' else 'da')
            temp = self.df.loc[self.df['lang'] == l]
            text_list = list(map(str, temp[['story']].to_numpy().flatten()))
            filtered = list(map(lambda x: self.__primary_lang_filter(x, seg, lang), text_list))
            self.df.loc[self.df['lang'] == l, 'story'] = np.array(filtered).reshape(temp['story'].shape)
            self.df.loc[self.df['lang'] == l, 'lang'] = lang
        if self.time:
            print(f"Filtering of other languages done in {self.__catch_time('s')} seconds")

    def translate(self):
        self.filter_languages()

        batch_size = self.batch
        set_l = set(self.df['lang'])
        set_l = set_l.intersection(
            set(map(lambda x: x.split('-')[2] if 'en' in x.split('-')[3:] else None, self.suffix)))
        set_l = set_l.intersection(set(pysbd.languages.LANGUAGE_CODES.keys()))
        set_l = set_l.union({'sv'})

        for l in set_l:
            temp = self.df.loc[self.df['lang'] == l]
            text_list = list(map(str, temp[['story']].to_numpy().flatten()))
            model_name = f'Helsinki-NLP/opus-mt-{l}-en'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            seg = pysbd.Segmenter(language=l if l != 'sv' else 'da')
            translations = []
            for text in text_list:
                inp = seg.segment(text)
                if len(inp) > batch_size:
                    all_decoded = []
                    for batch in np.array_split(inp, np.ceil(len(inp) / batch_size)):
                        tok = tokenizer(batch.tolist(), return_tensors="pt", padding=True).to(self.device)
                        translated = model.generate(**tok)
                        decoded = ' '.join([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
                        all_decoded.append(decoded)
                    translations.append(' '.join(all_decoded))
                else:
                    tok = tokenizer(inp, return_tensors="pt", padding=True).to(self.device)
                    translated = model.generate(**tok)
                    decoded = ' '.join([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
                    translations.append(decoded)
            self.df.loc[self.df['lang'] == l, 'story'] = np.array(translations).reshape(temp['story'].shape)
            del model, tokenizer, translations, decoded, translated, inp, text_list, temp
            torch.cuda.empty_cache()
        if self.time:
            print(f"Total time for the algorithm was {self.__catch_time('m')} minutes")
        self.df.to_csv(self.save_file, index=False)


if __name__ == '__main__':
    filename = sys.argv[1]
    datapath = join('.', 'drive', 'MyDrive', 'Project', 'Data', 'Scraped')
    analyzer = TextAnalysis(data_path=datapath, file_name=filename)
    analyzer.translate()
