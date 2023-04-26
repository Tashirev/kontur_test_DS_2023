import re
import pymorphy2
from nltk.corpus import stopwords
import torch
from catboost import Pool
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

morph = pymorphy2.MorphAnalyzer()
stopwords_ru = set(stopwords.words('russian'))
stopwords_ru.add('')
pool_kernel_size = 40
maxpool = torch.nn.MaxPool1d(pool_kernel_size)

class KonturTest():

    def __init__(self, config_words, models_catboost, tokenizer, model_embedding):
        super(KonturTest, self).__init__()
        self.config_words = config_words
        self.models_catboost = models_catboost
        self.tokenizer = tokenizer
        self.model_embedding = model_embedding


    # функция получает текст,
    # возвращает список лемматизированных слов (без стоп-слов)
    def lemmatize(self, text):

        words_list = []
        words = re.split(r'(%)', text)

        for part in words:
            words_list.extend(re.split(r'[)|(|:|.|-| |,]', part))
        lemmatize_word_list = list()

        for word in words_list:
            # приведение к нормальной форме
            normal_word = morph.parse(word)[0].normal_form
            # проверка на число
            try:
                normal_word = float(normal_word)
                normal_word = 'число значение'
            except:
                None

            # исключение стоп-слов
            if normal_word not in stopwords_ru:
                lemmatize_word_list.append(normal_word)

        if len(lemmatize_word_list) == 0:
            lemmatize_word_list.append('пустота')

        return lemmatize_word_list


    # Функция принимает список words_lemmatize, длину текста и список из 2-х слов,
    # осуществляет проверку на вхождение этих слов одновременно в words_lemmatize
    # и проверку длины текста,
    # возвращает predict (пустое список или длина текста < 1600), либо None
    def check_empty_fragment(self, words_lemmatize, words, text_len):

        # слова не входят или длина текста < 1700, вернуть predict
        if (len(set(words_lemmatize) & set(words)) == 0) or text_len < 1600:
            return {'text': [''], 'answer_start': [0], 'answer_end': [0]}
        else:  # входит хотябы одно слово, вернуть None
            return None


    # Функция принимает text и список регулярных фрагментов,
    # осуществляет проверку на вхождение данных фрагментов,
    # возвращает predict (фрагмент и координаты), либо None
    def check_regular_fragment(self, text, parts):

        for part in parts:
            first_index = text.find(part)
            if first_index != -1:
                return {'text': [part], 'answer_start': [first_index], 'answer_end': [first_index + len(part)]}

        return None  # не входит ни один фрагмент, вернуть None



    # Функция принимает текст и список слов,
    # возвращает датафрейм (слова и список номеров всех вхождений для каждого слова)
    def words_places_in_text(self, text, words):
        words_places_dict = {'word': [], 'places': []}
        for word in words:
            places = [_.start() for _ in re.finditer(word, text)]
            words_places_dict['word'].append(word)
            words_places_dict['places'].append(places)
        df = pd.DataFrame(words_places_dict)
        df.loc[df.shape[0]] = ['text_len', [len(text)]]
        return df



    # Функция созадет признаки на основании вхождения ключевых слов в text,
    # уплотняет признаковое пространство методом maxpool.
    # возвращает numpy вектор.
    def create_feature_for_places(self, text, words_for_places, words_for_places_removed):

        # 1 пункт
        # датафрейм ключевых слов с позициями в тексте.
        # удаляется последняя строка датафрейма с информацией о длине текста ('text_len')
        df_words_places_in_text = self.words_places_in_text(text, words_for_places)[:-1]

        # 2 пункт
        # датафрейм подложка заполненная 0, размер (число символов в тексте, число ключевых слов)
        df_words_places_in_text_new = pd.DataFrame(
            np.zeros([len(text), df_words_places_in_text['word'].shape[0]], dtype=int),
            columns=df_words_places_in_text['word'])
        # заполнение датафрейма ключевыми словами в месте их вхождения
        for word in words_for_places:
            index_word_list = df_words_places_in_text.query(f'word == @word').values[0][1]
            for index in index_word_list:
                df_words_places_in_text_new.loc[index, word] = 1

        # обрезка размера столбцов под область обнаружения строки (область определена из анализа)
        df_words_places_in_text_new = df_words_places_in_text_new[1000:1600]

        # Объединение колонок синонимов "контракт+договор" (при наличии), "руб+%+проц"
        if 'договор' in words_for_places:
            df_words_places_in_text_new['контракт'] = df_words_places_in_text_new['контракт'] + \
                                                      df_words_places_in_text_new['договор']
            df_words_places_in_text_new.drop(columns=['договор'], inplace=True)
        df_words_places_in_text_new['руб'] = df_words_places_in_text_new['руб'] + df_words_places_in_text_new['%'] + \
                                             df_words_places_in_text_new['проц']
        df_words_places_in_text_new.drop(columns=['%', 'проц'], inplace=True)

        # 3 пункт
        # words_for_places_removed_words = words_for_places
        tensor = torch.tensor(df_words_places_in_text_new[words_for_places_removed].values, dtype=float).T

        return torch.flatten(maxpool(tensor)).numpy()


    # создание списка вероятных фрагментов
    def create_parts(self, text, answer_start_near, answer_end_near, start_words, end_words, answer_start_delta, answer_end_delta):

        # получение большой подстроки, с добавлением дельт
        part_with_delta = text[answer_start_near - answer_start_delta: answer_end_near + answer_end_delta]

        # разделить фрагмент пополам
        half = int(len(part_with_delta) / 2)
        part_with_delta_start_half = part_with_delta[:half]
        part_with_delta_end_half = part_with_delta[half:]

        # генерация подстрок с границами из наиболее частых начальных и конечных слов
        index = 0
        part_start_list = []
        while (index < start_words.shape[0]):

            word_index_list = [_.start() for _ in
                               re.finditer(re.escape(start_words.index[index]), part_with_delta_start_half)]
            for word_index in word_index_list:
                part_start_list.append(part_with_delta_start_half[word_index:])
            index += 1

        index = 0
        part_end_list = []
        while (index < end_words.shape[0]):

            word_index_list = [_.start() for _ in
                               re.finditer(re.escape(end_words.index[index]), part_with_delta_end_half)]
            for word_index in word_index_list:
                part_end_list.append(part_with_delta_end_half[:word_index] + end_words.index[index])
            index += 1

        # попарная склейка подстрок (start+end), получение набора вариантов фрагмента
        # порядок склейки задан списком пар, для сохранения более вероятных (start_words, end_words)
        order = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 0], [1, 2], [2, 1], [2, 2]]
        part_list = []
        for i, j in order:
            try:  # применяется для отработки исключений по выходу индекса за предел списка, в случае коротких списков
                part_list.append(part_start_list[i] + part_end_list[j])
            except:
                part_list.append(f'Отсутствует достаточное количество {i}{j} фрагментов')

        # устранение повторений в списке, при повторении заменить 'Отсутствует достаточное количество (add_word) фрагментов'
        add_word = 0
        while len(set(part_list)) < len(order):
            part_list = list(set(part_list))
            part_list.append(f'Отсутствует достаточное количество {add_word} фрагментов')
            add_word += 1

        return part_list


    # создание ембеддинга фрагмента
    def create_embedding(self,text):
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model_embedding(**{k: v.to(self.model_embedding.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()



    # Функция принимает строку датафрейма,
    # Для каждого фрагмента из списка создаёт эмбеддинг,
    # по тексту и эмбеддингу предсказывается вероятность для фрагмента,
    # выбирается самый вероятный фрагмент функцией argmax(),
    # возвращается словарь (фрагмент, старт_фрагмента, конец_фрагмента)
    def predict_from_catboost_binary_class(self,text, part_list, model_classifier):

        predict_proba_list = []

        for part in part_list:
            embedding = self.create_embedding(part)
            data = pd.DataFrame({'part': part, 'embedding': [embedding]})
            pool = Pool(data=data,
                        embedding_features=['embedding'],
                        text_features=['part'])

            # добавление в список вероятности класса 1.
            predict_proba_list.append(model_classifier.predict_proba(pool)[0][1])

        # получение фрагмента из списка с максимальным значением predict_proba (по argmax)
        part = part_list[np.array(predict_proba_list).argmax()]
        first_index = text.find(part)

        return {'text': [part], 'answer_start': [first_index], 'answer_end': [first_index + len(part)]}

    # В качестве предскказания берётся нулевой элемент списка,
    # как ограниченный самыми часто повторяющимися словами
    # возвращается словарь (фрагмент, старт_фрагмента, конец_фрагмента)
    def predict_from_list_top(self, text, part_list):
        part = part_list[0]
        first_index = text.find(part)

        return {'text': [part], 'answer_start': [first_index], 'answer_end': [first_index + len(part)]}


    # функция предсказания, получает на вход текст документа и вид искомой информации
    def predict(self, text, label):
        words_lemmatize = self.lemmatize(text)
        text_len = len(text)
        text_lower = text.lower()

        if label == 'обеспечение исполнения контракта': # label == 0
            control_words = self.config_words['control_words_0']
            regular_parts = self.config_words['regular_parts_0']
            words_for_places = self.config_words['words_for_places_0']
            words_for_places_removed = self.config_words['words_for_places_removed_0']
            model_for_places_end = self.models_catboost['model_for_places_0_end']
            model_for_places_start = self.models_catboost['model_for_places_0_start']
            start_words = self.config_words['start_words_0']
            end_words = self.config_words['end_words_0']
        elif label == 'обеспечение гарантийных обязательств':  # label == 1
            control_words = self.config_words['control_words_1']
            regular_parts = self.config_words['regular_parts_1']
            words_for_places = self.config_words['words_for_places_1']
            words_for_places_removed = self.config_words['words_for_places_removed_1']
            model_for_places_end = self.models_catboost['model_for_places_1_end']
            model_for_places_start = self.models_catboost['model_for_places_1_start']
            start_words = self.config_words['start_words_1']
            end_words = self.config_words['end_words_1']
        else:
            raise Exception("Label error")

        # предсказание отсутствия фрагмента
        predict = self.check_empty_fragment(words_lemmatize, control_words, text_len)
        if predict != None:
            return predict

        # предсказание регулярного фрагмента
        predict = self.check_regular_fragment(text, regular_parts)
        if predict != None:
            return predict

        # примерные границы фрагмента
        features_for_places = self.create_feature_for_places(text_lower,words_for_places,words_for_places_removed)
        answer_start_near = model_for_places_start.predict(np.concatenate((np.array([text_len]),features_for_places))).astype(int)
        answer_end_near = model_for_places_end.predict(np.concatenate((np.array([text_len]),features_for_places))).astype(int)

        # список предпологаемых фрагментов
        part_list = self.create_parts(text, answer_start_near, answer_end_near, start_words, end_words, 40, 40)

        # Предсказания фрагментов в зависимости от label
        if label == 'обеспечение исполнения контракта': # label == 0
            predict = self.predict_from_catboost_binary_class(text,
                                                              part_list,
                                                              self.models_catboost['model_classifier_binary_class_0']
                                                              )
        else:  # label == 1
            predict = self.predict_from_list_top(text, part_list)

        return predict