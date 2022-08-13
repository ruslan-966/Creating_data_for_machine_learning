import pandas as pd
import numpy as np
import re
import csv
from model_param import *
from SQL_config import *
import psycopg2
import pymorphy2
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer

pd.options.display.max_colwidth = 100
pd.options.mode.chained_assignment = None  # default='warn'
morph = pymorphy2.MorphAnalyzer()


def take_bd_column(col_name):
    tuples_list = []
    try:
        connection = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=db_name,
            port=port
        )
        connection.autocommit = True
        # Получаем из базы данных содержимое колонки
        # для этого, Запрашиваем количество записей в таблице:
        with connection.cursor() as cursor:
            cursor.execute(f'SELECT {col_name} FROM {TABLE_NAME}')
            tuples_list = cursor.fetchall()
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        if connection:
            connection.close()
    return tuples_list


def x_and_y_creator():
    # Получаем первую колонку из базы данных
    data = take_bd_column("product_description")
    # переводим ее в Дата Фрейм:
    ds = pd.DataFrame(data)
    ds['name'] = ds[0].apply(take_name)
    ds['name'] = ds['name'].str.lower()     # Перевели в нижний регистр

    # print(len(ds['name'].unique()))         # Вывели количество уникальных значений
    a = ds['name'].value_counts()           # Посчитали и вывели количество уникальных вхождений
    # print(a)                                # Приняли, в качестве мусора: количество вхождений
                                            # уникальных значений которых будет меньше 13. Сформировали список.
    wast = []
    for i, el in zip(a.index, a.to_numpy()):
        if el <= 13:
            wast.append(i)
    wast.append('sim')      # Добавляем в мусор явные значения, которые не являются производителями телефонов
    wast.append('fhd')
    m = [v in wast for v in ds['name']]
    ds.loc[m, 'name'] = 'not found'

    print('Количество производителей телефонов: ', len(ds['name'].unique())-1)  # Вывели количество уникальных значений
    # a = ds['name'].value_counts()  # Посчитали и вывели количество уникальных вхождений, отсортировав по алфавиту
    # a_df = pd.DataFrame(a)
    # print(a_df.sort_index(axis=0))
    # удаляем из данных строки с мусором:
    print('Удалили строки с производителями, которрые не определились:')
    ds_clean = ds.loc[ds['name'] != 'not found']
    # print(ds_clean)
    # print('*' * 30)
    print('Перевели буквы в нижний регистр:')
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.lower())
    # print(ds_clean)
    # print('*' * 30)
    print('Удалили производителей из текста:')
    ds_clean[0] = ds_clean[0].replace('|'.join(ds_clean['name']), '', regex=True).str.strip()  # !!!!!!!!!!!!!!!!
    # print(ds_clean)
    # print('*' * 30)
    print(f'Удалили [, ], /, :, ., -')
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace('[', ''))
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace(']', ''))
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace('/', ' '))
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace('.', ''))
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace(':', ''))
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace('-', ' '))
    # print(ds_clean)
    # print('*' * 30)
    print('Заменили в десятичных числах comma, на dot:')
    ds_clean[0] = ds_clean[0].apply(from_comma_to_dot)
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace(',', ''))
    # print(ds_clean)
    # print('*' * 30)

    # Исправляем косяки перевода и кривых пальцев
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace('128gb256gb', '128gb 256gb'))
    ds_clean.loc[:, 0] = ds_clean.loc[:, 0].map(lambda x: x.replace('2340x1080p', '2340x1080'))
    ds_clean[0] = ds_clean[0].replace('|'.join(['amh', 'мач']), 'mah', regex=True)
    ds_clean[0] = ds_clean[0].apply(number_with_mah)
    ds_clean[0] = ds_clean[0].replace('4gbram', '4gb', regex=True)
    ds_clean[0] = ds_clean[0].replace('ram', '', regex=True)

    ds_clean[0] = ds_clean[0].apply(number_with_wt)
    ds_clean[0] = ds_clean[0].apply(number_with_w)
    ds_clean[0] = ds_clean[0].replace(r'(\d)вт', '\1w', regex=True)

    ds_clean[0] = ds_clean[0].apply(number_with_g_rus)
    ds_clean[0] = ds_clean[0].apply(number_with_g_ing)
    ds_clean[0] = ds_clean[0].replace(r"(\d)гб[\s,]", "\1gb.", regex=True)
    ds_clean[0] = ds_clean[0].replace(r'(\d)г\s', "\1gb ", regex=True)
    ds_clean[0] = ds_clean[0].replace(r'(\d)g\s', "\1gb ", regex=True)

    ds_clean[0] = ds_clean[0].apply(number_with_mp_ing)
    ds_clean[0] = ds_clean[0].apply(number_with_mp_rus)
    ds_clean[0] = ds_clean[0].apply(number_with_p_ing)
    ds_clean[0] = ds_clean[0].apply(number_with_p_rus)
    ds_clean[0] = ds_clean[0].replace(r'(\d)п', "\1mp", regex=True)
    ds_clean[0] = ds_clean[0].replace(r'(\d)мп', "\1mp", regex=True)
    ds_clean[0] = ds_clean[0].replace(r'(\d)p', "\1mp", regex=True)

    ds_clean[0] = ds_clean[0].replace(r'(\d)мгц', "\1mhz", regex=True)
    ds_clean[0] = ds_clean[0].apply(kill_all_spec_symbols)
    ds_clean[0] = ds_clean[0].apply(kill_short_words)
    print('Убрали косяки:')
    # print(ds_clean[0])
    # print('*' * 30)
    # Нормализуем слова:
    print('Создаем дополнительную колонку и заносим в нее список слов:')
    ds_clean[2] = ds_clean[0].apply(sentence_into_words)
    # print(ds_clean[2])
    # print('*' * 30)
    print('Переводим слова в нормальную форму:')
    ds_clean[2] = ds_clean[2].apply(morphological_analysis)
    # print(ds_clean[2])
    # print('*' * 30)
    if DEBUGGING_MODE:
        print('Отладочный режим включен:')
        print('Создаем единый список встретившихся слов для его очистки:')
        common_list = ds_clean[2].to_list()
        common_list = list(chain.from_iterable(common_list))
        # print(common_list)
        print('Размер списка до очистки от дубликатов:', len(common_list))
        common_list = list(set(common_list))
        print('Размер списка после очистки от дубликатов:', len(common_list))
        print('Отсортированный список:')
        common_list.sort()
        print(common_list)
        print('*' * 30)
        # выводим в excel, что бы понять, как сократить дубликаты
        df_list = pd.DataFrame(common_list)
        df_list.to_excel('common_list.xlsx')
    # Преобразовываем слова в векторное представление:
    # print(ds_clean)
    print('Перевели слова в векторное представление, для этого:')
    print('Перевели данные из датафрейма в лист:')
    document = ds_clean[2].to_list()
    # print(document)
    # print('*' * 30)
    # Преобразуем список, содержащий списки слов предложенией в список предложений
    lst = []
    for l_ in document:
        str_ = ' '.join(l_)
        lst.append(str_)
    print('Трансформировали список из слов в список из предложениий:')
    # print(lst)
    # print('*' * 30)
    # Create a Vectorizer Object
    vectorizer = CountVectorizer()
    vectorizer.fit(lst)
    # Printing the identified Unique words along with their indices
    print('#' * 50)
    dict_ = vectorizer.vocabulary_
    print("Vocabulary: ", dict_)
    # Encode the Document
    vector = vectorizer.transform(lst)
    # Summarizing the Encoded Texts
    print('%' * 50)
    print("Encoded Document is:")
    print(len(vector.toarray()))
    print(vector.toarray())
    print('-' * 50)
    # Перечень производителей телефонов:
    lst_creator = ds_clean['name'].to_list()
    print('Длина списка производителей телефонов не преобразованная:', len(lst_creator))
    values, counts = np.unique(lst_creator, return_counts=True)
    print(values)
    print(counts)
    with open('unique_names_of_manufacturers.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(values)
        # write.writerows(rows)
        # Планируем применить LR, ответы Y нам нужны в виде: [0, 1]
    # Попоробуем предсказать производителя xiaomi против всех остальных производителей.
    # Присвоим  xiaomi значение 1, остальным - 0, для этого создадим еще один столбец
    ds_clean['Y'] = pd.Series(np.zeros(len(ds_clean), int), index=ds_clean.index)
    ds_clean.loc[ds_clean['name'] == 'xiaomi', 'Y'] = 1
    # Сохраняем данные в два массива:
    np.savetxt(FILE_COUNT_X, vector.toarray(), delimiter=",")
    ds_clean['Y'].to_csv(FILE_COUNT_Y)
    return


def kill_short_words(string):
    return re.sub(r"\b\w{1,2}\b", "", string)


def kill_all_spec_symbols(string):
    return re.sub(r"[^а-яa-z\d\s]+", '', string)


def number_with_p_ing(string):
    return re.sub(r"(\d)\sp", r"\1mp ", string)


def number_with_p_rus(string):
    return re.sub(r"(\d)\sп", r"\1mp ", string)


def number_with_mp_ing(string):
    return re.sub(r"(\d)\smp", r"\1mp ", string)


def number_with_mp_rus(string):
    return re.sub(r"(\d)\sмп", r"\1mp ", string)


def number_with_g_rus(string):
    return re.sub(r"(\d)\sг", r"\1gb ", string)


def number_with_g_ing(string):
    return re.sub(r"(\d)\sg", r"\1gb ", string)


def number_with_wt(string):
    return re.sub(r"(\d)\sвт", r"\1w ", string)


def number_with_w(string):
    return re.sub(r"(\d)\sw", r"\1w ", string)


def number_with_mah(string):
    return re.sub(r"(\d)\s(mah)", r"\1\2", string)


def morphological_analysis(list_string):
    global morph
    words = []
    for word in list_string:
        p = morph.parse(word)[0]  # делаем разбор
        words.append(p.normal_form)
    return words


def sentence_into_words(string):
    return re.findall(r'[a-zа-я\d.]+', string)


def number_with_designation(string):
    return re.sub(r"(\d)\s([a-zа-я])", r"\1\2", string)


def from_comma_to_dot(string):
    return re.sub(r"(\d),(\d)", r"\1.\2", string)


def take_name(string):
    match = re.search(r'\b[a-zA-Z]+\b', string)
    return match[0] if match else 'Not found'


def main():
    x_and_y_creator()


if __name__ == "__main__":
    main()
