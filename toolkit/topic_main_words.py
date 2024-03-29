import pandas as pd


def get_topic_main_words(model, words, save_words_as: str):
    # topic_main_word(model, features)
    """
        model: modelo carregado do joblib, (models/lda-standing-committees.joblib)
        words: é o dicionário de palavras, (models/lda-features.csv)
    """

    topic_main_words = []
    for topic_association in model.components_:
        word_indexes = (-topic_association).argsort()[:10]
        topic_main_words.append(words.take(word_indexes))

    model_parameters = model.get_params()
    n_components = model_parameters['n_components']
    topic_main_words = pd.DataFrame(topic_main_words)
    topic_main_words.columns = [f'word_{i}' for i in range(10)]
    topic_main_words.insert(0, 'topic', [i for i in range(n_components)])

    if save_words_as:
        path = f'data/ready/{save_words_as}.csv'
        topic_main_words.to_csv(path, index=False)