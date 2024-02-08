import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def apply_vectorization(data, content_header, vocabulary=None,
                        percentiles=(5, 80), ngrams=(1, 2),
                        save_vocab_as=None):
    """
    Aplica o processo de vetorização, convertendo uma coleção de texto para uma matriz de contagem de tokens, tal que:
        * Cada linha da matriz é um documento da coleção.
        * Cada coluna da matriz é um token do vocabulário.
        * O vocabulário nada mais é do que um array de tokens que aparecem nos textos, tal que esses tokens não se repetem no vocabulário.
        * O vocabulário depende do parâmetro 'ngrams', que especifica quantos termos terá cada token.
    -----
    'term_frequency' é a matriz de contagem de tokens.
        O elemento MxN diz que o documento M possui (n) vezes o token N, tal que (n) é um inteiro. Ou seja, um matriz (Documento x Token).
    'features' é o vocabulário de tokens.
    """

    content = data[content_header].copy()
    min_frequency = int(data.shape[0] * percentiles[0] / 100)
    max_frequency = int(data.shape[0] * percentiles[1] / 100)

    vectorizer = CountVectorizer(max_features=None, vocabulary=vocabulary,
                                 min_df=max(min_frequency, 1), max_df=max_frequency,
                                 ngram_range=ngrams, lowercase=False,
                                 tokenizer=(lambda x: x))

    term_frequency = vectorizer.fit_transform(content)
    features = np.array(vectorizer.get_feature_names_out())

    if save_vocab_as:
        path = f'models/{save_vocab_as}.csv'
        features_list = pd.DataFrame({'word': features.tolist()})
        features_list.to_csv(path, index=False)

    return term_frequency, features
