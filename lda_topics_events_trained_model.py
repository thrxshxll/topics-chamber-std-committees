import joblib
import pandas as pd

import toolkit

DIR = f'lda_topics_events_trained_model'

METADATA_COLUMNS = ['event_id', 'event_type', 'committee', 'date', 'month']

NGRAMS = (1, 2)

print('Reading data...')
data = pd.read_csv('data/raw/speeches_per_event_2008_2019.csv')

print('\nReading model...')
model = joblib.load('models/lda-standing-committees.joblib')
vocabulary = pd.read_csv('models/lda-features.csv')['word'].tolist()

print('\nReading stopwords...')
stopwords = pd.read_csv('data/util/stopwords.csv')
stopwords = stopwords['word'].tolist()

print('\nExtracting metadata...')
data['month'] = data['date'].apply(toolkit.get_month_from_date)
event_metadata = toolkit.extract_metadata(data=data, columns=METADATA_COLUMNS)

print('\nApplying tokenization...')
data = toolkit.apply_tokenization(data=data, content_header='content',
                                  id_header='event_id',
                                  stopwords=stopwords)

print('\nApplying stemming using RSLP...')
data = toolkit.apply_stemming(data=data, content_header='tokens',
                              id_header='event_id')

print('\nApplying vectorization using Term-Frequency...')
tf_matrix, features = toolkit.apply_vectorization(data=data,
                                                  vocabulary=vocabulary,
                                                  content_header='stems',
                                                  ngrams=NGRAMS)

print('\nApplying Latent Dirichlet Allocation model to data...')
identifiers = data['event_id'].tolist()
document_topics = toolkit.apply_lda_model(model=model,
                                          term_frequency=tf_matrix,
                                          id_header='event_id',
                                          identifiers=identifiers)

toolkit.get_topic_main_words(model=model, words=features, save_words_as=f'{DIR}/topic_main_words')

print('\nApplying UMAP for dimension reduction of term frequency...')
toolkit.apply_umap_model(values=tf_matrix, id_header='event_id',
                         identifiers=identifiers, n_components=2,
                         save_as=f'{DIR}/umap-term-frequency')

print('\nApplying UMAP for dimension reduction of topic distribution...')
doc_topic_distribution = document_topics.copy().iloc[:, 2:]
toolkit.apply_umap_model(values=doc_topic_distribution, id_header='event_id',
                         identifiers=identifiers, n_components=2,
                         save_as=f'{DIR}/umap-event-topic-distribution')

save_as = f'data/ready/{DIR}/event-topic-distribution.csv'
document_topics = pd.merge(event_metadata, document_topics, on='event_id')
document_topics.to_csv(save_as, index=False)

print('\nDone!')
