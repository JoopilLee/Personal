'''
This example show how to perform a DMR topic model with multi-metadata using tomotopy
'''
import itertools
import pyLDAvis
import tomotopy as tp
import numpy as np
import treform as ptm
import re
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

kiwi = Kiwi()
stopwords = Stopwords()

input_file = '/Users/oiehhun/python/텍스트마이닝/tm_project/tm_data/sejong_연도_분류_본문.txt'

# You can get the sample data file from https://github.com/bab2min/g-dmr/tree/master/data .
corpus = tp.utils.Corpus()

for line in open(input_file, encoding='utf-8'):
    fd = line.strip().split('\t')
    # if len(fd) < 4:
    #     continue

    time_stamp = fd[0]
    section = fd[1]
    text = fd[2]
    filtered_fd = re.sub('[^가-힣- ]', '', text) # 한자 제거
    kiwi_tokens = kiwi.tokenize(filtered_fd, stopwords=stopwords) # 형태소 분석, 불용어 제거
    noun_words = [] # 명사인 단어 추출
    for token in kiwi_tokens: 
        if 'NN' in token.tag:
            noun_words.append(token.form)
    final_noun_words = [] # 1음절 제거
    for word in noun_words:
            if len(word) > 1:
                final_noun_words.append(word)
    corpus.add_doc(final_noun_words, multi_metadata=['y_' + time_stamp, 's_' + section])
# We add prefix 'y' for year-label and 'j' for journal-label

# We set a range of the first metadata as [2000, 2017]
# and one of the second metadata as [0, 1].
mdl = tp.DMRModel(tw=tp.TermWeight.ONE,  # tf-idf 쓰려면
                  k=5,
                  corpus=corpus
                  )
mdl.optim_interval = 20
mdl.burn_in = 200

mdl.train(0)

print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))

# Let's train the model
for i in range(0, 2000, 20): #iteration : 2000은 돼야함
    print('Iteration: {:04} LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04} LL per word: {:.4}'.format(2000, mdl.ll_per_word))

mdl.summary()

year_labels = sorted(l for l in mdl.multi_metadata_dict if l.startswith('y_'))
section_labels = sorted(l for l in mdl.multi_metadata_dict if l.startswith('s_'))

# calculate topic distribution with each metadata using get_topic_prior()
print('Topic distributions by year')
for l in year_labels:
    print(l, '\n', mdl.get_topic_prior(multi_metadata=[l]), '\n')

print('Topic distributions by journal')
for l in section_labels:
    print(l, '\n', mdl.get_topic_prior(multi_metadata=[l]), '\n')

# Also we can estimate topic distributions with multiple metadata
print('Topic distributions by year-journal')
for y, j in itertools.product(year_labels, section_labels):
    print(y, ',', j, '\n', mdl.get_topic_prior(multi_metadata=[y, j]), '\n')

topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq

prepared_data = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency,
    start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
    sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
)
pyLDAvis.save_html(prepared_data, 'ldavis_yeonsan.html')

# 결과를 파일에 저장
with open('king_dmr_ouput.txt', 'w', encoding='utf-8') as f:
    f.write('Num docs:{}, Num Vocabs:{}, Total Words:{}\n'.format(
        len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
    ))

    f.write('Iteration: {:04} LL per word: {:.4}\n'.format(2000, mdl.ll_per_word))

    mdl.summary(file=f)

    f.write('Topic distributions by year\n')
    for l in year_labels:
        f.write('{}\n{}\n'.format(l, mdl.get_topic_prior(multi_metadata=[l])))

    f.write('Topic distributions by journal\n')
    for l in section_labels:
        f.write('{}\n{}\n'.format(l, mdl.get_topic_prior(multi_metadata=[l])))

    f.write('Topic distributions by year-journal\n')
    for y, j in itertools.product(year_labels, section_labels):
        f.write('{}, {}\n{}\n'.format(y, j, mdl.get_topic_prior(multi_metadata=[y, j])))