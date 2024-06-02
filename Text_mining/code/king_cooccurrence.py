import os
import treform as ptm

#mecab_path='C:\\mecab\\mecab-ko-dic'
# mecab_path='../mecab/mecab-ko-dic-2.1.1-20180720'
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        # ptm.tokenizer.MeCab(mecab_path),
                        ptm.tokenizer.Komoran(),
                        ptm.helper.POSFilter('VV*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='../treform-main/stopwords/stopwordsKor.txt'))

corpus = ptm.CorpusFromFile('왕_txt/king_prepro.txt')
result = pipeline.processCorpus(corpus)

with open('../treform-main/sample_data/processed_전체왕.txt', 'w', encoding='utf-8') as f_out:
    for doc in result:
        for sent in doc:
            new_sent = ''
            for word in sent:
                new_sent += word + ' '
            new_sent = new_sent.strip()
            f_out.write(new_sent + "\n")
f_out.close()

input_file = '../treform-main/sample_data/processed_전체왕.txt'
output_file = '../treform-main/sample_data/co_전체왕.txt'
worker_number = 3
threshold_value = 1
#program_path = 'D:\\python_workspace\\treform\\external_programs\\'
program_path = '../treform-main/external_programs'
if os.path.isdir(program_path) != True:
    raise Exception(program_path + ' is not a directory')


co_occur = ptm.cooccurrence.CooccurrenceExternalManager(program_path=program_path, input_file=input_file,
                                                        output_file=output_file, threshold=threshold_value, num_workers=worker_number)

co_occur.execute()

co_results={}
vocabulary = {}
word_hist = {}
with open(output_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        fields = line.split()
        token1 = fields[0]
        token2 = fields[1]
        token3 = fields[2]

        tup=(str(token1),str(token2))
        co_results[tup]=float(token3)

        vocabulary[token1] = vocabulary.get(token1, 0) + 1
        vocabulary[token2] = vocabulary.get(token2, 0) + 1

        word_hist = dict(zip(vocabulary.keys(), vocabulary.values()))

graph_builder = ptm.graphml.GraphMLCreator()

#mode is either with_threshold or without_threshod
mode='with_threshold'

if mode is 'without_threshold':
    graph_builder.createGraphML(co_results, vocabulary.keys(), "test1.graphml")

elif mode is 'with_threshold':
    graph_builder.createGraphMLWithThresholdInDictionary(co_results, word_hist, "test.graphml",threshold=10)
    display_limit=30
    graph_builder.summarize_centrality(limit=display_limit)
    title = '동시출현 기반 그래프'
    file_name='test.png'
    graph_builder.plot_graph(title,file=file_name)
