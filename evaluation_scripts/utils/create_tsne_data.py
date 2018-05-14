#!/usr/bin/env python3

import pickle
import csv
import os
import sys

def main(argv):
    if len(argv) != 3:
        print('Usage:\n\tcreate_tsne_data.py <input ACFGs> <output csv path>')
        sys.exit(-1)

    input_acfgs = argv[1]
    output_csv = argv[2]
    tsne_data = pickle.load(open(input_acfgs, 'rb'))

    new_saver = tf.train.Saver({'embs': embs})

    data = []
    for arch in tsne_data.keys():
        for func in tsne_data[arch].keys():
            for filename in tsne_data[arch][func].keys():
                neighbors_tsne, attributes_tsne, u_init_tsne = get_graph_info_mat({'graph': tsne_data[arch][func][filename]})
                emb = sess.run(graph_emb, {neighbors_test: neighbors_tsne, attributes_test: attributes_tsne, u_init_test: u_init_tsne})
                data.append(['{}#{}#{}'.format(func, arch, filename), emb[0]])

    test_keys = []
    test_data = []
    for d in data:
        test_keys.append(d[0])
        test_data.append(d[1])

    embs = tf.Variable(np.array(test_data))
    tf.global_variables_initializer().run()
    new_saver.save(sess, 'experiment/embs')

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['dim{}'.format(x) for x in range(64)] + ['label'])
        for d in data:
            csv_writer.writerow(d[1].tolist() + [d[0]])


if __name__ == '__main__':
    main(sys.argv)
