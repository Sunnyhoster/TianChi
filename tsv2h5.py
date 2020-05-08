#!/usr/bin/env python

import tables
import base64
import numpy as np
import csv
import sys

csv.field_size_limit(sys.maxsize)

def tsv2h5_main(in_file='../data/processed/video_qa.tsv.0', h5_file='../data/processed/video_qa.h5'):
    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    id2idx_file = open('../data/processed/img_id2idx.txt', 'w')
    id2idx = []
    h5file = tables.open_file(
        h5_file, 'w', '')
    # Verify we can read a tsv
    with open(in_file, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        img_id = []
        img_h = []
        img_w = []
        num_boxes = []
        boxes = []
        features = h5file.create_earray('/', 'features', tables.Atom.from_dtype(np.dtype('Float32')), shape=[0, 36, 2048])
        for idx, item in enumerate(reader):
            item['image_id'] = item['image_id']
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            try:
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                          dtype=np.float32).reshape((item['num_boxes'],-1))
            except:
                item['boxes'] = boxes[-1]
                item['features'] = features[-1]
                print('[E]: {}'.format(item['image_id']))
            img_id.append(item['image_id'])
            img_h.append(item['image_h'])
            img_w.append(item['image_w'])

            num_boxes.append(item['num_boxes'])
            boxes.append(item['boxes'])
            features.append([item['features']])
            if idx % 1000 == 0:
                print('idx: {}'.format(idx))
            id2idx.append('{},{}'.format(idx, item['image_id']))
    id2idx_file.write('\n'.join(id2idx))    
    h5file.create_array('/', 'image_id', img_id, 'features')
    h5file.create_array('/', 'image_h', img_h, 'features')
    h5file.create_array('/', 'image_w', img_w, 'features')
    h5file.create_array('/', 'num_boxes', num_boxes, 'features')
    h5file.create_array('/', 'boxes', boxes, 'features')
    h5file.close()

if __name__ == '__main__':
    tsv2h5_main()
