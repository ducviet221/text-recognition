import argparse
from PIL import Image
from predictor import Predictor
import yaml
import os
import gdown
import numpy as np
import uuid
import requests

def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)


def download_config(id):
    url = 'https://raw.githubusercontent.com/pbcquoc/vietocr/master/config/{}'.format(id)
    r = requests.get(url)
    config = yaml.safe_load(r.text)
    return config


url_config = {'vgg_transformer':'vgg-transformer.yml','base':'base.yml',}

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        #base_config = download_config(url_config['base'])
        base_config = {}
        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name):
        base_config = download_config(url_config['base'])
        config = download_config(url_config[name])

        base_config.update(config)
        return Cfg(base_config)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':
    main()
