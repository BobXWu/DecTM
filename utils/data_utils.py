import os
import yaml
import numpy as np
import argparse


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_text(path):
    texts = list()
    with open(path, 'r') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path):
    with open(path, 'w') as file:
        for text in texts:
            file.write(text.strip() + '\n')

