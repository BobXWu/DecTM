# Code for Discovering Topics in Long-tailed Corpora with Causal Intervention

[ACL2021 Findings](https://aclanthology.org/2021.findings-acl.15.pdf)

## Usage
### 0. Prepare environment

Requirements:

    python==3.6
    tensorflow-gpu==1.13.1
    scipy==1.5.2
    scikit-learn==0.23.2

### 1. Prepare data

Download preprocessed datasets from [Google Drive](https://drive.google.com/drive/folders/1-QMjtmnazqIccBqfdOlxCNhYbGcmcnni) and extract files to the path ./data.

### 2. Run the model

    python main.py --data_dir ./data/{dataset} --output_dir ./output

### 3. Evaluation

topic coherence: [coherence score](https://github.com/dice-group/Palmetto).

topic diversity:

    python utils/TU.py --data_path {path of topic word file}


## Citation

If you are interested in our work, please cite as

    @inproceedings{wu2021discovering,
        title = "Discovering Topics in Long-tailed Corpora with Causal Intervention",
        author = "Wu, Xiaobao  and
        Li, Chunping  and
        Miao, Yishu",
        booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
        month = aug,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.findings-acl.15",
        doi = "10.18653/v1/2021.findings-acl.15",
        pages = "175--185",
    }

## Other related works

[EMNLP2020 Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder](https://github.com/BobXWu/NQTM)

[NLPCC2020 Learning Multilingual Topics with Neural Variational Inference](https://github.com/BobXWu/NMTM)
