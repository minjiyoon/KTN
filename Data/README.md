OAG dataset is provided in [Heterogeneous Graph Transformer (HGT) github repository](https://github.com/UCLA-DM/pyHGT)
For your convenience, we share their urls to dataset below.

1. Download `graph_CS.pk` from the [google drive](https://drive.google.com/drive/folders/1a85skqsMBwnJ151QpurLFSa9o2ymc_rq)
2. Place `graph_CS.pk` under `DATA/graph_CS/`
3. Download `SeqName_CS_20190919.tsv` from another [google drive](https://drive.google.com/drive/folders/1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz)
4. Place `SeqName_CS_20190919.tsv` under `DATA/graph_CS/`
5. Run `python Data/main.py`
6. If you want to extract Machine Learning (ML) or Computer Network (CN) subgraphs from the OAG CS graph, run `mkdir DATA/graph_ML; python DATA/extract.py`
