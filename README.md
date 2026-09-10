# Monroe Molecular Embeddings

Turns a SMILES string into a 720-dimensional embedding for the data-limited bioactivity modelling Monroe was built for. A graph transformer pretrained on 81 million PM6 molecules and 1,089 PubChem bioassays, it adds auxiliary edges encoding E/Z and R/S configuration so stereoisomers cannot collapse to identical inputs. Only the encoder is served; the authors' in-context TabPFN head, which carries much of their reported accuracy, needs a labelled support set per task. Conformer generation is seeded for reproducibility.



## Information
### Identifiers
- **Ersilia Identifier:** `eos8gop`
- **Slug:** `monroe-embeddings`

### Domain
- **Task:** `Representation`
- **Subtask:** `Featurization`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Descriptor`, `Embedding`, `Chemical graph model`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `720`
- **Output Consistency:** `Fixed`
- **Interpretation:** 720 features encoding molecular structure and stereochemistry from a graph transformer pretrained on quantum-chemical and bioassay data

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| feat_000 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 0 |
| feat_001 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 1 |
| feat_002 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 2 |
| feat_003 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 3 |
| feat_004 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 4 |
| feat_005 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 5 |
| feat_006 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 6 |
| feat_007 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 7 |
| feat_008 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 8 |
| feat_009 | float |  | Monroe stereochemistry-aware graph-transformer embedding dimension 9 |

_10 of 720 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`

### Resource Consumption


### References
- **Source Code**: [https://github.com/blazejba/Monroe](https://github.com/blazejba/Monroe)
- **Publication**: [https://doi.org/10.48550/arXiv.2608.18982](https://doi.org/10.48550/arXiv.2608.18982)
- **Publication Type:** `Preprint`
- **Publication Year:** `2026`
- **Ersilia Contributor:** [TiagoJanela](https://github.com/TiagoJanela)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos8gop
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos8gop
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
